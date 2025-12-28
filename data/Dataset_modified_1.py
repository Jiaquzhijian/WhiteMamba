"""
Dataset utilities for image-text triplet training with a CLIP-style backbone.

Key design goals (for ChineseCLIPBackbone integration):
- Dataset returns *raw* PIL images and raw text strings (no image transform, no tokenization).
- Collator only aggregates into lists; preprocessing is delegated to ChineseCLIPProcessor inside the backbone.
- Supports K negatives per anchor (multi-negative). For legacy code expecting a single negative, set num_negatives=1.

Expected item schema in `data_pairs`:
    {
        "image_path": "/abs/or/rel/path.jpg",
        "text": "some caption",
        "label": "optional_class_label"   # optional but recommended
    }

If `label` is provided, positives are drawn from the same label (excluding self),
negatives are drawn from different labels. If labels are missing, sampling falls back to random.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

def load_go_stone_triplets_binary(
        black_stone_folder: str,
        white_stone_folder: str,
        excel_path: str,
        filename_column: str = "文件名",
        black_text_column: str = "黑子文字",
        white_text_column: str = "白子文字",
        unrecognized_column: str = "无法识别",
        image_extensions: Optional[List[str]] = None,
        seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Load go stone triplets with binary labels (0 for white, 1 for black).

    Creates three types of entries:
    1. Anchor:
       - Option A: white stone image + unrecognized text (label 0)
       - Option B: black stone image + black text (label 1)
    2. Positive: black stone image + black text (label 1)
    3. Negative: white stone image + white text (label 0)

    Args:
        black_stone_folder: Path to folder containing black stone images (test-1)
        white_stone_folder: Path to folder containing white stone images (test-1.0)
        excel_path: Path to Excel file (.xls or .xlsx)
        filename_column: Column name for image filenames in Excel
        black_text_column: Column name for black stone text (黑子文字)
        white_text_column: Column name for white stone text (白子文字)
        unrecognized_column: Column name for unrecognized text (无法识别)
        image_extensions: List of image extensions to look for
        seed: Random seed for reproducibility

    Returns:
        List of dictionaries compatible with ImageTextTripletDataset
    """
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']

    # Validate paths
    for folder, name in [(black_stone_folder, "Black stone folder"),
                         (white_stone_folder, "White stone folder")]:
        if not os.path.exists(folder):
            raise FileNotFoundError(f"{name} not found: {folder}")

    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    # Read Excel file
    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        # Try with different engines
        try:
            df = pd.read_excel(excel_path, engine='openpyxl')
        except:
            try:
                df = pd.read_excel(excel_path, engine='xlrd')
            except:
                raise ValueError(f"Failed to read Excel file: {e}")

    # Check required columns
    required_columns = [filename_column, black_text_column, white_text_column, unrecognized_column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Excel missing required columns: {missing_columns}")

    # Build filename mappings
    def build_file_map(folder_path: str) -> Dict[str, str]:
        """Build mapping from filename stem to full path."""
        file_map = {}
        for ext in image_extensions:
            # Match both lowercase and uppercase extensions
            for img_path in Path(folder_path).glob(f"*{ext}"):
                file_map[img_path.stem] = str(img_path)
            for img_path in Path(folder_path).glob(f"*{ext.upper()}"):
                file_map[img_path.stem] = str(img_path)
        return file_map

    black_file_map = build_file_map(black_stone_folder)
    white_file_map = build_file_map(white_stone_folder)

    # Dedicated RNG
    rng = random.Random(seed)

    data_pairs = []

    for idx, row in df.iterrows():
        filename = str(row[filename_column])
        filename_stem = Path(filename).stem

        # Find matching black stone image
        black_img_path = None
        if filename_stem in black_file_map:
            black_img_path = black_file_map[filename_stem]
        else:
            # Try alternative matching
            alt_matches = [stem for stem in black_file_map.keys()
                           if filename_stem in stem or stem in filename_stem or filename in stem]
            if alt_matches:
                black_img_path = black_file_map[alt_matches[0]]

        # Find matching white stone image
        white_img_path = None
        if filename_stem in white_file_map:
            white_img_path = white_file_map[filename_stem]
        else:
            alt_matches = [stem for stem in white_file_map.keys()
                           if filename_stem in stem or stem in filename_stem or filename in stem]
            if alt_matches:
                white_img_path = white_file_map[alt_matches[0]]

        if not black_img_path:
            print(f"Warning: Black stone image for '{filename}' not found. Skipping row {idx}.")
            continue
        if not white_img_path:
            print(f"Warning: White stone image for '{filename}' not found. Skipping row {idx}.")
            continue

        # Get texts
        anchor_text = str(row[unrecognized_column]) if pd.notna(row[unrecognized_column]) else ""
        positive_text = str(row[black_text_column]) if pd.notna(row[black_text_column]) else ""
        negative_text = str(row[white_text_column]) if pd.notna(row[white_text_column]) else ""

        if not anchor_text or not positive_text or not negative_text:
            print(f"Warning: Empty text for '{filename}'. Skipping row {idx}.")
            continue

        # Create two types of anchors (randomly choose for each row)
        # Type A: white stone + unrecognized text (label 0)
        # Type B: black stone + black text (label 1)
        use_white_anchor = rng.choice([True, False])

        if use_white_anchor:
            # Anchor: white stone + unrecognized text (label 0)
            anchor_img_path = white_img_path
            anchor_label = 0  # white
            anchor_text_to_use = anchor_text
            anchor_type = "white_unrecognized"
        else:
            # Anchor: black stone + black text (label 1)
            anchor_img_path = black_img_path
            anchor_label = 1  # black
            anchor_text_to_use = positive_text  # Use black text
            anchor_type = "black_black"

        # Positive: black stone + black text (label 1)
        positive_img_path = black_img_path
        positive_label = 1  # black
        positive_text_to_use = positive_text

        # Negative: white stone + white text (label 0)
        negative_img_path = white_img_path
        negative_label = 0  # white
        negative_text_to_use = negative_text

        # 修正标签分配策略：确保锚点和正例共享相同的类别标签
        # 这样ImageTextTripletDataset才能正确采样
        # 关键：锚点和正例应该共享相同的标签，负例应该有不同的标签
        if use_white_anchor:
            # 白子锚点的情况：锚点和正例需要共享相同的标签
            # 但白子锚点（无法识别）和黑子正例（黑子文字）本质上是不同的
            # 我们需要一个标签来表示"同一棋局"的概念
            group_label = f"board_{idx}"

            # 1. 锚点条目（白子+无法识别）
            data_pairs.append({
                "image_path": anchor_img_path,
                "text": anchor_text_to_use,
                "label": group_label,  # 共享相同的组标签
                "original_filename": filename,
                "stone_type": "white",
                "text_type": anchor_type,
                "binary_label": anchor_label,
                "row_id": idx,
                "role": "anchor"
            })

            # 2. 正例条目（黑子+黑子文字）
            data_pairs.append({
                "image_path": positive_img_path,
                "text": positive_text_to_use,
                "label": group_label,  # 与锚点相同的组标签
                "original_filename": filename,
                "stone_type": "black",
                "text_type": "black_black",
                "binary_label": positive_label,
                "row_id": idx,
                "role": "positive"
            })

            # 3. 负例条目（白子+白子文字）- 不同的标签
            data_pairs.append({
                "image_path": negative_img_path,
                "text": negative_text_to_use,
                "label": f"negative_{idx}",  # 不同的标签
                "original_filename": filename,
                "stone_type": "white",
                "text_type": "white_white",
                "binary_label": negative_label,
                "row_id": idx,
                "role": "negative"
            })
        else:
            # 黑子锚点的情况：锚点和正例是相同的（都是黑子+黑子文字）
            # 这种情况下，我们需要不同的正例，所以需要特殊处理
            group_label = f"board_{idx}_black"

            # 1. 锚点条目（黑子+黑子文字）
            data_pairs.append({
                "image_path": anchor_img_path,
                "text": anchor_text_to_use,
                "label": group_label,
                "original_filename": filename,
                "stone_type": "black",
                "text_type": anchor_type,
                "binary_label": anchor_label,
                "row_id": idx,
                "role": "anchor"
            })

            # 2. 正例条目（黑子+黑子文字）- 从其他行找一个黑子作为正例
            # 这里简化处理，使用相同的图片和文字，但这不是理想情况
            data_pairs.append({
                "image_path": positive_img_path,
                "text": positive_text_to_use,
                "label": group_label,  # 与锚点相同的标签
                "original_filename": filename,
                "stone_type": "black",
                "text_type": "black_black",
                "binary_label": positive_label,
                "row_id": idx,
                "role": "positive"
            })

            # 3. 负例条目（白子+白子文字）
            data_pairs.append({
                "image_path": negative_img_path,
                "text": negative_text_to_use,
                "label": f"negative_{idx}",  # 不同的标签
                "original_filename": filename,
                "stone_type": "white",
                "text_type": "white_white",
                "binary_label": negative_label,
                "row_id": idx,
                "role": "negative"
            })

    if not data_pairs:
        raise ValueError("No valid triplets created. Check your data and paths.")

    # Summary
    total = len(data_pairs)
    white_anchors = len([d for d in data_pairs if d.get('text_type') == 'white_unrecognized'])
    black_anchors = len([d for d in data_pairs if d.get('text_type') == 'black_black'])
    positives = len([d for d in data_pairs if d.get('role') == 'positive'])
    negatives = len([d for d in data_pairs if d.get('role') == 'negative'])

    print(f"Created {total} total entries from {len(df)} Excel rows")
    print(f"  White anchors (white+unrecognized): {white_anchors}")
    print(f"  Black anchors (black+black_text): {black_anchors}")
    print(f"  Positives (black+black_text): {positives}")
    print(f"  Negatives (white+white_text): {negatives}")

    return data_pairs


def load_go_stone_triplets_simplified(
        black_stone_folder: str,
        white_stone_folder: str,
        excel_path: str,
        filename_column: str = "文件名",
        black_text_column: str = "黑子文字",
        white_text_column: str = "白子文字",
        unrecognized_column: str = "无法识别",
        image_extensions: Optional[List[str]] = None,
        use_white_anchor_only: bool = False,
        seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    简化的围棋棋子数据集加载函数，修复标签分配问题
    """
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']

    # 验证路径
    for folder in [black_stone_folder, white_stone_folder]:
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Folder not found: {folder}")
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    # 读取Excel
    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        try:
            df = pd.read_excel(excel_path, engine='openpyxl')
        except:
            try:
                df = pd.read_excel(excel_path, engine='xlrd')
            except:
                raise ValueError(f"Failed to read Excel file: {e}")

    required_columns = [filename_column, black_text_column, white_text_column, unrecognized_column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Excel missing required columns: {missing_columns}")

    # 构建文件映射
    def build_file_map(folder_path: str) -> Dict[str, str]:
        file_map = {}
        for ext in image_extensions:
            for pattern in [f"*{ext}", f"*{ext.upper()}"]:
                for img_path in Path(folder_path).glob(pattern):
                    file_map[img_path.stem] = str(img_path)
        return file_map

    black_file_map = build_file_map(black_stone_folder)
    white_file_map = build_file_map(white_stone_folder)

    rng = random.Random(seed)
    data_pairs = []

    for idx, row in df.iterrows():
        filename = str(row[filename_column])
        filename_stem = Path(filename).stem

        # 查找图片
        black_img_path = None
        if filename_stem in black_file_map:
            black_img_path = black_file_map[filename_stem]
        else:
            # 尝试模糊匹配
            for stem in black_file_map.keys():
                if stem in filename_stem or filename_stem in stem:
                    black_img_path = black_file_map[stem]
                    break

        white_img_path = None
        if filename_stem in white_file_map:
            white_img_path = white_file_map[filename_stem]
        else:
            for stem in white_file_map.keys():
                if stem in filename_stem or filename_stem in stem:
                    white_img_path = white_file_map[stem]
                    break

        if not black_img_path or not white_img_path:
            continue

        # 获取文本
        anchor_text = str(row[unrecognized_column]) if pd.notna(row[unrecognized_column]) else ""
        positive_text = str(row[black_text_column]) if pd.notna(row[black_text_column]) else ""
        negative_text = str(row[white_text_column]) if pd.notna(row[white_text_column]) else ""

        if not anchor_text or not positive_text or not negative_text:
            continue

        # 确定锚点类型
        if use_white_anchor_only:
            is_white_anchor = True
        else:
            is_white_anchor = rng.choice([True, False])

        # 为每个棋局创建一个唯一的组ID
        # 这个组ID确保锚点和正例可以被正确匹配
        group_id = f"board_{idx}"

        # 1. 锚点
        if is_white_anchor:
            # 白子锚点：白子图片 + 无法识别文字
            data_pairs.append({
                "image_path": white_img_path,
                "text": anchor_text,
                "label": group_id,  # 与正例共享相同的组ID
                "original_filename": filename,
                "stone_type": "white",
                "text_type": "unrecognized",
                "row_id": idx,
                "role": "anchor_white"
            })
        else:
            # 黑子锚点：黑子图片 + 黑子文字
            data_pairs.append({
                "image_path": black_img_path,
                "text": positive_text,
                "label": group_id,  # 与正例共享相同的组ID
                "original_filename": filename,
                "stone_type": "black",
                "text_type": "black",
                "row_id": idx,
                "role": "anchor_black"
            })

        # 2. 正例：黑子图片 + 黑子文字
        data_pairs.append({
            "image_path": black_img_path,
            "text": positive_text,
            "label": group_id,  # 与锚点相同的组ID
            "original_filename": filename,
            "stone_type": "black",
            "text_type": "black",
            "row_id": idx,
            "role": "positive"
        })

        # 3. 负例：白子图片 + 白子文字（使用不同的组ID）
        data_pairs.append({
            "image_path": white_img_path,
            "text": negative_text,
            "label": f"negative_{idx}",  # 不同的组ID，确保被采样为负例
            "original_filename": filename,
            "stone_type": "white",
            "text_type": "white",
            "row_id": idx,
            "role": "negative"
        })

    return data_pairs


def _load_pairs_from_json(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"JSON must contain a list of dicts, got: {type(data)}")
    return data


def _build_categories(data_pairs: Sequence[Dict[str, Any]], label_key: str = "label") -> Dict[Any, List[int]]:
    cats: Dict[Any, List[int]] = {}
    for i, item in enumerate(data_pairs):
        lbl = item.get(label_key, None)
        if lbl is None:
            continue
        cats.setdefault(lbl, []).append(i)
    return cats


class ImageTextTripletDataset(Dataset):
    """
    Triplet (or multi-negative) dataset for image-text pairs.

    Returns one sample as:
        {
            "anchor":   {"image": PIL.Image, "text": str, "index": int, "label": Any},
            "positive": {"image": PIL.Image, "text": str, "index": int, "label": Any},
            "negatives": [
                {"image": PIL.Image, "text": str, "index": int, "label": Any},
                ...
            ]
        }

    Notes:
    - No image transforms and no tokenization are applied here (to keep alignment with CLIP processor).
    - Paths are resolved relative to `root_dir` if provided.
    """

    def __init__(
        self,
        data_pairs: Optional[List[Dict[str, Any]]] = None,
        json_path: Optional[str] = None,
        root_dir: Optional[str] = None,
        categories: Optional[Dict[Any, List[int]]] = None,
        num_negatives: int = 1,
        label_key: str = "label",
        seed: int = 42,
    ) -> None:
        if data_pairs is None:
            if json_path is None:
                raise ValueError("Provide either `data_pairs` or `json_path`.")
            data_pairs = _load_pairs_from_json(json_path)

        if not isinstance(data_pairs, list) or len(data_pairs) == 0:
            raise ValueError("`data_pairs` must be a non-empty list of dicts.")

        self.data_pairs = data_pairs
        self.root_dir = root_dir
        self.num_negatives = max(1, int(num_negatives))
        self.label_key = label_key

        # Categories: label -> indices. If not provided, build from label_key if possible.
        self.categories = categories if categories is not None else _build_categories(self.data_pairs, label_key=label_key)

        # For fast sampling
        self.indices = list(range(len(self.data_pairs)))

        # Dedicated RNG for reproducibility (does not affect global random state)
        self._rng = random.Random(seed)

        # Precompute index->label map (may be None)
        self._labels: List[Any] = [item.get(label_key, None) for item in self.data_pairs]

    def __len__(self) -> int:
        return len(self.data_pairs)

    def _resolve_path(self, p: str) -> str:
        if self.root_dir is None:
            return p
        # If p is already absolute, keep it; else join with root_dir
        if os.path.isabs(p):
            return p
        return os.path.join(self.root_dir, p)

    def _sample_positive(self, anchor_idx: int) -> int:
        """Sample a positive index for anchor."""
        anchor_label = self._labels[anchor_idx]
        if anchor_label is not None and anchor_label in self.categories:
            candidates = [i for i in self.categories[anchor_label] if i != anchor_idx]
            if len(candidates) > 0:
                return self._rng.choice(candidates)

        # Fallback: random different index
        candidates = [i for i in self.indices if i != anchor_idx]
        return self._rng.choice(candidates)

    def _sample_negatives(self, anchor_idx: int, positive_idx: int) -> List[int]:
        """Sample K negative indices for anchor."""
        anchor_label = self._labels[anchor_idx]

        if anchor_label is not None and len(self.categories) > 1:
            # negatives from different labels
            candidates = [i for i in self.indices if i != anchor_idx and i != positive_idx and self._labels[i] != anchor_label]
        else:
            # Fallback: any other sample
            candidates = [i for i in self.indices if i != anchor_idx and i != positive_idx]

        if len(candidates) == 0:
            # Degenerate case: dataset too small
            return [positive_idx] * self.num_negatives

        if len(candidates) >= self.num_negatives:
            return self._rng.sample(candidates, self.num_negatives)
        # Not enough unique candidates: sample with replacement
        return [self._rng.choice(candidates) for _ in range(self.num_negatives)]

    def _load_pair(self, idx: int) -> Dict[str, Any]:
        item = self.data_pairs[idx]
        img_path = item.get("image_path", None)
        txt = item.get("text", None)
        if img_path is None or txt is None:
            raise KeyError(f"Each item must contain 'image_path' and 'text'. Problem at idx={idx}: {item}")

        img_path = self._resolve_path(str(img_path))
        image = Image.open(img_path).convert("RGB")
        label = item.get(self.label_key, None)

        return {"image": image, "text": str(txt), "index": idx, "label": label}

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        anchor_idx = idx
        positive_idx = self._sample_positive(anchor_idx)
        negative_indices = self._sample_negatives(anchor_idx, positive_idx)

        anchor = self._load_pair(anchor_idx)
        positive = self._load_pair(positive_idx)
        negatives = [self._load_pair(nidx) for nidx in negative_indices]

        return {"anchor": anchor, "positive": positive, "negatives": negatives}

class GoStoneTripletDataset(ImageTextTripletDataset):
    """
    Custom dataset specifically for go stone recognition.
    """

    def __init__(
            self,
            black_stone_folder: str,
            white_stone_folder: str,
            excel_path: str,
            filename_column: str = "文件名",
            black_text_column: str = "黑子文字",
            white_text_column: str = "白子文字",
            unrecognized_column: str = "无法识别",
            root_dir: Optional[str] = None,
            num_negatives: int = 1,
            seed: int = 42,
            use_white_anchor_only: bool = False,
    ):
        # Load data pairs using our function
        data_pairs = load_go_stone_triplets_simplified(
            black_stone_folder=black_stone_folder,
            white_stone_folder=white_stone_folder,
            excel_path=excel_path,
            filename_column=filename_column,
            black_text_column=black_text_column,
            white_text_column=white_text_column,
            unrecognized_column=unrecognized_column,
            use_white_anchor_only=use_white_anchor_only,
            seed=seed,
        )

        # Initialize parent class
        super().__init__(
            data_pairs=data_pairs,
            root_dir=root_dir,
            num_negatives=num_negatives,
            seed=seed,
        )


# Example usage:
def example_usage():
    """Example of how to use the go stone dataset functions."""

    print("=== Method 1: Using the standalone function ===")
    # Method 1: Use the standalone function
    data_pairs = load_go_stone_triplets_binary(
        black_stone_folder="test-1",
        white_stone_folder="test-1.0",
        excel_path="test.xls",
        filename_column="文件名",
        black_text_column="黑子文字",
        white_text_column="白子文字",
        unrecognized_column="无法识别"
    )

    # Create dataset
    dataset1 = ImageTextTripletDataset(
        data_pairs=data_pairs,
        root_dir=None,  # paths are already absolute
        num_negatives=1,
        label_key="label"
    )

    # Test one sample
    sample1 = dataset1[0]
    print(f"Anchor text: {sample1['anchor']['text']}")
    print(f"Positive text: {sample1['positive']['text']}")
    print(f"Negative text: {sample1['negatives'][0]['text']}")

    print("\n=== Method 2: Using the custom dataset class ===")
    # Method 2: Use the custom dataset class
    dataset2 = GoStoneTripletDataset(
        black_stone_folder="test-1",
        white_stone_folder="test-1.0",
        excel_path="test.xls",
        filename_column="文件名",
        black_text_column="黑子文字",
        white_text_column="白子文字",
        unrecognized_column="无法识别",
        num_negatives=1,
        use_white_anchor_only=True  # Only use white stones as anchors
    )

    # Test one sample
    sample2 = dataset2[0]
    print(f"Anchor text: {sample2['anchor']['text']}")
    print(f"Positive text: {sample2['positive']['text']}")
    print(f"Negative text: {sample2['negatives'][0]['text']}")

    return dataset1, dataset2


class TripletCollator:
    """
    Collator that preserves raw PIL+str for a CLIP-style backbone.

    Output format:
        {
          "anchor":   {"image": List[PIL], "text": List[str], "index": LongTensor, "label": List[Any]},
          "positive": {"image": List[PIL], "text": List[str], "index": LongTensor, "label": List[Any]},
          "negatives": {
              "image": List[List[PIL]] or List[PIL] (if K==1),
              "text":  List[List[str]]  or List[str]  (if K==1),
              "index": LongTensor of shape (B,K) or (B,),
              "label": List[List[Any]]  or List[Any]
          },
          "num_negatives": int
        }

    If all samples have the same K negatives, K is inferred from the first sample.
    """

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(batch) == 0:
            raise ValueError("Empty batch.")

        B = len(batch)
        K = len(batch[0]["negatives"])
        for item in batch:
            if len(item["negatives"]) != K:
                raise ValueError("All samples in a batch must have the same number of negatives.")

        def collect_pair(key: str) -> Dict[str, Any]:
            images = [item[key]["image"] for item in batch]
            texts = [item[key]["text"] for item in batch]
            indices = torch.tensor([int(item[key]["index"]) for item in batch], dtype=torch.long)
            labels = [item[key].get("label", None) for item in batch]
            return {"image": images, "text": texts, "index": indices, "label": labels}

        anchor = collect_pair("anchor")
        positive = collect_pair("positive")

        # Negatives: (B, K)
        neg_images_2d: List[List[Image.Image]] = []
        neg_texts_2d: List[List[str]] = []
        neg_indices_2d: List[List[int]] = []
        neg_labels_2d: List[List[Any]] = []

        for item in batch:
            neg_images_2d.append([n["image"] for n in item["negatives"]])
            neg_texts_2d.append([n["text"] for n in item["negatives"]])
            neg_indices_2d.append([int(n["index"]) for n in item["negatives"]])
            neg_labels_2d.append([n.get("label", None) for n in item["negatives"]])

        neg_idx = torch.tensor(neg_indices_2d, dtype=torch.long)  # (B,K)

        negatives: Dict[str, Any]
        if K == 1:
            # Flatten to keep legacy compatibility
            negatives = {
                "image": [row[0] for row in neg_images_2d],
                "text": [row[0] for row in neg_texts_2d],
                "index": neg_idx[:, 0],
                "label": [row[0] for row in neg_labels_2d],
            }
        else:
            negatives = {
                "image": neg_images_2d,
                "text": neg_texts_2d,
                "index": neg_idx,
                "label": neg_labels_2d,
            }

        return {"anchor": anchor, "positive": positive, "negatives": negatives, "num_negatives": K}

if __name__ == "__main__":
    # Test the functions
    try:
        dataset1, dataset2 = example_usage()
        print("\nDataset created successfully!")
        print(f"Dataset 1 size: {len(dataset1)}")
        print(f"Dataset 2 size: {len(dataset2)}")
    except Exception as e:
        print(f"Error: {e}")