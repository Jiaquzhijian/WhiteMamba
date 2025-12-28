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

import torch
from torch.utils.data import Dataset
from PIL import Image


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
