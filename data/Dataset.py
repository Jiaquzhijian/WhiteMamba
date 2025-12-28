import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import random
from typing import Dict, List, Tuple, Optional


class ImageTextTripletDataset(Dataset):
    """
    图文三元组数据集
    每个样本包含：(锚点图文对，正例图文对，负例图文对)
    """

    def __init__(
            self,
            data_pairs: List[Dict],  # 原始图文对列表 [{"image_path": "...", "text": "...", "label": ...}]
            categories: Optional[Dict] = None,  # 类别信息 {label: [sample_indices]}
            transform=None,  # 图像变换
            tokenizer=None,  # 文本分词器
            num_negatives: int = 1,  # 每个锚点的负例数量
            hard_negative_ratio: float = 0.3,  # 难负例比例
    ):
        """
        初始化数据集

        Args:
            data_pairs: 图文对列表，每个元素包含图像路径和文本
            categories: 类别到样本索引的映射，如果为None则根据文本相似度构建
            transform: 图像预处理变换
            tokenizer: 文本分词器
            num_negatives: 每个锚点对应的负例数量
            hard_negative_ratio: 难负例（语义相近但不同）的比例
        """
        self.data_pairs = data_pairs
        self.transform = transform
        self.tokenizer = tokenizer
        self.num_negatives = num_negatives
        self.hard_negative_ratio = hard_negative_ratio

        # 构建类别索引（如果提供了类别信息）
        if categories is not None:
            self.categories = categories
        else:
            # 如果没有提供类别信息，可以基于文本聚类或随机分组
            self.categories = self._build_semantic_groups()

        # 预计算所有图像的路径和文本
        self.image_paths = [pair["image_path"] for pair in data_pairs]
        self.texts = [pair["text"] for pair in data_pairs]

        # 构建所有可能的样本索引
        self.indices = list(range(len(data_pairs)))

        # 预加载所有图像（可选，适用于小数据集）
        self.preload_images = False
        if len(data_pairs) < 10000:  # 仅当数据集较小时预加载
            self.images = self._preload_images()
            self.preload_images = True

    def _build_semantic_groups(self) -> Dict:
        """构建语义分组（简化的基于文本关键词的分组）"""
        groups = {}
        for idx, pair in enumerate(self.data_pairs):
            # 这里使用简单的基于标签的分组，实际应用中可以使用更复杂的语义分组
            label = pair.get("label", hash(pair["text"]) % 10)  # 简化分组
            if label not in groups:
                groups[label] = []
            groups[label].append(idx)
        return groups

    def _preload_images(self) -> List[Image.Image]:
        """预加载所有图像到内存"""
        images = []
        for path in self.image_paths:
            image = Image.open(path).convert('RGB')
            images.append(image)
        return images

    def _get_image(self, idx: int) -> Image.Image:
        """获取指定索引的图像"""
        if self.preload_images:
            return self.images[idx]
        else:
            return Image.open(self.image_paths[idx]).convert('RGB')

    def _find_positive_pair(self, anchor_idx: int) -> int:
        """为锚点找到正例图文对"""
        anchor_label = None

        # 查找锚点所属的类别
        for label, indices in self.categories.items():
            if anchor_idx in indices:
                anchor_label = label
                break

        if anchor_label is None:
            # 如果找不到类别，随机选择一个不同的样本作为正例（简化处理）
            return random.choice([i for i in self.indices if i != anchor_idx])

        # 从相同类别中选择一个不同的样本作为正例
        same_category = [i for i in self.categories[anchor_label] if i != anchor_idx]

        if len(same_category) > 0:
            return random.choice(same_category)
        else:
            # 如果类别中只有一个样本，随机选择一个不同的样本
            return random.choice([i for i in self.indices if i != anchor_idx])

    def _find_negative_pairs(self, anchor_idx: int, positive_idx: int) -> List[int]:
        """为锚点找到负例图文对"""
        anchor_label = None

        # 查找锚点所属的类别
        for label, indices in self.categories.items():
            if anchor_idx in indices:
                anchor_label = label
                break

        if anchor_label is None:
            # 如果找不到类别，随机选择不同的样本作为负例
            candidates = [i for i in self.indices if i != anchor_idx and i != positive_idx]
            return random.sample(candidates, min(self.num_negatives, len(candidates)))

        # 区分简单负例和难负例
        hard_neg_count = int(self.num_negatives * self.hard_negative_ratio)
        easy_neg_count = self.num_negatives - hard_neg_count

        negatives = []

        # 1. 简单负例：完全不同类别的样本
        if easy_neg_count > 0:
            easy_candidates = []
            for label, indices in self.categories.items():
                if label != anchor_label:
                    easy_candidates.extend(indices)

            if len(easy_candidates) > 0:
                easy_negatives = random.sample(
                    easy_candidates,
                    min(easy_neg_count, len(easy_candidates))
                )
                negatives.extend(easy_negatives)

        # 2. 难负例：相似但不同类别的样本
        if hard_neg_count > 0 and len(negatives) < self.num_negatives:
            # 这里可以根据语义相似度选择难负例
            # 简化：随机选择不同类别的样本
            remaining_needed = self.num_negatives - len(negatives)
            all_candidates = [i for i in self.indices
                              if i != anchor_idx and i != positive_idx and i not in negatives]

            if len(all_candidates) > 0:
                additional_negatives = random.sample(
                    all_candidates,
                    min(remaining_needed, len(all_candidates))
                )
                negatives.extend(additional_negatives)

        return negatives

    def __len__(self) -> int:
        """返回数据集中样本数量"""
        return len(self.data_pairs)

    def __getitem__(self, idx: int) -> Tuple[Dict, Dict, List[Dict]]:
        """
        获取一个三元组样本

        Returns:
            anchor: 锚点图文对
            positive: 正例图文对
            negatives: 负例图文对列表
        """
        # 锚点样本
        anchor_idx = idx
        anchor_image = self._get_image(anchor_idx)
        anchor_text = self.texts[anchor_idx]

        # 查找正例
        positive_idx = self._find_positive_pair(anchor_idx)
        positive_image = self._get_image(positive_idx)
        positive_text = self.texts[positive_idx]

        # 查找负例
        negative_indices = self._find_negative_pairs(anchor_idx, positive_idx)

        # 应用图像变换
        if self.transform:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)

        # 构建返回字典
        anchor = {
            "image": anchor_image,
            "text": anchor_text,
            "image_path": self.image_paths[anchor_idx],
            "index": anchor_idx
        }

        positive = {
            "image": positive_image,
            "text": positive_text,
            "image_path": self.image_paths[positive_idx],
            "index": positive_idx
        }

        negatives = []
        for neg_idx in negative_indices:
            neg_image = self._get_image(neg_idx)
            if self.transform:
                neg_image = self.transform(neg_image)

            neg_pair = {
                "image": neg_image,
                "text": self.texts[neg_idx],
                "image_path": self.image_paths[neg_idx],
                "index": neg_idx
            }
            negatives.append(neg_pair)

        return anchor, positive, negatives

    def get_batch_triplet(self, batch_size: int = 32) -> Tuple[Dict, Dict, List[Dict]]:
        """获取一个批量的三元组（用于训练）"""
        batch_indices = random.sample(self.indices, min(batch_size, len(self.indices)))

        anchors = []
        positives = []
        negatives_list = []

        for idx in batch_indices:
            anchor, positive, negatives = self[idx]
            anchors.append(anchor)
            positives.append(positive)
            negatives_list.append(negatives[0])  # 取第一个负例

        return anchors, positives, negatives_list


class TripletCollator:
    """用于DataLoader的collate_fn，处理三元组数据"""

    def __init__(self, tokenizer=None, max_length=64):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        """
        处理一个batch的三元组数据

        Args:
            batch: 列表，每个元素是(anchor, positive, negatives)

        Returns:
            整理后的batch数据
        """
        anchors, positives, negatives_list = zip(*batch)

        # 提取图像和文本
        anchor_images = [item['image'] for item in anchors]
        anchor_texts = [item['text'] for item in anchors]

        positive_images = [item['image'] for item in positives]
        positive_texts = [item['text'] for item in positives]

        negative_images = [item['image'] for item in negatives_list]
        negative_texts = [item['text'] for item in negatives_list]

        # 堆叠图像张量
        if isinstance(anchor_images[0], torch.Tensor):
            anchor_images = torch.stack(anchor_images)
            positive_images = torch.stack(positive_images)
            negative_images = torch.stack(negative_images)

        # 文本分词（如果提供了分词器）
        if self.tokenizer:
            anchor_texts = self.tokenizer(
                anchor_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            positive_texts = self.tokenizer(
                positive_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            negative_texts = self.tokenizer(
                negative_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )

        # 构建batch数据
        batch_data = {
            'anchor': {
                'image': anchor_images,
                'text': anchor_texts,
                'indices': [item['index'] for item in anchors]
            },
            'positive': {
                'image': positive_images,
                'text': positive_texts,
                'indices': [item['index'] for item in positives]
            },
            'negative': {
                'image': negative_images,
                'text': negative_texts,
                'indices': [item['index'] for item in negatives_list]
            }
        }

        return batch_data


# 使用示例
def create_triplet_dataloader(config: Dict):
    """
    创建三元组数据加载器

    Args:
        config: 配置字典，包含数据路径、变换等信息
    """
    # 1. 加载数据
    with open(config['data_path'], 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 构建类别信息（如果数据中有标签）
    categories = {}
    for idx, item in enumerate(data):
        label = item.get('label')
        if label not in categories:
            categories[label] = []
        categories[label].append(idx)

    # 3. 创建数据集
    dataset = ImageTextTripletDataset(
        data_pairs=data,
        categories=categories,
        transform=config['image_transform'],  # 图像预处理
        tokenizer=None,  # 分词器在collator中处理
        num_negatives=config.get('num_negatives', 1),
        hard_negative_ratio=config.get('hard_negative_ratio', 0.3)
    )

    # 4. 创建collator
    collator = TripletCollator(
        tokenizer=config['text_tokenizer'],  # Chinese-CLIP的分词器
        max_length=config.get('max_length', 64)
    )

    # 5. 创建DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        collate_fn=collator,
        pin_memory=True
    )

    return dataloader


# 简化的使用方式
class SimpleTripletDataset(Dataset):
    """简化版本的三元组数据集"""

    def __init__(self, data_pairs, transform=None):
        self.data_pairs = data_pairs
        self.transform = transform
        self.indices = list(range(len(data_pairs)))

        # 假设每个样本都有标签用于构建正负例
        self.labels = [pair.get('label', 0) for pair in data_pairs]

        # 按标签分组
        self.label_to_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        # 锚点
        anchor_pair = self.data_pairs[idx]
        anchor_label = self.labels[idx]

        # 正例：相同标签的不同样本
        positive_idx = idx
        while positive_idx == idx:
            positive_idx = random.choice(self.label_to_indices[anchor_label])
        positive_pair = self.data_pairs[positive_idx]

        # 负例：不同标签的样本
        negative_label = random.choice([l for l in self.label_to_indices.keys() if l != anchor_label])
        negative_idx = random.choice(self.label_to_indices[negative_label])
        negative_pair = self.data_pairs[negative_idx]

        # 加载图像和应用变换
        anchor_image = Image.open(anchor_pair['image_path']).convert('RGB')
        positive_image = Image.open(positive_pair['image_path']).convert('RGB')
        negative_image = Image.open(negative_pair['image_path']).convert('RGB')

        if self.transform:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)

        return {
            'anchor': {'image': anchor_image, 'text': anchor_pair['text']},
            'positive': {'image': positive_image, 'text': positive_pair['text']},
            'negative': {'image': negative_image, 'text': negative_pair['text']}
        }