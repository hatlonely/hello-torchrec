"""PyTorch Dataset"""
import json
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple


class RecDataset(Dataset):
    """推荐系统数据集"""

    def __init__(self, path: str):
        self.path = path
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Dict]:
        """加载样本数据"""
        samples = []
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                samples.append(json.loads(line.strip()))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """获取单个样本（原始格式）"""
        return self.samples[idx]

    def get_features(self, idx: int) -> Tuple[Dict, float]:
        """获取特征和标签（用于训练）"""
        sample = self.samples[idx]

        # 稀疏特征（categorical）
        sparse_features = {
            'user_id': sample['user_id'],
            'user_age': sample['user_age'],
            'user_gender': sample['user_gender'],
            'user_city': sample['user_city'],
            'ad_id': sample['ad_id'],
            'campaign_id': sample['campaign_id'],
            'advertiser_id': sample['advertiser_id'],
            'ad_category': sample['ad_category'],
            'creative_type': sample['creative_type'],
            'device_type': sample['device_type'],
            'os_type': sample['os_type'],
            'placement_id': sample['placement_id'],
            'hour': sample['hour'],
        }

        # 列表特征
        list_features = {
            'user_interests': sample['user_interests']
        }

        label = sample['label']

        return {
            'sparse': sparse_features,
            'list': list_features,
            'label': label
        }

    def get_statistics(self) -> Dict:
        """获取数据集统计信息"""
        total = len(self.samples)
        clicks = sum(s['click'] for s in self.samples)
        ctr = clicks / total

        return {
            'total_samples': total,
            'clicks': clicks,
            'ctr': ctr,
            'num_users': len(set(s['user_id'] for s in self.samples)),
            'num_ads': len(set(s['ad_id'] for s in self.samples)),
        }
