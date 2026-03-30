"""数据加载器"""
import json
from typing import List, Dict, Iterator
from dataclasses import dataclass


@dataclass
class DataLoader:
    """数据加载器"""
    path: str

    def load_all(self) -> List[Dict]:
        """加载所有数据"""
        samples = []
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                samples.append(json.loads(line.strip()))
        return samples

    def load_iter(self) -> Iterator[Dict]:
        """流式加载数据（迭代器）"""
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                yield json.loads(line.strip())

    def load_n(self, n: int) -> List[Dict]:
        """加载前n条数据"""
        samples = []
        with open(self.path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                samples.append(json.loads(line.strip()))
        return samples


def load_samples(path: str) -> List[Dict]:
    """便捷函数：加载所有样本"""
    loader = DataLoader(path)
    return loader.load_all()
