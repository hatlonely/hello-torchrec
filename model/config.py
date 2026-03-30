"""模型配置"""
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class EmbeddingConfig:
    """Embedding 配置"""
    embedding_dim: int = 64  # embedding 维度
    num_embeddings: Dict[str, int] = None  # 各特征的数量

    def __post_init__(self):
        if self.num_embeddings is None:
            # 默认特征基数
            self.num_embeddings = {
                'user_id': 10000,
                'user_age': 5,  # 0-4，需要5个embedding
                'user_gender': 2,
                'user_city': 100,
                'ad_id': 1000,
                'campaign_id': 100,
                'advertiser_id': 50,
                'ad_category': 20,
                'creative_type': 3,
                'device_type': 3,
                'os_type': 4,
                'placement_id': 10,
                'hour': 24,
            }


@dataclass
class DNNModelConfig:
    """DNN 模型配置"""
    embedding_config: EmbeddingConfig = None

    # MLP 层配置
    hidden_dims: List[int] = None

    # 其他配置
    dropout: float = 0.1
    batch_norm: bool = True
    activation: str = 'relu'

    # 设备配置
    device: str = 'cpu'  # 支持 cpu 和 cuda

    def __post_init__(self):
        if self.embedding_config is None:
            self.embedding_config = EmbeddingConfig()
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256, 128, 64]
