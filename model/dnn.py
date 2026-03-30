"""DNN 模型"""
import torch
import torch.nn as nn
from typing import Dict, List

from torchrec.modules.embedding_modules import EmbeddingBagCollection, EmbeddingBagConfig
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

from .config import DNNModelConfig


class DNNModel(nn.Module):
    """基于 torchrec 的 DNN 推荐模型"""

    def __init__(self, config: DNNModelConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        # 1. Embedding 层（使用 torchrec）
        self.embedding_bag_collection = self._create_embedding_collection()

        # 计算 embedding 总维度
        self.embedding_dim = config.embedding_config.embedding_dim
        num_features = len(config.embedding_config.num_embeddings)
        total_embedding_dim = self.embedding_dim * num_features

        # 2. MLP 层
        layers = []
        input_dim = total_embedding_dim

        for hidden_dim in config.hidden_dims:
            if config.batch_norm:
                layers.append(nn.BatchNorm1d(input_dim))
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            input_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())

        self.mlp = nn.Sequential(*layers)

        # 初始化权重
        self._init_weights()

    def _create_embedding_collection(self):
        """创建 EmbeddingBagCollection"""
        embedding_tables = []
        emb_config = self.config.embedding_config

        for feature_name, num_embeddings in emb_config.num_embeddings.items():
            eb_config = EmbeddingBagConfig(
                name=feature_name,
                embedding_dim=emb_config.embedding_dim,
                num_embeddings=num_embeddings,
                feature_names=[feature_name],
            )
            embedding_tables.append(eb_config)

        return EmbeddingBagCollection(
            tables=embedding_tables,
            device=torch.device(self.config.device)
        )

    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, sparse_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播

        Args:
            sparse_features: 字典，key 为特征名，value 为 Tensor [batch_size]
                           或 KeyedJaggedTensor

        Returns:
            predictions: [batch_size, 1]
        """
        batch_size = next(iter(sparse_features.values())).size(0)

        # 将输入转换为 KeyedJaggedTensor 格式
        kjt_input = self._to_keyed_jagged_tensor(sparse_features)

        # 获取 embeddings
        pooled_embeddings = self.embedding_bag_collection(kjt_input)

        # 拼接所有 embedding
        embeddings_list = []
        for key in pooled_embeddings.keys():
            emb = pooled_embeddings[key]  # [batch_size, embedding_dim]
            embeddings_list.append(emb)

        concatenated = torch.cat(embeddings_list, dim=1)  # [batch_size, total_dim]

        # 通过 MLP
        predictions = self.mlp(concatenated)  # [batch_size, 1]

        return predictions

    def _to_keyed_jagged_tensor(self, sparse_features: Dict[str, torch.Tensor]):
        """
        将字典格式的特征转换为 KeyedJaggedTensor

        Args:
            sparse_features: {feature_name: Tensor [batch_size]}

        Returns:
            KeyedJaggedTensor
        """
        batch_size = next(iter(sparse_features.values())).size(0)
        keys = []
        values = []
        lengths = []

        # KeyedJaggedTensor expects:
        # - keys: feature names (one per feature, not per batch element)
        # - values: all values concatenated
        # - lengths: how many values per (key, index) pair

        for key in sorted(sparse_features.keys()):
            feature_tensor = sparse_features[key]  # [batch_size]
            keys.append(key)
            values.append(feature_tensor)
            # Each feature has 1 value per sample
            lengths.extend([1] * batch_size)

        # 将所有值拼接
        all_values = torch.cat(values, dim=0)

        return KeyedJaggedTensor.from_lengths_sync(
            keys=keys,
            values=all_values,
            lengths=torch.tensor(lengths, dtype=torch.long)
        )

    def predict(self, sparse_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """预测接口"""
        self.eval()
        with torch.no_grad():
            predictions = self.forward(sparse_features)
        return predictions
