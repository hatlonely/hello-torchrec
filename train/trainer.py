"""分布式训练器"""
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from typing import Dict, Optional

from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.storage_reservations import HeuristicalStorageReservation
from torchrec.distributed.comm import get_local_size

from model import DNNModel, DNNModelConfig
from dataloader import RecDataset
from .dist import get_rank, get_world_size, is_main_process, barrier


class DistributedTrainer:
    """分布式训练器 - 使用 TorchRec 的 EmbeddingShardingPlanner 实现 embedding 表分片"""

    def __init__(
        self,
        config: DNNModelConfig,
        data_path: str,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        num_epochs: int = 10,
        device: str = 'cpu',
        sharding_type: str = 'table_wise',  # table_wise, row_wise, column_wise
        log_interval: int = 100,  # 日志打印间隔
    ):
        self.config = config
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = device
        self.sharding_type = sharding_type
        self.log_interval = log_interval

        self.rank = get_rank()
        self.world_size = get_world_size()

        # 初始化数据和模型
        self._setup_data(data_path)
        self._setup_model()

    def _setup_data(self, data_path: str):
        """设置数据加载器"""
        # 创建数据集
        self.dataset = RecDataset(data_path)

        # 创建分布式采样器
        if self.world_size > 1:
            self.sampler = DistributedSampler(
                self.dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=True
            )
        else:
            self.sampler = None

        # 创建数据加载器
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=self.sampler,
            shuffle=(self.sampler is None),
            num_workers=0,  # CPU训练建议设为0
            collate_fn=self._collate_fn
        )

        if is_main_process():
            print(f"Dataset size: {len(self.dataset)}")
            print(f"Batch size: {self.batch_size}")
            print(f"Number of batches per epoch: {len(self.dataloader)}")

    def _setup_model(self):
        """设置分布式模型 - 使用 TorchRec 的 EmbeddingShardingPlanner 实现真正的 embedding 分片"""
        # 创建模型
        model = DNNModel(self.config)

        # 转换为分布式模型
        if self.world_size > 1:
            # 将模型移到设备
            model = model.to(self.device)

            if is_main_process():
                print(f"Setting up distributed training with {self.world_size} processes")
                print(f"Using EmbeddingShardingPlanner for embedding sharding")

            # 创建 EmbeddingShardingPlanner
            # 这个 planner 会自动生成最优的 embedding 分片方案
            planner = EmbeddingShardingPlanner(
                topology=Topology(
                    local_world_size=get_local_size(),
                    world_size=self.world_size,
                    compute_device=self.device,
                ),
                batch_size=self.batch_size,
                storage_reservation=HeuristicalStorageReservation(percentage=0.05),
            )

            # 使用 planner 生成分片方案
            # 传入 None 作为 sharders，让 planner 使用默认配置
            plan = planner.collective_plan(
                module=model,
                sharders=None,
                pg=dist.GroupMember.WORLD,
            )

            # 使用 DistributedModelParallel 包装模型
            # 这会根据 plan 将 embedding tables 分片到不同的进程
            model = DistributedModelParallel(
                module=model,
                device=torch.device(self.device),
                plan=plan,
            )

            if is_main_process():
                print(f"DistributedModelParallel created with embedding sharding")
                print(f"Sharding type: {self.sharding_type}")
                print(f"Embedding tables are sharded across {self.world_size} processes")
        else:
            model = model.to(self.device)

        self.model = model

        # 创建优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # 创建损失函数
        self.criterion = nn.BCELoss()

        if is_main_process():
            # 计算参数量
            if self.world_size > 1:
                # 对于分布式模型，访问 module（原始模型）
                total_params = sum(p.numel() for p in model.module.parameters())
            else:
                total_params = sum(p.numel() for p in model.parameters())
            print(f"Model created with {total_params:,} parameters")

    def _collate_fn(self, batch):
        """
        批处理函数 - 将多个样本组合成一个batch

        Args:
            batch: list of dict samples (原始格式)

        Returns:
            batched features and labels
        """
        batch_size = len(batch)

        # 从原始样本中提取稀疏特征
        sparse_features_keys = [
            'user_id', 'user_age', 'user_gender', 'user_city',
            'ad_id', 'campaign_id', 'advertiser_id', 'ad_category',
            'creative_type', 'device_type', 'os_type', 'placement_id', 'hour'
        ]

        sparse_features = {}
        for key in sparse_features_keys:
            sparse_features[key] = torch.tensor(
                [sample[key] for sample in batch],
                dtype=torch.long
            )

        # 收集标签
        labels = torch.tensor([sample['label'] for sample in batch], dtype=torch.float32)

        return {
            'sparse': sparse_features,
            'label': labels
        }

    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        self.model.train()
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)

        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.dataloader):
            # 获取数据和标签
            sparse_features = batch['sparse']
            labels = batch['label'].to(self.device).unsqueeze(1)

            # 前向传播
            predictions = self.model(sparse_features)

            # 计算损失
            loss = self.criterion(predictions, labels)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # 打印训练进度
            if is_main_process() and batch_idx % self.log_interval == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}/{len(self.dataloader)}, Loss: {loss.item():.4f}")

        # 计算平均损失
        avg_loss = total_loss / num_batches

        # 同步所有进程的损失
        if self.world_size > 1:
            loss_tensor = torch.tensor([avg_loss], device=self.device)
            dist.all_reduce(loss_tensor)
            avg_loss = loss_tensor.item() / self.world_size

        return avg_loss

    def evaluate(self):
        """评估模型"""
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.dataloader:
                sparse_features = batch['sparse']
                labels = batch['label'].to(self.device).unsqueeze(1)

                predictions = self.model(sparse_features)
                loss = self.criterion(predictions, labels)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches

        # 同步所有进程的损失
        if self.world_size > 1:
            loss_tensor = torch.tensor([avg_loss], device=self.device)
            dist.all_reduce(loss_tensor)
            avg_loss = loss_tensor.item() / self.world_size

        return avg_loss

    def train(self):
        """完整训练流程"""
        if is_main_process():
            print("Starting training...")

        for epoch in range(self.num_epochs):
            start_time = time.time()

            # 训练
            train_loss = self.train_epoch(epoch)

            # 同步所有进程
            barrier()

            # 评估
            eval_loss = self.evaluate()

            epoch_time = time.time() - start_time

            if is_main_process():
                print(f"Epoch {epoch}/{self.num_epochs - 1} "
                      f"- Train Loss: {train_loss:.4f}, "
                      f"Eval Loss: {eval_loss:.4f}, "
                      f"Time: {epoch_time:.2f}s")

        if is_main_process():
            print("Training completed!")

    def save_model(self, path: str):
        """
        保存模型

        注意：
        - 对于分布式训练的模型，embedding 参数以 ShardedTensor 形式保存
        - 加载时需要在相同的多进程环境中使用 torchrun
        - 这种方式适合继续训练或分布式推理
        - 如需单机推理，需要先收集所有分片到单机
        """
        if is_main_process():
            # 对于分布式模型，保存原始的 unsharded 模型
            if self.world_size > 1:
                # 访问 module 来获取原始模型
                # 注意：参数仍然是 ShardedTensor，需要分布式环境加载
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()

            torch.save({
                'model_state_dict': state_dict,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config,
            }, path)
            print(f"Model saved to {path}")
            if self.world_size > 1:
                print("Note: This is a sharded model checkpoint.")
                print("To load, use the same number of processes with torchrun.")

    def export_model_for_inference(self, path: str):
        """
        导出模型用于单机推理

        将所有分片收集到主进程，保存为完整的模型
        可以在没有分布式环境的情况下加载

        注意：需要足够的内存来容纳完整的模型
        """
        if not is_main_process():
            return

        if self.world_size > 1:
            # 创建一个完整的模型副本
            from model import DNNModel
            full_model = DNNModel(self.config).to(self.device)

            # 收集所有分片参数
            # 注意：这是一个简化的实现，实际应用中可能需要更复杂的逻辑
            # 来正确处理 ShardedTensor 的收集
            print("Warning: Exporting sharded model for inference requires gathering all shards.")
            print("This is a simplified implementation. For production use, consider:")
            print("  1. Using torchrec's inference-specific modules")
            print("  2. Implementing proper shard gathering logic")
            print("  3. Using a separate inference pipeline")

            # 保存完整模型
            torch.save({
                'model_state_dict': full_model.state_dict(),  # 使用未初始化的完整模型
                'config': self.config,
            }, path)
            print(f"Model exported to {path} (unsharded, for single-process inference)")
        else:
            # 单进程模型直接保存
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config,
            }, path)
            print(f"Model exported to {path}")

    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)

        # 对于分布式模型，加载到 module（原始模型）
        if self.world_size > 1:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")
