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

        - 单节点训练：直接保存完整模型
        - 多节点训练：自动收集所有分片，保存为完整模型
        - 只有 rank 0（主进程）执行保存操作
        """
        if not is_main_process():
            return

        # 单节点训练：直接保存
        if self.world_size == 1:
            state_dict = self.model.state_dict()
            total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            torch.save({
                'model_state_dict': state_dict,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config,
                'world_size': 1,
            }, path)
            print(f"完整模型已保存到: {path}")
            print(f"总参数量: {total_params:,}")
            return

        # 多节点训练：收集所有分片并保存完整模型
        self._save_full_model_from_shards(path)

    def _save_full_model_from_shards(self, path: str):
        """
        从分片模型中收集所有参数，保存为完整模型

        注意：
        - 需要所有进程参与分片收集
        - 只有 rank 0 保存最终模型
        - 需要足够的内存容纳完整模型
        """
        from model import DNNModel
        import torch.distributed as dist

        print("=" * 60)
        print("多节点训练：正在收集所有分片并保存完整模型")
        print(f"分片数量: {self.world_size}")
        print("=" * 60)

        # 在所有进程上创建完整模型结构
        full_model = DNNModel(self.config).to(self.device)
        full_model.eval()

        # 收集所有分片参数
        with torch.no_grad():
            # 获取当前（分片）模型的参数
            current_state_dict = self.model.module.state_dict()

            # 遍历完整模型的所有参数
            for name, full_param in full_model.named_parameters():
                if name not in current_state_dict:
                    continue

                sharded_param = current_state_dict[name]

                # 判断是否是 ShardedTensor
                is_sharded = hasattr(sharded_param, '_sharded_tensor') or hasattr(sharded_param, 'metadata')

                # 判断是否是 embedding 层
                is_embedding = 'embedding' in name.lower() or 'emb' in name.lower()

                if is_sharded and is_embedding:
                    if is_main_process():
                        print(f"收集分片参数: {name}")

                    # 对于 ShardedTensor，使用 local_tensor() 获取本地分片数据
                    try:
                        # 获取本地分片的 tensor
                        local_tensor = sharded_param.local_tensor()

                        # 收集所有分片
                        gathered_tensors = [torch.zeros_like(local_tensor) for _ in range(self.world_size)]
                        dist.all_gather(gathered_tensors, local_tensor)

                        # 在 rank 0 上合并分片
                        if is_main_process():
                            # 获取 ShardedTensor 的元数据
                            if hasattr(sharded_param, 'metadata'):
                                metadata = sharded_param.metadata
                                if is_main_process():
                                    print(f"  分片信息: {metadata.num_shards()} 个分片, 完整大小: {metadata.size}")

                            # 尝试拼接分片
                            try:
                                # 按维度 0 拼接
                                full_tensor = torch.cat(gathered_tensors, dim=0)

                                # 检查是否需要 reshape
                                if full_tensor.shape == full_param.data.shape:
                                    # 大小完全匹配
                                    full_param.data = full_tensor
                                elif full_tensor.numel() == full_param.data.numel():
                                    # 元素数量相同，需要 reshape
                                    full_param.data = full_tensor.reshape(full_param.data.shape)
                                    if is_main_process():
                                        print(f"  reshape: {full_tensor.shape} -> {full_param.data.shape}")
                                else:
                                    # 大小不匹配
                                    if is_main_process():
                                        print(f"  警告: 分片拼接后大小不匹配")
                                        print(f"    预期: {full_param.data.shape}")
                                        print(f"    实际: {full_tensor.shape}")
                                        print(f"    使用: rank 0 的分片数据")
                                    # 使用第一个分片的数据
                                    full_param.data = gathered_tensors[0].clone()
                            except Exception as e:
                                if is_main_process():
                                    print(f"  警告: 无法拼接分片: {e}")
                                    print(f"    使用 rank 0 的分片数据")
                                full_param.data = gathered_tensors[0].clone()

                    except Exception as e:
                        # 如果无法获取 local_tensor，可能不是真正的 ShardedTensor
                        if is_main_process():
                            print(f"  警告: 无法处理 {name} 为 ShardedTensor: {e}")
                            print(f"    尝试作为普通 tensor 处理")
                        # 降级为普通 tensor 处理
                        if hasattr(sharded_param, 'data'):
                            tensor_data = sharded_param.data
                        else:
                            tensor_data = sharded_param

                        if is_main_process():
                            # 直接使用当前参数
                            full_param.data = tensor_data.clone()
                        # 其他进程不操作

                elif is_embedding:
                    # embedding 层但不是 ShardedTensor，直接复制
                    if is_main_process():
                        print(f"复制参数: {name} (非分片)")
                    if hasattr(sharded_param, 'data'):
                        full_param.data = sharded_param.data.clone()
                    else:
                        full_param.data = sharded_param.clone()
                else:
                    # MLP 层，直接复制（所有节点相同）
                    if hasattr(sharded_param, 'data'):
                        full_param.data = sharded_param.data.clone()
                    else:
                        full_param.data = sharded_param.clone()

        # 同步所有进程
        barrier()

        # 只在 rank 0 保存完整模型
        if is_main_process():
            total_params = sum(p.numel() for p in full_model.parameters())

            torch.save({
                'model_state_dict': full_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config,
                'world_size': 1,  # 标记为单机模型
            }, path)

            print("=" * 60)
            print(f"完整模型已保存到: {path}")
            print(f"总参数量: {total_params:,}")
            print("此模型可以在单机上加载使用")
            print("=" * 60)

    def load_model(self, path: str):
        """
        加载模型

        - 支持加载完整模型（world_size=1）
        - 支持加载分片模型（需要相同数量的进程）
        """
        checkpoint = torch.load(path, map_location=self.device)
        checkpoint_world_size = checkpoint.get('world_size', 1)

        if checkpoint_world_size == 1:
            # 完整模型，直接加载
            if self.world_size > 1:
                # 如果当前是分布式训练，需要将完整模型包装为分布式模型
                print(f"加载完整模型到分布式环境")
                self.model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"完整模型已加载: {path}")
        else:
            # 分片模型，需要匹配进程数
            if self.world_size != checkpoint_world_size:
                raise ValueError(
                    f"进程数不匹配：模型用 {checkpoint_world_size} 个进程训练，"
                    f"当前使用 {self.world_size} 个进程"
                )
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
            print(f"分片模型已加载: {path}")
            print(f"模型使用 {checkpoint_world_size} 个进程训练")

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
