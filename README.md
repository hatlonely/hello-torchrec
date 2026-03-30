# 广告推荐系统 - 基于 TorchRec 的分布式训练

使用 Meta 的 **TorchRec** 框架构建的分布式广告推荐系统，支持多节点训练超大规模稀疏特征。

## 特性

- 🚀 **分布式 Embedding Sharding**：使用 TorchRec 的 EmbeddingShardingPlanner 实现真正的 embedding 表分片，支持超大规模稀疏特征训练
- 📊 **TorchRec 集成**：使用 TorchRec 的 EmbeddingBagCollection 和 DistributedModelParallel 高效处理大规模稀疏特征
- 💻 **CPU 友好**：完全支持 CPU 训练，无需 GPU 也能快速上手
- 🎯 **端到端流程**：样本生成 → 数据加载 → 分布式训练 → 模型评估
- ⚙️ **YAML 配置**：所有参数通过 YAML 配置文件管理，便于实验和生产部署
- 🔄 **智能分片**：EmbeddingShardingPlanner 自动生成最优的 embedding 表分片方案（table_wise、row_wise、column_wise）
- 🐳 **Docker 支持**：提供完整的 Docker 镜像和部署脚本，支持容器化和多节点部署

## 快速开始

### 1. 环境配置

```bash
# 安装依赖（CPU 版本）
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
```

**依赖版本：**
- PyTorch 2.5.0 (CPU)
- TorchRec 1.0.0 (CPU)
- FBGEMM 1.0.0 (CPU)

### 2. 生成训练数据

```bash
# 生成 10000 条样本
python -m sample.main --num-samples 10000 --output-path data/samples.json
```

**样本包含字段：**
- 用户特征：`user_id`, `user_age`, `user_gender`, `user_city`, `user_interests`
- 广告特征：`ad_id`, `campaign_id`, `advertiser_id`, `ad_category`, `creative_type`
- 上下文特征：`device_type`, `os_type`, `placement_id`, `hour`
- 标签：`label` (0-1 的点击率)

### 3. 训练模型

**使用配置文件训练（推荐）：**

```bash
# 快速测试（1 epoch，小模型）
python -m train.main --config config/quick_test.yaml

# 基础训练（10 epochs）
python -m train.main --config config/base.yaml

# 分布式训练（配置文件中指定 sharding_type）
python -m train.main --config config/distributed.yaml
```

**多进程分布式训练：**

```bash
# 使用 torchrun 启动多进程训练
torchrun --nproc_per_node=2 -m train.main --config config/distributed.yaml

# 多节点训练
torchrun \
  --nnodes=2 \
  --nproc_per_node=4 \
  --master_addr="192.168.1.1" \
  --master_port=29500 \
  --node_rank=0 \
  -m train.main \
  --config config/distributed.yaml
```

**自定义配置：**

```bash
# 1. 复制配置文件模板
cp config/base.yaml config/my_config.yaml

# 2. 修改配置文件中的参数
# 3. 使用自定义配置训练
python -m train.main --config config/my_config.yaml
```

### 4. 测试模型

```bash
# 测试模型功能
python -m model.main

# 测试数据加载
python -m dataloader.main

# 快速训练测试
python -m train.test
```

## Docker 多节点训练

### 构建镜像

```bash
docker build -t hello-torchrec:latest .
```

### 单机多容器训练（推荐）

在同一台机器上启动多个 Docker 容器进行多节点训练：

```bash
# 使用默认配置（3 个节点，每节点 2 个进程）
./multi-node-docker-train.sh

# 自定义配置
./multi-node-docker-train.sh --nodes 2 --nproc 1 --config config/quick_test.yaml

# 清理容器
./multi-node-docker-train.sh --clean
```

**参数说明：**
- `--nodes N`: 节点数量（默认: 3）
- `--nproc N`: 每个节点的进程数（默认: 2）
- `--config FILE`: 配置文件（默认: config/distributed.yaml）
- `--clean`: 清理所有容器和网络

**脚本特性：**
- ✅ 自动创建 Docker 网络实现容器间通信
- ✅ 自动启动多个容器模拟多节点环境
- ✅ 训练完成后自动清理容器
- ✅ 支持自定义节点数量和进程数
- ✅ 实时显示训练日志

**查看日志：**
```bash
# 查看特定容器的日志
docker logs -f torchrec-node0  # master 节点
docker logs -f torchrec-node1  # worker 节点 1
docker logs -f torchrec-node2  # worker 节点 2
```

### 跨机器多节点训练

如果需要在多台物理机器上部署，手动启动每个节点：

**节点 0 (Master)** - IP: 192.168.1.10
```bash
docker run --rm --name torchrec-node0 \
    -v "$(pwd)/outputs:/app/outputs" \
    -p 29500:29500 \
    hello-torchrec:latest \
    torchrun \
    --nnodes=3 \
    --nproc_per_node=2 \
    --master_addr="192.168.1.10" \
    --master_port=29500 \
    --node_rank=0 \
    -m train.main \
    --config config/distributed.yaml
```

**节点 1** - IP: 192.168.1.11
```bash
docker run --rm --name torchrec-node1 \
    -v "$(pwd)/outputs:/app/outputs" \
    hello-torchrec:latest \
    torchrun \
    --nnodes=3 \
    --nproc_per_node=2 \
    --master_addr="192.168.1.10" \
    --master_port=29500 \
    --node_rank=1 \
    -m train.main \
    --config config/distributed.yaml
```

**节点 2** - IP: 192.168.1.12
```bash
docker run --rm --name torchrec-node2 \
    -v "$(pwd)/outputs:/app/outputs" \
    hello-torchrec:latest \
    torchrun \
    --nnodes=3 \
    --nproc_per_node=2 \
    --master_addr="192.168.1.10" \
    --master_port=29500 \
    --node_rank=2 \
    -m train.main \
    --config config/distributed.yaml
```

## 配置文件说明

所有配置文件位于 `config/` 目录下：

| 配置文件 | 说明 | 适用场景 |
|----------|------|----------|
| `quick_test.yaml` | 1 epoch，小模型 | 快速测试和调试 |
| `base.yaml` | 10 epochs，中等模型 | 日常训练 |
| `distributed.yaml` | 20 epochs，大模型，row_wise 分片 | 多进程分布式训练 |

### 配置文件结构

```yaml
# 数据配置
data:
  path: "data/samples.json"

# 模型配置
model:
  embedding:
    dim: 64                    # embedding 维度
    num_embeddings:            # 可选：自定义各特征的基数
      user_id: 10000
      user_age: 5
      # ...

  mlp:
    hidden_dims: [512, 256, 128, 64]  # MLP 隐藏层
    dropout: 0.1                # Dropout 比例
    batch_norm: true            # 是否使用 BatchNorm

# 训练配置
training:
  batch_size: 256
  learning_rate: 0.001
  num_epochs: 10
  device: "cpu"                # cpu 或 cuda

# 分布式配置
distributed:
  sharding_type: "table_wise"  # 当前版本使用 DDP，此配置保留供未来扩展

# 输出配置
output:
  save_path: null              # 模型保存路径
  log_interval: 100            # 日志打印间隔
```

## 核心参数说明

### 分布式训练说明

当前版本使用 **TorchRec 的 EmbeddingShardingPlanner** 实现真正的分布式 embedding 训练：

- **Embedding Sharding**：Embedding 表被分片到不同的进程，每个进程只存储部分 embedding
- **智能规划**：EmbeddingShardingPlanner 自动生成最优的分片方案
- **分片策略**：支持 table_wise、row_wise、column_wise 等分片策略
- **高效通信**：只在需要时进行 All-to-All 通信，比数据并行更高效
- **超大规模支持**：可训练远超单机内存容量的超大规模 embedding 表

**工作原理：**
1. **EmbeddingShardingPlanner** 分析模型结构和拓扑
2. 生成最优的 **ShardingPlan**，指定每个 embedding table 的分片方式
3. **DistributedModelParallel** 根据 plan 将 embedding tables 分片到不同进程
4. 训练时自动处理分片间的通信和梯度同步

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `embedding.dim` | 64 | Embedding 向量维度 |
| `mlp.hidden_dims` | [512, 256, 128, 64] | MLP 隐藏层维度 |
| `mlp.dropout` | 0.1 | Dropout 比例 |
| `mlp.batch_norm` | true | 是否使用 BatchNorm |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `training.batch_size` | 256 | Batch size |
| `training.learning_rate` | 0.001 | 学习率 |
| `training.num_epochs` | 10 | 训练轮数 |
| `training.device` | cpu | 设备类型 (cpu/cuda) |
| `output.log_interval` | 100 | 日志打印间隔 |

## 使用示例

### 示例 1：快速迭代实验

```bash
# 1. 创建实验配置
cp config/quick_test.yaml config/experiment_1.yaml

# 2. 修改配置（比如改变 embedding 维度）
# vim config/experiment_1.yaml

# 3. 运行实验
python -m train.main --config config/experiment_1.yaml
```

### 示例 2：生产环境训练

```bash
# 生成大规模数据
python -m sample.main --num-samples 1000000

# 使用分布式配置训练
torchrun --nproc_per_node=4 -m train.main --config config/distributed.yaml
```

### 示例 3：超大规模多节点训练

```bash
# 在 master 节点
torchrun \
  --nnodes=8 \
  --nproc_per_node=8 \
  --master_addr="10.0.0.1" \
  --master_port=29500 \
  --node_rank=0 \
  -m train.main \
  --config config/distributed.yaml
```

## 目录结构

```
hello-torchrec/
├── config/              # 配置文件目录
│   ├── base.yaml       # 基础训练配置
│   ├── distributed.yaml # 分布式训练配置
│   └── quick_test.yaml # 快速测试配置
├── sample/              # 样本生成模块
│   ├── schema.py       # 数据结构定义
│   ├── generator.py    # 样本生成器
│   └── main.py         # 生成入口
├── dataloader/         # 数据加载模块
│   ├── dataset.py      # PyTorch Dataset
│   ├── loader.py       # 数据加载工具
│   └── main.py         # 测试入口
├── model/              # 模型定义模块
│   ├── config.py       # 模型配置
│   ├── dnn.py          # DNN 模型（使用 torchrec）
│   └── main.py         # 测试入口
├── train/              # 训练模块
│   ├── dist.py         # 分布式初始化
│   ├── trainer.py      # 分布式训练器
│   ├── main.py         # 训练入口（读取 YAML 配置）
│   └── test.py         # 测试脚本
├── data/               # 数据目录
│   └── samples.json    # 训练样本
├── outputs/            # 输出目录
│   └── model.pt        # 保存的模型
├── requirements.txt    # 依赖列表
└── README.md          # 本文件
```

## 架构说明

### 模型架构

```
输入 (13 个稀疏特征)
    ↓
EmbeddingBagCollection (TorchRec)
    ↓
Embedding 拼接 (13 × 64 = 832 维)
    ↓
MLP: 832 → 512 → 256 → 128 → 64 → 1
    ↓
Sigmoid
    ↓
输出: 点击概率 (0-1)
```

### 分布式训练架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    EmbeddingShardingPlanner                     │
│            (自动生成最优 embedding 表分片方案)                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
         ┌───────────────────┴───────────────────┐
         │                                       │
         ▼                                       ▼
┌─────────────────┐                     ┌─────────────────┐
│   Process 0     │                     │   Process 1     │
│  (Shard 0)      │                     │  (Shard 1)      │
│                 │                     │                 │
│ user_id emb     │                     │ ad_id emb       │
│   [0-5000)      │                     │   [0-5000)      │
│                 │                     │                 │
│ ad_category emb │                     │ placement_id emb│
│   [0-50)        │                     │   [0-10)        │
└────────┬────────┘                     └────────┬────────┘
         │                                       │
         └───────────┬───────────────────────────┘
                     │
            DistributedModelParallel
            (根据分片计划自动通信)
                     │
            All-to-All 通信（仅在需要时）
                     │
              合并 embedding 输出
                     │
              MLP 前向传播（每个进程独立）
```

## 性能优化建议

1. **数据加载**：增大 `batch_size` 以提高吞吐量
2. **分片策略**：
   - 特征数量多但基数小 → `table_wise`
   - 特征基数大（亿级）→ `row_wise`
   - Embedding 维度大 → `column_wise`
3. **网络**：多节点训练使用高速网络（InfiniBand、10GbE）
4. **配置调优**：通过修改 YAML 配置快速实验不同参数组合

## 模型保存和加载

### 训练中保存模型

在配置文件中设置 `save_path` 来保存模型：

```yaml
output:
  save_path: "outputs/model.pt"
  log_interval: 100
```

训练完成后会自动保存模型检查点。

### 加载模型继续训练

**分布式训练的模型（embedding sharding）：**

```bash
# 使用相同数量的进程加载
torchrun --nproc_per_node=2 -m train.main --config config/distributed.yaml
```

**注意：** 使用 embedding sharding 的模型保存时包含 ShardedTensor，需要在相同的多进程环境中加载。

### 导出模型用于推理

如需在单机环境中部署推理，可以使用 `export_model_for_inference` 方法：

```python
from train.trainer import DistributedTrainer
from model import DNNModelConfig, EmbeddingConfig

config = DNNModelConfig(...)
trainer = DistributedTrainer(config, ...)
trainer.export_model_for_inference("outputs/inference_model.pt")
```

**注意：** 导出完整模型需要收集所有分片，需要足够的内存。

## 故障排查

**问题 1：找不到配置文件**
```bash
# 确保配置文件路径正确
python -m train.main --config config/base.yaml
```

**问题 2：多进程训练卡住**
```bash
# 检查防火墙和端口设置
# 确保所有节点能访问 master_addr:master_port
```

**问题 3：加载模型时出现 ShardedTensor 错误**
```bash
# 错误：RuntimeError: Need to initialize default process group
# 原因：尝试在单进程中加载分布式训练的模型
# 解决方案1：使用相同数量的进程加载
torchrun --nproc_per_node=2 -m train.main --config config/distributed.yaml

# 解决方案2：训练时使用 export_model_for_inference 导出完整模型
```

## 许可证

MIT License
