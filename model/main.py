"""DNN 模型测试"""
import torch
from model.config import DNNModelConfig, EmbeddingConfig
from model.dnn import DNNModel


def test_dnn():
    """测试使用 torchrec 的 DNN 模型"""
    print("=== Testing DNNModel with torchrec ===")

    # 创建配置
    config = DNNModelConfig(
        embedding_config=EmbeddingConfig(
            embedding_dim=32,
        ),
        hidden_dims=[128, 64, 32],
        dropout=0.1,
        device='cpu'
    )

    # 创建模型
    model = DNNModel(config)
    print(f"Model created successfully")
    print(f"Device: {next(model.parameters()).device}")

    # 创建测试数据
    batch_size = 4
    sparse_features = {
        'user_id': torch.randint(0, 10000, (batch_size,)),
        'user_age': torch.randint(0, 4, (batch_size,)),
        'user_gender': torch.randint(0, 2, (batch_size,)),
        'user_city': torch.randint(0, 100, (batch_size,)),
        'ad_id': torch.randint(0, 1000, (batch_size,)),
        'campaign_id': torch.randint(0, 100, (batch_size,)),
        'advertiser_id': torch.randint(0, 50, (batch_size,)),
        'ad_category': torch.randint(0, 20, (batch_size,)),
        'creative_type': torch.randint(0, 3, (batch_size,)),
        'device_type': torch.randint(0, 3, (batch_size,)),
        'os_type': torch.randint(0, 4, (batch_size,)),
        'placement_id': torch.randint(0, 10, (batch_size,)),
        'hour': torch.randint(0, 24, (batch_size,)),
    }

    # 前向传播
    model.eval()
    with torch.no_grad():
        predictions = model(sparse_features)

    print(f"Input batch size: {batch_size}")
    print(f"Output shape: {predictions.shape}")
    print(f"Output values: {predictions.squeeze().tolist()}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()


if __name__ == "__main__":
    test_dnn()
