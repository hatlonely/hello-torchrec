"""训练入口"""
import argparse
import yaml
import os
from model import DNNModelConfig, EmbeddingConfig
from train.trainer import DistributedTrainer
from train.dist import init_distributed, cleanup_distributed, is_main_process


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def print_config(config: dict, config_path: str):
    """打印配置信息"""
    print("=" * 60)
    print("Training Configuration:")
    print(f"  Config file: {config_path}")
    print(f"  Data path: {config['data']['path']}")
    print(f"  Model:")
    print(f"    Embedding dim: {config['model']['embedding']['dim']}")
    print(f"    MLP hidden dims: {config['model']['mlp']['hidden_dims']}")
    print(f"    Dropout: {config['model']['mlp']['dropout']}")
    print(f"    Batch norm: {config['model']['mlp']['batch_norm']}")
    print(f"  Training:")
    print(f"    Batch size: {config['training']['batch_size']}")
    print(f"    Learning rate: {config['training']['learning_rate']}")
    print(f"    Num epochs: {config['training']['num_epochs']}")
    print(f"    Device: {config['training']['device']}")
    print(f"  Distributed:")
    print(f"    Sharding type: {config['distributed']['sharding_type']}")
    print(f"  Output:")
    print(f"    Save path: {config['output'].get('save_path', 'None')}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Train DNN recommendation model with torchrec')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML file')
    parser.add_argument('--export-full', type=str, default=None,
                       help='Export full model for single-machine inference (path to save)')
    args = parser.parse_args()

    # 加载配置文件
    config = load_config(args.config)

    # 初始化分布式训练
    is_distributed, rank, world_size = init_distributed()

    # 打印配置（只在主进程）
    if is_main_process():
        print_config(config, args.config)
        if is_distributed:
            print(f"  Distributed training: {world_size} processes")
            print("=" * 60)

    # 创建模型配置
    model_config = DNNModelConfig(
        embedding_config=EmbeddingConfig(
            embedding_dim=config['model']['embedding']['dim'],
            num_embeddings=config['model']['embedding'].get('num_embeddings'),
        ),
        hidden_dims=config['model']['mlp']['hidden_dims'],
        dropout=config['model']['mlp']['dropout'],
        batch_norm=config['model']['mlp']['batch_norm'],
        device=config['training']['device']
    )

    # 创建训练器
    trainer = DistributedTrainer(
        config=model_config,
        data_path=config['data']['path'],
        batch_size=config['training']['batch_size'],
        learning_rate=config['training']['learning_rate'],
        num_epochs=config['training']['num_epochs'],
        device=config['training']['device'],
        sharding_type=config['distributed']['sharding_type'],
        log_interval=config['output'].get('log_interval', 100)
    )

    # 开始训练
    trainer.train()

    # 保存模型
    save_path = config['output'].get('save_path')
    if save_path:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        trainer.save_model(save_path)

    # 导出完整模型（如果指定）
    if args.export_full:
        os.makedirs(os.path.dirname(args.export_full), exist_ok=True)
        trainer.export_model_for_inference(args.export_full)

    # 清理分布式环境
    cleanup_distributed()


if __name__ == "__main__":
    main()
