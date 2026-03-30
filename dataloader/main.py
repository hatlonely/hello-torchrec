"""数据加载测试"""
from dataloader.loader import load_samples
from dataloader.dataset import RecDataset


def test_loader():
    """测试数据加载器"""
    print("=== Testing DataLoader ===")
    samples = load_samples('data/samples.json')
    print(f"Loaded {len(samples)} samples")
    print(f"First sample: {samples[0]}")
    print()


def test_dataset():
    """测试 RecDataset"""
    print("=== Testing RecDataset ===")
    dataset = RecDataset('data/samples.json')

    print(f"Dataset size: {len(dataset)}")

    # 获取单个样本
    sample = dataset[0]
    print(f"Raw sample: {sample}")

    # 获取特征
    features = dataset.get_features(0)
    print(f"Features: {features}")

    # 统计信息
    stats = dataset.get_statistics()
    print(f"Statistics: {stats}")


if __name__ == "__main__":
    test_loader()
    test_dataset()
