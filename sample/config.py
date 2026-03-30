"""样本生成配置"""
from dataclasses import dataclass


@dataclass
class GeneratorConfig:
    """生成器配置"""
    num_samples: int = 10000  # 生成样本数量
    output_path: str = "dataset/samples.json"  # 输出路径
    output_format: str = "json"  # 输出格式
    random_seed: int = 42  # 随机种子
    click_threshold: float = 0.5  # 点击判定阈值

    # 数据分布配置
    age_distribution: list = None  # 年龄分布 [18-24, 25-34, 35-49, 50+]
    device_distribution: list = None  # 设备分布 [mobile, desktop, tablet]

    def __post_init__(self):
        if self.age_distribution is None:
            self.age_distribution = [0.25, 0.35, 0.25, 0.15]
        if self.device_distribution is None:
            self.device_distribution = [0.7, 0.2, 0.1]
