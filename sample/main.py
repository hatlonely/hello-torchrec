"""样本生成主入口"""
from sample.schema import FeatureConfig
from sample.config import GeneratorConfig
from sample.generator import SampleGenerator


def main():
    # 特征配置
    feature_config = FeatureConfig(
        num_users=10000,
        num_ads=1000,
        num_campaigns=100,
        num_advertisers=50,
        num_cities=100,
        num_categories=20,
        num_interests=30,
        num_placements=10
    )

    # 生成器配置
    generator_config = GeneratorConfig(
        num_samples=10000,
        output_path="data/samples.json",
        output_format="json",
        random_seed=42,
        click_threshold=0.5
    )

    # 生成样本
    generator = SampleGenerator(feature_config, generator_config)
    samples = generator.generate()
    generator.save(samples)

    # 打印统计信息
    clicks = sum(s['click'] for s in samples)
    ctr = clicks / len(samples)
    print(f"Clicks: {clicks}, CTR: {ctr:.4f}")


if __name__ == "__main__":
    main()
