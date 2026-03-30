"""测试训练模块"""
import argparse
from train.main import load_config, main


def test_trainer_with_config():
    """使用配置文件测试训练"""
    print("=== Testing Training with Config File ===")

    # 使用快速测试配置
    config_path = 'config/quick_test.yaml'

    # 模拟命令行参数
    import sys
    sys.argv = ['train.main', '--config', config_path]

    main()


if __name__ == "__main__":
    test_trainer_with_config()
