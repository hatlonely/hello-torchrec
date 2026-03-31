"""训练模块"""
from .dist import init_distributed, cleanup_distributed, get_rank, get_world_size, is_main_process
from .trainer import DistributedTrainer

__all__ = [
    'init_distributed',
    'cleanup_distributed',
    'get_rank',
    'get_world_size',
    'is_main_process',
    'DistributedTrainer',
]
