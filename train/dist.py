"""分布式训练初始化"""
import os
import torch
import torch.distributed as dist


def init_distributed():
    """
    初始化分布式训练环境

    环境变量:
        RANK: 当前进程的rank
        WORLD_SIZE: 总进程数
        MASTER_ADDR: master节点地址
        MASTER_PORT: master节点端口
        LOCAL_RANK: 本地rank (用于单机多卡)

    Returns:
        is_distributed: 是否为分布式训练
        rank: 当前进程rank
        world_size: 总进程数
    """
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    if world_size > 1:
        # 初始化进程组
        backend = 'gloo'  # CPU训练使用gloo，GPU训练使用nccl
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size
        )

        # 设置当前设备
        torch.cuda.set_device(local_rank) if torch.cuda.is_available() else None

        print(f"Initialized distributed training: rank={rank}, world_size={world_size}, backend={backend}")
        return True, rank, world_size

    return False, rank, 1


def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank():
    """获取当前rank"""
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size():
    """获取world size"""
    return dist.get_world_size() if dist.is_initialized() else 1


def is_main_process():
    """是否为主进程 (rank 0)"""
    return get_rank() == 0


def barrier():
    """同步所有进程"""
    if dist.is_initialized():
        dist.barrier()
