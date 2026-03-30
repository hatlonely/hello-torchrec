"""数据字段定义"""
from dataclasses import dataclass
from typing import List


@dataclass
class SampleSchema:
    """样本字段定义"""
    # 用户特征
    user_id: int
    user_age: int  # 年龄段: 1:18-24, 2:25-34, 3:35-49, 4:50+
    user_gender: int  # 0: female, 1: male
    user_city: int
    user_interests: List[int]  # 兴趣标签列表

    # 广告特征
    ad_id: int
    campaign_id: int
    advertiser_id: int
    ad_category: int
    creative_type: int  # 0: image, 1: video, 2: carousel

    # 上下文特征
    timestamp: int
    device_type: int  # 0: mobile, 1: desktop, 2: tablet
    os_type: int  # 0: iOS, 1: Android, 2: Windows, 3: macOS
    placement_id: int
    hour: int  # 0-23

    # 标签
    label: float  # 点击率 0-1
    click: int  # 是否点击 0/1


@dataclass
class FeatureConfig:
    """特征配置（基数）"""
    num_users: int = 10000
    num_ads: int = 1000
    num_campaigns: int = 100
    num_advertisers: int = 50
    num_cities: int = 100
    num_categories: int = 20
    num_interests: int = 30
    num_placements: int = 10
