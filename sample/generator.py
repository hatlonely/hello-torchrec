"""样本生成器"""
import random
import time
import json
from typing import List, Dict
from .schema import SampleSchema, FeatureConfig
from .config import GeneratorConfig


class SampleGenerator:
    """样本生成器"""

    def __init__(self, feature_config: FeatureConfig, generator_config: GeneratorConfig):
        self.feature_config = feature_config
        self.config = generator_config
        random.seed(self.config.random_seed)

    def _generate_user_features(self) -> Dict:
        """生成用户特征"""
        # 年龄段（按分布）
        user_age = random.choices(
            [1, 2, 3, 4],  # 18-24, 25-34, 35-49, 50+
            weights=self.config.age_distribution
        )[0]

        # 兴趣标签（2-5个）
        num_interests = random.randint(2, 5)
        user_interests = random.sample(
            range(self.feature_config.num_interests),
            num_interests
        )

        return {
            'user_id': random.randint(0, self.feature_config.num_users - 1),
            'user_age': user_age,
            'user_gender': random.randint(0, 1),
            'user_city': random.randint(0, self.feature_config.num_cities - 1),
            'user_interests': user_interests
        }

    def _generate_ad_features(self) -> Dict:
        """生成广告特征"""
        return {
            'ad_id': random.randint(0, self.feature_config.num_ads - 1),
            'campaign_id': random.randint(0, self.feature_config.num_campaigns - 1),
            'advertiser_id': random.randint(0, self.feature_config.num_advertisers - 1),
            'ad_category': random.randint(0, self.feature_config.num_categories - 1),
            'creative_type': random.randint(0, 2)
        }

    def _generate_context_features(self) -> Dict:
        """生成上下文特征"""
        timestamp = int(time.time())
        hour = (timestamp // 3600) % 24

        return {
            'timestamp': timestamp,
            'device_type': random.choices(
                [0, 1, 2],  # mobile, desktop, tablet
                weights=self.config.device_distribution
            )[0],
            'os_type': random.randint(0, 3),
            'placement_id': random.randint(0, self.feature_config.num_placements - 1),
            'hour': hour
        }

    def _generate_label(self, user_features: Dict, ad_features: Dict, context_features: Dict) -> Dict:
        """基于规则生成标签"""
        score = 0.5  # 基准分

        # 规则1: 年龄因素
        if user_features['user_age'] == 1:  # 18-24岁
            score += 0.2
        elif user_features['user_age'] == 4:  # 50岁+
            score -= 0.1

        # 规则2: 性别因素
        if user_features['user_gender'] == 0:  # 女性
            score += 0.05

        # 规则3: 设备因素
        if context_features['device_type'] == 0:  # mobile
            score += 0.15
        elif context_features['device_type'] == 2:  # tablet
            score -= 0.05

        # 规则4: 广告类别与兴趣匹配
        if ad_features['ad_category'] in user_features['user_interests']:
            score += 0.25

        # 规则5: 时间因素（晚间高峰 19-23点）
        if 19 <= context_features['hour'] <= 23:
            score += 0.1

        # 规则6: 广告位效果（位置0-3为黄金位置）
        if context_features['placement_id'] <= 2:
            score += 0.2
        elif context_features['placement_id'] >= 7:
            score -= 0.1

        # 规则7: 创意类型
        if ad_features['creative_type'] == 1:  # video
            score += 0.1
        elif ad_features['creative_type'] == 2:  # carousel
            score += 0.05

        # 添加随机噪声
        noise = random.uniform(-0.1, 0.1)
        score = max(0, min(1, score + noise))

        # 根据阈值决定是否点击
        click = 1 if score >= self.config.click_threshold else 0

        return {'label': round(score, 4), 'click': click}

    def generate_one(self) -> Dict:
        """生成一条样本"""
        user_features = self._generate_user_features()
        ad_features = self._generate_ad_features()
        context_features = self._generate_context_features()
        labels = self._generate_label(user_features, ad_features, context_features)

        return {
            **user_features,
            **ad_features,
            **context_features,
            **labels
        }

    def generate(self) -> List[Dict]:
        """批量生成样本"""
        samples = []
        for _ in range(self.config.num_samples):
            samples.append(self.generate_one())
        return samples

    def save(self, samples: List[Dict]):
        """保存样本到文件（JSON Lines 格式）"""
        output_path = self.config.output_path

        if self.config.output_format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        else:
            raise ValueError(f"Unsupported format: {self.config.output_format}")

        print(f"Generated {len(samples)} samples to {output_path}")
