"""
对抗网络抖动测试
在代理转发链路中人为加入随机延迟或包大小扰动
观察并评估AI模型识别准确率的抗干扰能力
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import random
import time
import logging
from collections import defaultdict
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns

from model import TrafficCNN
from data_preprocessing import ISCXDataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JitterSimulator:
    """网络抖动模拟器"""

    def __init__(self, jitter_type='delay', jitter_level=0.1):
        """
        Args:
            jitter_type: 抖动类型 ('delay', 'padding', 'mixed')
            jitter_level: 抖动强度 (0.0-1.0)
        """
        self.jitter_type = jitter_type
        self.jitter_level = jitter_level

    def add_delay_jitter(self, features: np.ndarray, delay_range=(10, 100)) -> np.ndarray:
        """添加延迟抖动（模拟网络延迟变化）

        Args:
            features: 原始特征
            delay_range: 延迟范围(毫秒)

        Returns:
            添加延迟抖动后的特征
        """
        jittered = features.copy()
        num_packets = len(jittered)

        for i in range(num_packets):
            if jittered[i] > 0 and random.random() < self.jitter_level:
                delay_ms = random.uniform(delay_range[0], delay_range[1])
                jittered[i] = max(0, jittered[i] - delay_ms * 0.5)

        return jittered

    def add_padding_jitter(self, features: np.ndarray,
                          padding_range=(50, 200)) -> np.ndarray:
        """添加填充抖动（模拟包大小扰动）

        Args:
            features: 原始特征
            padding_range: 填充大小范围

        Returns:
            添加填充抖动后的特征
        """
        jittered = features.copy()
        num_packets = len(jittered)

        for i in range(num_packets):
            if jittered[i] > 0 and random.random() < self.jitter_level:
                padding = random.uniform(padding_range[0], padding_range[1])
                jittered[i] = min(1500, jittered[i] + padding)

        return jittered

    def add_mixed_jitter(self, features: np.ndarray) -> np.ndarray:
        """添加混合抖动

        Args:
            features: 原始特征

        Returns:
            添加混合抖动后的特征
        """
        jittered = features.copy()

        num_packets = len(jittered)
        num_jittered = int(num_packets * self.jitter_level)

        indices = random.sample(range(num_packets), min(num_jittered, num_packets))

        for i in indices:
            if jittered[i] > 0:
                jitter_method = random.choice(['delay', 'padding'])

                if jitter_method == 'delay':
                    delay_ms = random.uniform(10, 100)
                    jittered[i] = max(0, jittered[i] - delay_ms * 0.5)
                else:
                    padding = random.uniform(50, 200)
                    jittered[i] = min(1500, jittered[i] + padding)

        return jittered

    def apply(self, features: np.ndarray) -> np.ndarray:
        """应用抖动

        Args:
            features: 原始特征

        Returns:
            添加抖动后的特征
        """
        if self.jitter_level == 0:
            return features

        if isinstance(features, list):
            features = np.array(features)

        if self.jitter_type == 'delay':
            return self.add_delay_jitter(features)
        elif self.jitter_type == 'padding':
            return self.add_padding_jitter(features)
        elif self.jitter_type == 'mixed':
            return self.add_mixed_jitter(features)
        else:
            return features


class JitterTestRunner:
    """抖动测试运行器"""

    APP_LABELS = {
        0: 'Video',
        1: 'Chat',
        2: 'FileTransfer',
        3: 'Browsing'
    }

    def __init__(self, model_path='saved_models/best_model.pth', device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.model = TrafficCNN(input_length=100, num_classes=4)
        self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

        self.results = {}

    def load_model(self, model_path):
        """加载模型"""
        if not os.path.exists(model_path):
            logger.warning(f"模型文件不存在: {model_path}")
            return

        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"成功加载模型: {model_path}")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")

    def classify(self, features: np.ndarray) -> Tuple[int, float]:
        """分类单个样本

        Args:
            features: 特征向量

        Returns:
            (预测类别ID, 预测概率)
        """
        with torch.no_grad():
            if isinstance(features, list):
                features = torch.FloatTensor(features)
            elif isinstance(features, np.ndarray):
                features = torch.FloatTensor(features)

            features = features.to(self.device)

            if len(features.shape) == 1:
                features = features.unsqueeze(0)

            outputs = self.model(features)
            probs = F.softmax(outputs, dim=1)[0]

            return int(probs.argmax()), float(probs.max())

    def test_jitter_levels(self, X_test: np.ndarray, y_test: np.ndarray,
                          jitter_type='delay',
                          jitter_levels=None,
                          samples_per_level=100) -> Dict:
        """测试不同抖动级别的影响

        Args:
            X_test: 测试特征
            y_test: 测试标签
            jitter_type: 抖动类型
            jitter_levels: 抖动级别列表
            samples_per_level: 每个级别的测试样本数

        Returns:
            测试结果字典
        """
        if jitter_levels is None:
            jitter_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

        logger.info(f"\n{'='*60}")
        logger.info(f"开始 {jitter_type} 抖动测试")
        logger.info(f"{'='*60}")

        results = {
            'jitter_levels': [],
            'accuracies': [],
            'confidences': []
        }

        for level in jitter_levels:
            simulator = JitterSimulator(jitter_type=jitter_type, jitter_level=level)

            correct = 0
            total = 0
            confidences = []

            sample_indices = random.sample(range(len(X_test)),
                                         min(samples_per_level, len(X_test)))

            for idx in sample_indices:
                original_features = X_test[idx]
                true_label = y_test[idx]

                jittered_features = simulator.apply(original_features)

                pred_label, confidence = self.classify(jittered_features)

                confidences.append(confidence)

                if pred_label == true_label:
                    correct += 1
                total += 1

            accuracy = 100 * correct / total if total > 0 else 0
            avg_confidence = np.mean(confidences) * 100

            results['jitter_levels'].append(level)
            results['accuracies'].append(accuracy)
            results['confidences'].append(avg_confidence)

            logger.info(f"Jitter Level: {level:.1f} | "
                       f"Accuracy: {accuracy:.2f}% | "
                       f"Avg Confidence: {avg_confidence:.2f}%")

        self.results[jitter_type] = results
        return results

    def test_all_jitter_types(self, X_test: np.ndarray, y_test: np.ndarray,
                             jitter_levels=None,
                             samples_per_level=100) -> Dict:
        """测试所有抖动类型

        Args:
            X_test: 测试特征
            y_test: 测试标签
            jitter_levels: 抖动级别列表
            samples_per_level: 每个级别的测试样本数

        Returns:
            所有类型的测试结果
        """
        jitter_types = ['delay', 'padding', 'mixed']

        all_results = {}

        for jitter_type in jitter_types:
            results = self.test_jitter_levels(
                X_test, y_test,
                jitter_type=jitter_type,
                jitter_levels=jitter_levels,
                samples_per_level=samples_per_level
            )
            all_results[jitter_type] = results

        return all_results

    def plot_jitter_impact(self, results_dict=None, save_path='saved_models/jitter_test.png'):
        """绘制抖动影响图

        Args:
            results_dict: 测试结果字典
            save_path: 保存路径
        """
        if results_dict is None:
            results_dict = self.results

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        colors = {'delay': '#FF6B6B', 'padding': '#4ECDC4', 'mixed': '#45B7D1'}

        for jitter_type, results in results_dict.items():
            color = colors.get(jitter_type, '#666666')
            label = {
                'delay': 'Network Delay',
                'padding': 'Packet Padding',
                'mixed': 'Mixed Jitter'
            }.get(jitter_type, jitter_type)

            axes[0].plot(results['jitter_levels'], results['accuracies'],
                        marker='o', color=color, label=label, linewidth=2)

            axes[1].plot(results['jitter_levels'], results['confidences'],
                        marker='s', color=color, label=label, linewidth=2)

        axes[0].set_xlabel('Jitter Level', fontsize=12)
        axes[0].set_ylabel('Accuracy (%)', fontsize=12)
        axes[0].set_title('Impact of Network Jitter on Classification Accuracy',
                          fontsize=14)
        axes[0].legend(loc='lower left')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 105])

        axes[1].set_xlabel('Jitter Level', fontsize=12)
        axes[1].set_ylabel('Average Confidence (%)', fontsize=12)
        axes[1].set_title('Impact of Network Jitter on Prediction Confidence',
                          fontsize=14)
        axes[1].legend(loc='lower left')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 105])

        plt.tight_layout()

        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else 'saved_models',
                   exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"抖动测试图已保存: {save_path}")

    def generate_report(self):
        """生成抖动测试报告"""
        print("\n" + "=" * 60)
        print("网络抖动对抗测试报告")
        print("=" * 60)

        for jitter_type, results in self.results.items():
            jitter_name = {
                'delay': '网络延迟抖动',
                'padding': '数据包填充抖动',
                'mixed': '混合抖动'
            }.get(jitter_type, jitter_type)

            print(f"\n{jitter_name}:")
            print("-" * 40)

            for i, level in enumerate(results['jitter_levels']):
                acc = results['accuracies'][i]
                conf = results['confidences'][i]

                bar_len = int(acc / 5)
                bar = '█' * bar_len + '░' * (20 - bar_len)

                print(f"  Level {level:.1f}: {bar} {acc:.1f}% | Conf: {conf:.1f}%")

        baseline_acc = self.results.get('delay', {}).get('accuracies', [100])[0]
        worst_acc = min([min(r['accuracies']) for r in self.results.values()])

        print("\n" + "-" * 60)
        print("总结:")
        print(f"  基准准确率 (无抖动): {baseline_acc:.2f}%")
        print(f"  最低准确率: {worst_acc:.2f}%")
        print(f"  性能下降: {baseline_acc - worst_acc:.2f}%")

        if worst_acc > baseline_acc * 0.7:
            print(f"  评估: 模型具有较好的抗干扰能力")
        else:
            print(f"  评估: 模型抗干扰能力有待提高")

        print("=" * 60)


def run_jitter_test(model_path='saved_models/best_model.pth',
                   num_samples=200,
                   jitter_levels=None):
    """运行抖动测试

    Args:
        model_path: 模型文件路径
        num_samples: 测试样本数量
        jitter_levels: 抖动级别列表
    """
    if jitter_levels is None:
        jitter_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    logger.info("准备测试数据...")
    processor = ISCXDataProcessor(max_packets=100)
    X, y = processor.generate_synthetic_dataset(
        num_samples_per_class=num_samples // 4,
        output_file='data/jitter_test_data.pkl'
    )

    X_test = X
    y_test = y

    logger.info(f"测试数据: {len(X_test)} 样本")

    runner = JitterTestRunner(model_path=model_path)

    results = runner.test_all_jitter_types(
        X_test, y_test,
        jitter_levels=jitter_levels,
        samples_per_level=50
    )

    runner.plot_jitter_impact(results)
    runner.generate_report()

    return results


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='对抗网络抖动测试')
    parser.add_argument('--model', type=str,
                        default='saved_models/best_model.pth',
                        help='模型文件路径')
    parser.add_argument('--samples', type=int, default=200,
                        help='测试样本数量')
    parser.add_argument('--levels', type=str, default='0,0.1,0.2,0.3,0.4,0.5',
                        help='抖动级别列表(逗号分隔)')

    args = parser.parse_args()

    jitter_levels = [float(x) for x in args.levels.split(',')]

    run_jitter_test(
        model_path=args.model,
        num_samples=args.samples,
        jitter_levels=jitter_levels
    )


if __name__ == '__main__':
    main()
