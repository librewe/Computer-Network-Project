"""
模型可视化与对比实验
绘制训练过程的Loss/Accuracy曲线图及混淆矩阵热力图
引入MLP和LSTM模型进行对比
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import time
import logging

from model import TrafficCNN, TrafficLSTM, TrafficMLP, create_model
from data_preprocessing import ISCXDataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelVisualizer:
    """模型可视化工具"""

    APP_LABELS = {
        0: 'Video',
        1: 'Chat',
        2: 'FileTransfer',
        3: 'Browsing'
    }

    def __init__(self, save_dir='saved_models'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_training_curves(self, history_dict, save_path=None):
        """绘制多个模型的训练曲线对比

        Args:
            history_dict: 字典，key为模型名称，value为包含train_losses, val_losses等的历史记录
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        colors = {'CNN': '#FF6B6B', 'LSTM': '#4ECDC4', 'MLP': '#45B7D1'}

        for model_name, history in history_dict.items():
            color = colors.get(model_name, '#666666')

            if 'train_losses' in history:
                axes[0].plot(history['train_losses'],
                            label=f'{model_name} Train',
                            color=color, linestyle='-', alpha=0.8)
            if 'val_losses' in history:
                axes[0].plot(history['val_losses'],
                            label=f'{model_name} Val',
                            color=color, linestyle='--', alpha=0.8)

            if 'train_accs' in history:
                axes[1].plot(history['train_accs'],
                            label=f'{model_name} Train',
                            color=color, linestyle='-', alpha=0.8)
            if 'val_accs' in history:
                axes[1].plot(history['val_accs'],
                            label=f'{model_name} Val',
                            color=color, linestyle='--', alpha=0.8)

        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14)
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].set_title('Training and Validation Accuracy', fontsize=14)
        axes[1].legend(loc='lower right')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.save_dir, 'training_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"训练曲线对比图已保存: {save_path}")

    def plot_confusion_matrix_heatmap(self, cm, title='Confusion Matrix',
                                       save_path=None):
        """绘制混淆矩阵热力图

        Args:
            cm: 混淆矩阵
            title: 标题
            save_path: 保存路径
        """
        plt.figure(figsize=(10, 8))

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=[self.APP_LABELS[i] for i in range(cm.shape[0])],
            yticklabels=[self.APP_LABELS[i] for i in range(cm.shape[0])],
            cbar_kws={'label': 'Count'}
        )

        plt.title(title, fontsize=14)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.save_dir, 'confusion_matrix.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"混淆矩阵已保存: {save_path}")

    def plot_confusion_matrices_comparison(self, cm_dict, save_path=None):
        """绘制多个模型的混淆矩阵对比

        Args:
            cm_dict: 字典，key为模型名称，value为混淆矩阵
        """
        n_models = len(cm_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))

        if n_models == 1:
            axes = [axes]

        for idx, (model_name, cm) in enumerate(cm_dict.items()):
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=[self.APP_LABELS[i] for i in range(cm.shape[0])],
                yticklabels=[self.APP_LABELS[i] for i in range(cm.shape[0])],
                ax=axes[idx],
                cbar_kws={'label': 'Count'}
            )
            axes[idx].set_title(f'{model_name} Confusion Matrix', fontsize=12)
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.save_dir, 'confusion_matrices_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"混淆矩阵对比图已保存: {save_path}")

    def plot_model_comparison_bar(self, results_dict, save_path=None):
        """绘制模型性能对比柱状图

        Args:
            results_dict: 字典，key为模型名称，value为包含accuracy, precision等指标的字典
        """
        models = list(results_dict.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(models))
        width = 0.2

        for i, metric in enumerate(metrics):
            values = [results_dict[m].get(metric, 0) for m in models]
            bars = ax.bar(x + i * width, values, width, label=metric.capitalize())

            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{val:.1f}', ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Score (%)', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models)
        ax.legend(loc='lower right')
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.save_dir, 'model_comparison_bar.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"模型对比柱状图已保存: {save_path}")

    def plot_inference_latency(self, latency_dict, save_path=None):
        """绘制推理延迟对比图

        Args:
            latency_dict: 字典，key为模型名称，value为延迟列表(毫秒)
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        models = list(latency_dict.keys())
        colors = {'CNN': '#FF6B6B', 'LSTM': '#4ECDC4', 'MLP': '#45B7D1'}

        latencies = [np.mean(latency_dict[m]) for m in models]
        stds = [np.std(latency_dict[m]) for m in models]

        bars = axes[0].bar(models, latencies, yerr=stds,
                           color=[colors.get(m, '#666666') for m in models],
                           capsize=5, alpha=0.8)
        axes[0].set_ylabel('Latency (ms)', fontsize=12)
        axes[0].set_title('Average Inference Latency', fontsize=14)
        axes[0].grid(True, alpha=0.3, axis='y')

        for bar, lat in zip(bars, latencies):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stds) * 0.1,
                        f'{lat:.2f}ms', ha='center', va='bottom', fontsize=10)

        bp = axes[1].boxplot([latency_dict[m] for m in models],
                             labels=models,
                             patch_artist=True)
        for patch, model in zip(bp['boxes'], models):
            patch.set_facecolor(colors.get(model, '#666666'))
            patch.set_alpha(0.7)

        axes[1].set_ylabel('Latency (ms)', fontsize=12)
        axes[1].set_title('Inference Latency Distribution', fontsize=14)
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.save_dir, 'inference_latency.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"推理延迟对比图已保存: {save_path}")

    def plot_learning_rate_schedule(self, lrs_list, save_path=None):
        """绘制学习率调度曲线

        Args:
            lrs_list: 学习率历史列表
        """
        plt.figure(figsize=(10, 5))
        plt.plot(lrs_list, linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.title('Learning Rate Schedule', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.save_dir, 'learning_rate_schedule.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"学习率曲线已保存: {save_path}")


class ModelComparator:
    """模型对比器"""

    APP_LABELS = {
        0: 'Video',
        1: 'Chat',
        2: 'FileTransfer',
        3: 'Browsing'
    }

    def __init__(self, input_length=100, num_classes=4, device=None):
        self.input_length = input_length
        self.num_classes = num_classes

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.models = {}
        self.results = {}

    def prepare_data(self, num_samples_per_class=200):
        """准备测试数据"""
        processor = ISCXDataProcessor(max_packets=self.input_length)
        X, y = processor.generate_synthetic_dataset(
            num_samples_per_class=num_samples_per_class,
            output_file='data/test_data.pkl'
        )

        self.X_test = torch.FloatTensor(X).to(self.device)
        self.y_test = torch.LongTensor(y).to(self.device)

        logger.info(f"测试数据: {len(self.X_test)} 样本")

    def train_and_evaluate(self, model_type, epochs=20, batch_size=64,
                          learning_rate=0.001):
        """训练并评估单个模型"""
        logger.info(f"\n{'='*50}")
        logger.info(f"训练 {model_type.upper()} 模型")
        logger.info(f"{'='*50}")

        model = create_model(
            model_type=model_type,
            input_length=self.input_length,
            num_classes=self.num_classes
        ).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        train_dataset = TensorDataset(self.X_test, self.y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        history = {
            'train_losses': [],
            'train_accs': []
        }

        start_time = time.time()

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            correct = 0
            total = 0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

            avg_loss = total_loss / len(train_loader)
            accuracy = 100 * correct / total

            history['train_losses'].append(avg_loss)
            history['train_accs'].append(accuracy)

            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {accuracy:.2f}%")

        train_time = time.time() - start_time

        model.eval()
        with torch.no_grad():
            outputs = model(self.X_test)
            _, predicted = torch.max(outputs, 1)

            y_pred = predicted.cpu().numpy()
            y_true = self.y_test.cpu().numpy()

            accuracy = 100 * (predicted == self.y_test).sum().item() / len(self.y_test)

            from sklearn.metrics import precision_recall_fscore_support
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='weighted'
            )

            cm = confusion_matrix(y_true, y_pred)

        latency = self.measure_latency(model, num_iterations=100)

        self.models[model_type] = model
        self.results[model_type] = {
            'accuracy': accuracy,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100,
            'train_time': train_time,
            'latency': latency,
            'history': history,
            'confusion_matrix': cm
        }

        logger.info(f"\n{model_type.upper()} 结果:")
        logger.info(f"  准确率: {accuracy:.2f}%")
        logger.info(f"  精确率: {precision*100:.2f}%")
        logger.info(f"  召回率: {recall*100:.2f}%")
        logger.info(f"  F1分数: {f1*100:.2f}%")
        logger.info(f"  训练时间: {train_time:.2f}秒")
        logger.info(f"  推理延迟: {latency:.3f}ms")

        return model, history

    def measure_latency(self, model, num_iterations=100):
        """测量推理延迟"""
        model.eval()

        dummy_input = torch.randn(1, self.input_length).to(self.device)

        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)

        latencies = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.time()
                _ = model(dummy_input)
                latencies.append((time.time() - start) * 1000)

        return np.mean(latencies)

    def compare_models(self, model_types=['cnn', 'lstm', 'mlp'],
                      epochs=20, batch_size=64):
        """对比多个模型"""
        history_dict = {}

        for model_type in model_types:
            model, history = self.train_and_evaluate(
                model_type, epochs=epochs, batch_size=batch_size
            )
            history_dict[model_type.upper()] = history

        return history_dict

    def generate_report(self, save_dir='saved_models'):
        """生成对比报告"""
        visualizer = ModelVisualizer(save_dir=save_dir)

        history_dict = {name: self.results[name]['history']
                       for name in self.results}
        visualizer.plot_training_curves(history_dict)

        cm_dict = {name: self.results[name]['confusion_matrix']
                  for name in self.results}
        visualizer.plot_confusion_matrices_comparison(cm_dict)

        visualizer.plot_model_comparison_bar(self.results)

        latency_dict = {name: [self.results[name]['latency']] * 50
                       for name in self.results}
        visualizer.plot_inference_latency(latency_dict)

        print("\n" + "=" * 60)
        print("模型对比报告")
        print("=" * 60)
        print(f"\n{'模型':<10} {'准确率':<12} {'精确率':<12} {'召回率':<12} {'F1分数':<12} {'延迟':<10}")
        print("-" * 60)
        for name, result in self.results.items():
            print(f"{name.upper():<10} {result['accuracy']:.2f}%      "
                  f"{result['precision']:.2f}%      {result['recall']:.2f}%      "
                  f"{result['f1']:.2f}%      {result['latency']:.3f}ms")
        print("-" * 60)

        best_model = max(self.results.items(),
                        key=lambda x: x[1]['accuracy'])[0]
        print(f"\n最佳模型: {best_model.upper()} "
              f"(准确率: {self.results[best_model]['accuracy']:.2f}%)")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='模型可视化与对比')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=64, help='批量大小')
    parser.add_argument('--compare', action='store_true',
                        help='对比所有模型')

    args = parser.parse_args()

    if args.compare:
        comparator = ModelComparator(input_length=100, num_classes=4)
        comparator.prepare_data(num_samples_per_class=200)
        comparator.compare_models(
            model_types=['cnn', 'lstm', 'mlp'],
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        comparator.generate_report()
    else:
        print("使用 --compare 参数运行以对比所有模型")


if __name__ == '__main__':
    main()
