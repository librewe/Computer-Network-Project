"""
模型训练模块 - 使用PyTorch构建1D-CNN分类模型
实现完整的训练循环和评估流程
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import time

from config import *
from data_preprocessing import DataPreprocessor
from model import TrafficCNN, create_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """模型训练器"""

    def __init__(self, config):
        self.config = config

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")

        self.model = TrafficCNN(
            input_length=config['INPUT_LENGTH'],
            num_classes=config['NUM_CLASSES'],
            dropout_rate=config['DROPOUT_RATE']
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['LEARNING_RATE'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum().item()

        epoch_loss = total_loss / total
        epoch_acc = correct / total

        return epoch_loss, epoch_acc

    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += predicted.eq(labels.data).cpu().sum().item()

        epoch_loss = total_loss / total
        epoch_acc = correct / total

        return epoch_loss, epoch_acc

    def test(self, test_loader):
        """测试模型"""
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return all_labels, all_preds

    def train(self, train_loader, val_loader, epochs=50, patience=10):
        """训练主循环"""
        logger.info("开始训练...")
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0

        for epoch in range(epochs):
            start_time = time.time()

            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            self.scheduler.step(val_loss)

            elapsed = time.time() - start_time

            logger.info(
                f"Epoch [{epoch + 1}/{epochs}] | "
                f"训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.4f} | "
                f"验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.4f} | "
                f"耗时: {elapsed:.2f}s"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict()
                patience_counter = 0
                logger.info(f"发现更好的模型，保存中...")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"验证损失连续{patience}轮没有改善，提前停止训练")
                    break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        logger.info("训练完成!")

    def save_model(self, path):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'best_val_acc': max(self.val_accs) if self.val_accs else 'N/A'
        }
        torch.save(checkpoint, path)
        logger.info(f"模型已保存到: {path}")

    def plot_training_curves(self, save_path=None):
        """绘制训练曲线"""
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='训练损失')
        plt.plot(self.val_losses, label='验证损失')
        plt.title('训练和验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label='训练准确率')
        plt.plot(self.val_accs, label='验证准确率')
        plt.title('训练和验证准确率')
        plt.xlabel('Epoch')
        plt.ylabel('准确率')
        plt.legend()

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"训练曲线已保存到: {save_path}")

        plt.close()

    def generate_report(self, test_loader, class_names, save_path=None):
        """生成评估报告"""
        y_true, y_pred = self.test(test_loader)

        report = classification_report(y_true, y_pred, target_names=class_names)
        cm = confusion_matrix(y_true, y_pred)

        logger.info("\n" + "=" * 60)
        logger.info("测试集评估报告")
        logger.info("=" * 60)
        logger.info("\n分类报告:\n")
        logger.info(report)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title('混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')

        if save_path:
            report_dir = os.path.dirname(save_path)
            os.makedirs(report_dir, exist_ok=True)

            plt.savefig(save_path.replace('.txt', '_cm.png'), dpi=150, bbox_inches='tight')
            logger.info(f"混淆矩阵已保存到: {save_path.replace('.txt', '_cm.png')}")

            with open(save_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("加密流量分类模型评估报告\n")
                f.write("=" * 60 + "\n\n")
                f.write("分类报告:\n")
                f.write(report)
                f.write("\n混淆矩阵:\n")
                f.write(str(cm))
                f.write("\n\n")
                f.write("=" * 60 + "\n")
                f.write("评估完成\n")
                f.write("=" * 60 + "\n")

            logger.info(f"评估报告已保存到: {save_path}")

        plt.close()

        return report, cm


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("模型训练模块")
    logger.info("=" * 60)

    config = {
        'INPUT_LENGTH': INPUT_LENGTH,
        'NUM_CLASSES': NUM_CLASSES,
        'DROPOUT_RATE': DROPOUT_RATE,
        'LEARNING_RATE': LEARNING_RATE,
        'BATCH_SIZE': BATCH_SIZE,
        'EPOCHS': EPOCHS,
        'EARLY_STOPPING_PATIENCE': EARLY_STOPPING_PATIENCE,
        'MAX_PACKETS': MAX_PACKETS,
        'MIN_PACKETS': MIN_PACKETS,
        'NORMALIZATION_METHOD': NORMALIZATION_METHOD,
        'CLASS_MAP': CLASS_MAP,
        'REVERSE_CLASS_MAP': REVERSE_CLASS_MAP,
        'CACHE_DIR': CACHE_DIR,
        'PREPROCESSED_DATA_FILE': PREPROCESSED_DATA_FILE
    }

    logger.info("加载数据集...")
    preprocessor = DataPreprocessor(config)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.load_or_preprocess()

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                 torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                               torch.tensor(y_val, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'], shuffle=False)

    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"验证集大小: {len(val_dataset)}")
    logger.info(f"测试集大小: {len(test_dataset)}")

    trainer = ModelTrainer(config)

    logger.info(f"\n模型结构:\n{trainer.model}")
    param_count = sum(p.numel() for p in trainer.model.parameters())
    logger.info(f"模型参数量: {param_count:,}")

    trainer.train(train_loader, val_loader,
                  epochs=config['EPOCHS'],
                  patience=config['EARLY_STOPPING_PATIENCE'])

    trainer.save_model(os.path.join(SAVED_MODEL_DIR, 'best_model.pth'))
    torch.save(trainer.model.state_dict(), os.path.join(SAVED_MODEL_DIR, 'final_model.pth'))

    trainer.plot_training_curves(os.path.join(SAVED_MODEL_DIR, 'training_curves.png'))

    class_names = [config['REVERSE_CLASS_MAP'][i] for i in range(config['NUM_CLASSES'])]
    trainer.generate_report(test_loader, class_names,
                            os.path.join(SAVED_MODEL_DIR, 'evaluation_report.txt'))

    logger.info("\n" + "=" * 60)
    logger.info("训练流程完成!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
