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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrafficCNN(nn.Module):
    """1D-CNN 流量分类模型"""

    def __init__(self, config):
        super(TrafficCNN, self).__init__()

        self.config = config
        self.input_length = config['INPUT_LENGTH']
        self.num_classes = config['NUM_CLASSES']

        layers = []

        layers.append(nn.Conv1d(1, 64, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool1d(2))

        layers.append(nn.Conv1d(64, 128, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool1d(2))

        layers.append(nn.Conv1d(128, 256, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool1d(2))

        self.conv_layers = nn.Sequential(*layers)

        with torch.no_grad():
            dummy_input = torch.randn(1, 1, self.input_length)
            conv_output = self.conv_layers(dummy_input)
            self.flatten_size = conv_output.numel()

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, 256),
            nn.ReLU(),
            nn.Dropout(config['DROPOUT_RATE']),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(config['DROPOUT_RATE']),
            nn.Linear(128, self.num_classes)
        )

    def forward(self, x):
        """前向传播"""
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        x = self.conv_layers(x)
        x = self.fc_layers(x)

        return x


class ModelTrainer:
    """模型训练器"""

    def __init__(self, config):
        self.config = config

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")

        self.model = TrafficCNN(config).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['LEARNING_RATE'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0.0
        self.early_stopping_counter = 0

        os.makedirs(config['OUTPUT_DIR'], exist_ok=True)

    def prepare_data(self):
        """准备训练数据"""
        preprocessor = DataPreprocessor(self.config)

        processed_path = 'data/processed_dataset.pkl'
        dataset = preprocessor.load_processed_dataset(processed_path)

        if dataset is None:
            dataset = preprocessor.generate_synthetic_dataset()
            preprocessor.save_dataset(dataset, processed_path)

        self.X_train = torch.FloatTensor(dataset['X_train']).to(self.device)
        self.y_train = torch.LongTensor(dataset['y_train']).to(self.device)
        self.X_val = torch.FloatTensor(dataset['X_val']).to(self.device)
        self.y_val = torch.LongTensor(dataset['y_val']).to(self.device)
        self.X_test = torch.FloatTensor(dataset['X_test']).to(self.device)
        self.y_test = torch.LongTensor(dataset['y_test']).to(self.device)

        train_dataset = TensorDataset(self.X_train, self.y_train)
        val_dataset = TensorDataset(self.X_val, self.y_val)
        test_dataset = TensorDataset(self.X_test, self.y_test)

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.config['BATCH_SIZE'], shuffle=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.config['BATCH_SIZE'], shuffle=False
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.config['BATCH_SIZE'], shuffle=False
        )

        logger.info(f"训练集: {len(self.X_train)} 样本")
        logger.info(f"验证集: {len(self.X_val)} 样本")
        logger.info(f"测试集: {len(self.X_test)} 样本")

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_X, batch_y in self.train_loader:
            self.optimizer.zero_grad()

            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total

        return avg_loss, accuracy

    def evaluate(self, data_loader):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        avg_loss = total_loss / len(data_loader)
        accuracy = 100 * correct / total

        return avg_loss, accuracy, all_preds, all_labels

    def train(self):
        """训练模型"""
        logger.info("=" * 60)
        logger.info("开始训练模型")
        logger.info(f"批量大小: {self.config['BATCH_SIZE']}")
        logger.info(f"学习率: {self.config['LEARNING_RATE']}")
        logger.info(f"训练轮数: {self.config['EPOCHS']}")
        logger.info(f"早停耐心值: {self.config['EARLY_STOPPING_PATIENCE']}")
        logger.info("=" * 60)

        start_time = time.time()

        for epoch in range(self.config['EPOCHS']):
            epoch_start = time.time()

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, _, _ = self.evaluate(self.val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            self.scheduler.step(val_loss)

            epoch_time = time.time() - epoch_start

            logger.info(
                f"Epoch {epoch+1}/{self.config['EPOCHS']} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                f"Time: {epoch_time:.1f}s"
            )

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_model('best_model.pth')
                self.early_stopping_counter = 0
                logger.info(f"  -> 保存最佳模型 (Val Acc: {val_acc:.2f}%)")
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= self.config['EARLY_STOPPING_PATIENCE']:
                    logger.info(f"  -> 早停触发，停止训练")
                    break

        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"训练完成! 总时间: {total_time:.1f}秒")
        logger.info(f"最佳验证准确率: {self.best_val_acc:.2f}%")
        logger.info("=" * 60)

        self.save_model('final_model.pth')
        self.evaluate_and_report()

    def evaluate_and_report(self):
        """评估并生成报告"""
        logger.info("\n测试集评估:")
        test_loss, test_acc, predictions, labels = self.evaluate(self.test_loader)
        logger.info(f"测试准确率: {test_acc:.2f}%")

        class_names = list(self.config['CLASS_MAP'].keys())

        print("\n" + "=" * 60)
        print("分类报告:")
        print("=" * 60)
        report = classification_report(
            labels, predictions,
            target_names=class_names,
            output_dict=True
        )
        print(classification_report(
            labels, predictions,
            target_names=class_names
        ))

        cm = confusion_matrix(labels, predictions)
        self.plot_confusion_matrix(cm, class_names)

        report_path = os.path.join(self.config['OUTPUT_DIR'], self.config['REPORT_FILENAME'])
        self.save_report(report, cm, class_names, report_path)

        return report

    def plot_confusion_matrix(self, cm, class_names):
        """绘制混淆矩阵"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        plt.savefig(os.path.join(self.config['OUTPUT_DIR'], 'confusion_matrix.png'), dpi=150)
        plt.close()
        logger.info(f"混淆矩阵已保存")

    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        epochs_range = range(1, len(self.train_losses) + 1)

        ax1.plot(epochs_range, self.train_losses, 'b-', label='训练损失')
        ax1.plot(epochs_range, self.val_losses, 'r-', label='验证损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Curve')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(epochs_range, self.train_accs, 'b-', label='训练准确率')
        ax2.plot(epochs_range, self.val_accs, 'r-', label='验证准确率')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Accuracy Curve')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.config['OUTPUT_DIR'], 'training_curves.png'), dpi=150)
        plt.close()
        logger.info(f"训练曲线已保存")

    def save_model(self, filename):
        """保存模型"""
        filepath = os.path.join(self.config['OUTPUT_DIR'], filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
        }, filepath)
        logger.info(f"模型已保存到: {filepath}")

    def save_report(self, report, cm, class_names, filepath):
        """保存评估报告"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("模型评估报告\n")
            f.write("=" * 60 + "\n\n")

            f.write("1. 配置参数\n")
            f.write("-" * 40 + "\n")
            f.write(f"模型类型: {self.config['MODEL_TYPE']}\n")
            f.write(f"输入长度: {self.config['INPUT_LENGTH']}\n")
            f.write(f"类别数量: {self.config['NUM_CLASSES']}\n")
            f.write(f"批量大小: {self.config['BATCH_SIZE']}\n")
            f.write(f"学习率: {self.config['LEARNING_RATE']}\n")
            f.write(f"训练轮数: {len(self.train_losses)}\n")
            f.write(f"最佳验证准确率: {self.best_val_acc:.2f}%\n")
            f.write("\n")

            f.write("2. 分类指标\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'类别':<15} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1':<10}\n")
            f.write("-" * 60 + "\n")

            for class_name in class_names:
                metrics = report[class_name]
                f.write(f"{class_name:<15} {metrics['recall']*100:<10.2f} "
                        f"{metrics['precision']*100:<10.2f} "
                        f"{metrics['recall']*100:<10.2f} "
                        f"{metrics['f1-score']*100:<10.2f}\n")

            f.write("-" * 60 + "\n")
            f.write(f"{'macro avg':<15} {report['macro avg']['recall']*100:<10.2f} "
                    f"{report['macro avg']['precision']*100:<10.2f} "
                    f"{report['macro avg']['recall']*100:<10.2f} "
                    f"{report['macro avg']['f1-score']*100:<10.2f}\n")
            f.write(f"{'weighted avg':<15} {report['weighted avg']['recall']*100:<10.2f} "
                    f"{report['weighted avg']['precision']*100:<10.2f} "
                    f"{report['weighted avg']['recall']*100:<10.2f} "
                    f"{report['weighted avg']['f1-score']*100:<10.2f}\n")
            f.write("\n")

            f.write("3. 混淆矩阵\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'':<12}" + "".join([f"{c:<12}" for c in class_names]) + "\n")
            for i, row in enumerate(cm):
                f.write(f"{class_names[i]:<12}" + "".join([f"{v:<12}" for v in row]) + "\n")

        logger.info(f"评估报告已保存到: {filepath}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='训练流量分类模型')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='批量大小')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='学习率')
    parser.add_argument('--max-packets', type=int, default=MAX_PACKETS, help='最大数据包数')

    args = parser.parse_args()

    config = {
        'MAX_PACKETS': args.max_packets,
        'MIN_PACKETS': MIN_PACKETS,
        'NORMALIZATION_METHOD': NORMALIZATION_METHOD,
        'TRAIN_RATIO': TRAIN_RATIO,
        'VAL_RATIO': VAL_RATIO,
        'TEST_RATIO': TEST_RATIO,
        'MODEL_TYPE': MODEL_TYPE,
        'INPUT_LENGTH': INPUT_LENGTH,
        'NUM_CLASSES': NUM_CLASSES,
        'DROPOUT_RATE': DROPOUT_RATE,
        'BATCH_SIZE': args.batch_size,
        'LEARNING_RATE': args.lr,
        'EPOCHS': args.epochs,
        'EARLY_STOPPING_PATIENCE': EARLY_STOPPING_PATIENCE,
        'OUTPUT_DIR': OUTPUT_DIR,
        'MODEL_FILENAME': MODEL_FILENAME,
        'REPORT_FILENAME': REPORT_FILENAME,
        'CLASS_MAP': CLASS_MAP,
        'REVERSE_CLASS_MAP': REVERSE_CLASS_MAP
    }

    trainer = ModelTrainer(config)
    trainer.prepare_data()
    trainer.train()
    trainer.plot_training_curves()

    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"模型保存在: {config['OUTPUT_DIR']}")
    print(f"评估报告: {config['OUTPUT_DIR']}{config['REPORT_FILENAME']}")
    print("=" * 60)


if __name__ == '__main__':
    main()
