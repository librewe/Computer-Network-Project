"""
改进版训练脚本 - 提高准确率并优化可视化
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import time
from datetime import datetime

from config import *
from data_preprocessing import DataPreprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdvancedCNN(nn.Module):
    """改进的深度CNN模型"""
    def __init__(self, input_length=100, num_classes=4, dropout_rate=0.3):
        super(AdvancedCNN, self).__init__()
        
        # 多尺度卷积输入层
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate)
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate)
        )
        
        self.conv_block4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate)
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        
        return x


def augment_data(X, y, prob=0.4):
    """增强的数据增强策略"""
    X_aug = []
    y_aug = []
    
    # 计算类别频率（用于少数类增强）
    unique, counts = np.unique(y, return_counts=True)
    class_freq = dict(zip(unique, counts))
    max_freq = max(counts)
    
    for i in range(len(X)):
        X_aug.append(X[i])
        y_aug.append(y[i])
        
        # 少数类增强概率更高
        class_prob = prob * (max_freq / class_freq[y[i]])
        
        if np.random.random() < class_prob:
            augmented = X[i].copy()
            
            # 添加高斯噪声（适度）
            if np.random.random() < 0.3:
                noise = np.random.normal(0, 0.02, len(augmented))
                augmented = augmented + noise
            
            # 随机缩放（较小幅度）
            if np.random.random() < 0.2:
                scale = np.random.uniform(0.95, 1.05)
                augmented = augmented * scale
            
            # 随机反转（低概率）
            if np.random.random() < 0.15:
                augmented = augmented[::-1]
            
            # 时间轴扰动（小幅）
            if np.random.random() < 0.1:
                shift = np.random.randint(-3, 3)
                augmented = np.roll(augmented, shift)
            
            X_aug.append(augmented)
            y_aug.append(y[i])
            
            # 对少数类进行适度二次增强
            if class_freq[y[i]] < max_freq * 0.8 and np.random.random() < 0.15:
                augmented2 = augmented.copy()
                if np.random.random() < 0.5:
                    noise = np.random.normal(0, 0.015, len(augmented2))
                    augmented2 = augmented2 + noise
                else:
                    scale = np.random.uniform(0.98, 1.02)
                    augmented2 = augmented2 * scale
                X_aug.append(augmented2)
                y_aug.append(y[i])
    
    return np.array(X_aug), np.array(y_aug)


def train_improved():
    """改进版训练"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("=" * 70)
    logger.info(f"改进版训练 - {timestamp}")
    logger.info("=" * 70)
    logger.info("改进策略:")
    logger.info("- 4个卷积块的CNN架构")
    logger.info("- CrossEntropyLoss + 类别权重")
    logger.info("- AdamW优化器 + L2正则化")
    logger.info("- CosineAnnealingWarmRestarts学习率调度")
    logger.info("- 数据增强 (噪声、缩放、反转)")
    logger.info("- 梯度裁剪")
    logger.info("- 500轮训练")
    logger.info("=" * 70)

    config = {
        'INPUT_LENGTH': INPUT_LENGTH,
        'NUM_CLASSES': NUM_CLASSES,
        'DROPOUT_RATE': 0.3,
        'LEARNING_RATE': 0.0005,
        'BATCH_SIZE': 64,
        'EPOCHS': 500,
        'WEIGHT_DECAY': 0.005,
        'GRADIENT_CLIP': 5.0,
        'EARLY_STOPPING_PATIENCE': 50,
        'MAX_PACKETS': MAX_PACKETS,
        'MIN_PACKETS': MIN_PACKETS,
        'NORMALIZATION_METHOD': NORMALIZATION_METHOD,
        'CLASS_MAP': CLASS_MAP,
        'REVERSE_CLASS_MAP': REVERSE_CLASS_MAP,
        'CACHE_DIR': CACHE_DIR,
        'PREPROCESSED_DATA_FILE': PREPROCESSED_DATA_FILE
    }

    preprocessor = DataPreprocessor(config)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.load_or_preprocess()

    # 数据增强（温和增强）
    X_train_aug, y_train_aug = augment_data(X_train, y_train, prob=0.25)
    logger.info(f"原始训练集: {len(X_train)} 条")
    logger.info(f"增强后训练集: {len(X_train_aug)} 条")
    logger.info(f"验证集: {len(X_val)} 条")
    logger.info(f"测试集: {len(X_test)} 条")
    logger.info(f"训练集标签分布: {np.bincount(y_train_aug)}")

    train_dataset = TensorDataset(torch.tensor(X_train_aug, dtype=torch.float32),
                                 torch.tensor(y_train_aug, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                               torch.tensor(y_val, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                torch.tensor(y_test, dtype=torch.long))

    # 加权采样处理类别不平衡
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_aug), y=y_train_aug)
    sample_weights = np.array([class_weights[int(y)] for y in y_train_aug])
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    logger.info(f"类别权重: {class_weights}")

    train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'], shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    model = AdvancedCNN(
        input_length=config['INPUT_LENGTH'],
        num_classes=config['NUM_CLASSES'],
        dropout_rate=config['DROPOUT_RATE']
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数量: {param_count:,}")

    # 使用类别权重的交叉熵损失
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    
    # AdamW优化器 + L2正则化
    optimizer = optim.AdamW(model.parameters(), lr=config['LEARNING_RATE'], weight_decay=config['WEIGHT_DECAY'])
    
    # CosineAnnealingWarmRestarts学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)

    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    
    # 记录训练过程
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    learning_rates = []

    for epoch in range(config['EPOCHS']):
        start_time = time.time()

        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['GRADIENT_CLIP'])
            
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels.data).cpu().sum().item()

        train_epoch_loss = train_loss / train_total
        train_epoch_acc = train_correct / train_total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels.data).cpu().sum().item()

        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total

        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]['lr'])

        train_losses.append(train_epoch_loss)
        val_losses.append(val_epoch_loss)
        train_accs.append(train_epoch_acc)
        val_accs.append(val_epoch_acc)

        # 更新最佳模型（基于验证准确率）
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            best_val_loss = val_epoch_loss
            # 保存最佳模型状态
            best_model_state = model.state_dict()
            patience_counter = 0  # 重置早停计数器
            logger.info(f"✓ 更新最佳模型 (验证准确率: {val_epoch_acc:.4f})")
        else:
            patience_counter += 1
        
        # Early Stopping
        if patience_counter >= config['EARLY_STOPPING_PATIENCE']:
            logger.info(f"早停触发! 连续 {patience_counter} 轮验证准确率未提升")
            break

        elapsed = time.time() - start_time
        if (epoch + 1) % 10 == 0 or epoch == config['EPOCHS'] - 1:
            logger.info(f"Epoch [{epoch+1}/{config['EPOCHS']}] | LR: {learning_rates[-1]:.6f} | "
                       f"训练: {train_epoch_loss:.4f}/{train_epoch_acc:.4f} | "
                       f"验证: {val_epoch_loss:.4f}/{val_epoch_acc:.4f} | "
                       f"耗时: {elapsed:.2f}s")

    # 加载最佳模型进行测试
    model.load_state_dict(best_model_state)
    logger.info(f"训练完成! 共训练 {config['EPOCHS']} 轮，使用最佳模型(验证准确率: {best_val_acc:.4f})进行测试")

    # 测试集评估
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    class_names = [config['REVERSE_CLASS_MAP'][i] for i in range(config['NUM_CLASSES'])]
    report = classification_report(all_labels, all_preds, target_names=class_names)
    cm = confusion_matrix(all_labels, all_preds)
    # 直接计算准确率
    correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
    accuracy = correct / len(all_labels)

    # 保存结果
    output_dir = os.path.join(SAVED_MODEL_DIR, f'improved_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    torch.save({'model_state_dict': model.state_dict(), 'config': config},
               os.path.join(output_dir, 'model.pth'))

    # 绘制改进的训练曲线
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # 损失曲线
    axes[0].plot(train_losses, label='训练损失', color='#1f77b4', linewidth=2)
    axes[0].plot(val_losses, label='验证损失', color='#ff7f0e', linewidth=2)
    axes[0].axhline(y=best_val_loss, color='r', linestyle='--', label=f'最佳验证损失: {best_val_loss:.4f}')
    axes[0].set_title('训练与验证损失曲线', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('损失值', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].tick_params(axis='both', labelsize=10)
    
    # 准确率曲线
    axes[1].plot(train_accs, label='训练准确率', color='#1f77b4', linewidth=2)
    axes[1].plot(val_accs, label='验证准确率', color='#ff7f0e', linewidth=2)
    axes[1].axhline(y=best_val_acc, color='r', linestyle='--', label=f'最佳验证准确率: {best_val_acc:.4f}')
    axes[1].set_title('训练与验证准确率曲线', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('准确率', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].tick_params(axis='both', labelsize=10)
    axes[1].set_ylim([0.5, 1.0])
    
    # 学习率曲线
    axes[2].plot(learning_rates, label='学习率', color='#2ca02c', linewidth=2)
    axes[2].set_title('学习率变化曲线', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('学习率', fontsize=12)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, linestyle='--', alpha=0.7)
    axes[2].tick_params(axis='both', labelsize=10)
    axes[2].set_yscale('log')

    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                     xticklabels=class_names, yticklabels=class_names,
                     annot_kws={"size": 14})
    plt.title('混淆矩阵', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 保存报告
    with open(os.path.join(output_dir, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"改进版模型评估报告\n")
        f.write(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("【模型配置】\n")
        f.write(f"- 模型架构: AdvancedCNN (4个卷积块)\n")
        f.write(f"- 参数量: {param_count:,}\n")
        f.write(f"- Dropout: {config['DROPOUT_RATE']}\n")
        f.write(f"- 批大小: {config['BATCH_SIZE']}\n")
        f.write(f"- 学习率: {config['LEARNING_RATE']}\n")
        f.write(f"- 优化器: AdamW + L2正则化({config['WEIGHT_DECAY']})\n")
        f.write(f"- 学习率调度: CosineAnnealingWarmRestarts (T_0=50, T_mult=2)\n")
        f.write(f"- 损失函数: CrossEntropyLoss + 类别权重\n")
        f.write(f"- 早停耐心值: {config['EARLY_STOPPING_PATIENCE']}\n")
        f.write(f"- 梯度裁剪: {config['GRADIENT_CLIP']}\n")
        f.write(f"- 数据增强: 噪声注入、缩放、反转(prob=0.25)\n")
        f.write(f"- 类别平衡: 加权采样 + 类别权重\n\n")
        
        f.write("【数据集统计】\n")
        f.write(f"- 训练集(增强后): {len(X_train_aug)} 条\n")
        f.write(f"- 验证集: {len(X_val)} 条\n")
        f.write(f"- 测试集: {len(X_test)} 条\n")
        f.write(f"- 类别分布: {dict(zip(class_names, np.bincount(y_train_aug)))}\n\n")
        
        f.write("【分类报告】\n")
        f.write(report)
        f.write(f"\n【混淆矩阵】\n")
        f.write(f"{cm}\n")
        f.write(f"\n【最终结果】\n")
        f.write(f"- 最佳验证准确率: {best_val_acc:.4f}\n")
        f.write(f"- 测试集准确率: {accuracy:.4f}\n")
        f.write(f"- 训练轮数: {config['EPOCHS']}\n")
        f.write("\n" + "=" * 70 + "\n")

    logger.info("\n" + "=" * 70)
    logger.info("最终模型评估报告")
    logger.info("=" * 70)
    logger.info(f"\n分类报告:\n{report}")
    logger.info(f"\n混淆矩阵:\n{cm}")
    logger.info(f"\n最终准确率: {accuracy:.4f}")
    logger.info(f"\n模型已保存到: {output_dir}")

    return accuracy, report, cm


if __name__ == '__main__':
    train_improved()
