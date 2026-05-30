"""
实验版训练脚本 - 增强版本
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

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out

class SEBlock(nn.Module):
    """Squeeze-and-Excitation注意力块"""
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class ImprovedResNet(nn.Module):
    """改进的残差网络 + SE注意力"""
    def __init__(self, input_length=100, num_classes=4, dropout_rate=0.3):
        super(ImprovedResNet, self).__init__()
        
        # 初始卷积
        self.init_conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # 残差层
        self.layer1 = nn.Sequential(
            ResidualBlock(64, 64),
            SEBlock(64),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate)
        )
        
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            SEBlock(128),
            nn.Dropout(dropout_rate)
        )
        
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            SEBlock(256),
            nn.Dropout(dropout_rate)
        )
        
        self.layer4 = nn.Sequential(
            ResidualBlock(256, 512, stride=2),
            SEBlock(512),
            nn.AdaptiveMaxPool1d(1)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        x = self.init_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

def augment_data(X, y, prob=0.3):
    """数据增强"""
    X_aug = []
    y_aug = []
    
    for i in range(len(X)):
        X_aug.append(X[i])
        y_aug.append(y[i])
        
        if np.random.random() < prob:
            augmented = X[i].copy()
            
            if np.random.random() < 0.3:
                noise = np.random.normal(0, 0.02, len(augmented))
                augmented = augmented + noise
            
            if np.random.random() < 0.2:
                scale = np.random.uniform(0.95, 1.05)
                augmented = augmented * scale
            
            X_aug.append(augmented)
            y_aug.append(y[i])
    
    return np.array(X_aug), np.array(y_aug)

def train_experiment(model_type='resnet_se'):
    """训练实验"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    config = {
        'INPUT_LENGTH': INPUT_LENGTH,
        'NUM_CLASSES': NUM_CLASSES,
        'DROPOUT_RATE': 0.25,
        'LEARNING_RATE': 0.0005,
        'BATCH_SIZE': 64,
        'EPOCHS': 800,
        'WEIGHT_DECAY': 0.005,
        'GRADIENT_CLIP': 5.0,
        'EARLY_STOPPING_PATIENCE': 80,
        'MAX_PACKETS': MAX_PACKETS,
        'MIN_PACKETS': MIN_PACKETS,
        'NORMALIZATION_METHOD': NORMALIZATION_METHOD,
        'CLASS_MAP': CLASS_MAP,
        'REVERSE_CLASS_MAP': REVERSE_CLASS_MAP
    }
    
    logger.info("加载/预处理数据...")
    preprocessor = DataPreprocessor(config)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.load_or_preprocess()
    
    logger.info(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
    
    # 数据增强
    X_train, y_train = augment_data(X_train, y_train, prob=0.3)
    logger.info(f"增强后训练集: {X_train.shape}")
    
    # 计算类别权重
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    
    # 数据转换为Tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
    
    # 创建数据加载器（使用加权采样）
    train_sampler = WeightedRandomSampler(
        weights=torch.tensor(class_weights[y_train], dtype=torch.float32),
        num_samples=len(y_train),
        replacement=True
    )
    
    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=config['BATCH_SIZE'],
        sampler=train_sampler
    )
    
    val_loader = DataLoader(
        TensorDataset(X_val_tensor, y_val_tensor),
        batch_size=config['BATCH_SIZE'],
        shuffle=False
    )
    
    test_loader = DataLoader(
        TensorDataset(X_test_tensor, y_test_tensor),
        batch_size=config['BATCH_SIZE'],
        shuffle=False
    )
    
    # 创建模型
    model = ImprovedResNet(
        input_length=config['INPUT_LENGTH'],
        num_classes=config['NUM_CLASSES'],
        dropout_rate=config['DROPOUT_RATE']
    ).to(device)
    
    logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 损失函数和优化器
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=config['LEARNING_RATE'], weight_decay=config['WEIGHT_DECAY'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    
    # 训练循环
    best_val_acc = 0.0
    early_stopping_count = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    logger.info("开始训练...")
    start_time = time.time()
    
    for epoch in range(config['EPOCHS']):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['GRADIENT_CLIP'])
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        scheduler.step()
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        train_loss_avg = train_loss / train_total
        val_loss_avg = val_loss / val_total
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{config['EPOCHS']} | "
                      f"Train Loss: {train_loss_avg:.4f} | Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss_avg:.4f} | Val Acc: {val_acc:.4f}")
        
        # 早停
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stopping_count = 0
            # 保存到临时文件
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            early_stopping_count += 1
            if early_stopping_count >= config['EARLY_STOPPING_PATIENCE']:
                logger.info(f"早停触发，最佳验证准确率: {best_val_acc:.4f}")
                break
    
    end_time = time.time()
    logger.info(f"训练完成，耗时: {(end_time - start_time)/60:.2f} 分钟")
    
    # 测试
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = test_correct / test_total
    logger.info(f"测试准确率: {test_acc:.4f}")
    
    # 保存报告和模型
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f'saved_models/experiment_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存最佳模型到实验目录
    torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
    logger.info(f"模型已保存到 {os.path.join(save_dir, 'best_model.pth')}")
    
    report = classification_report(all_labels, all_preds, target_names=list(REVERSE_CLASS_MAP.values()))
    cm = confusion_matrix(all_labels, all_preds)
    
    with open(os.path.join(save_dir, 'report.txt'), 'w') as f:
        f.write(f"实验时间: {timestamp}\n")
        f.write(f"模型类型: {model_type}\n")
        f.write(f"测试准确率: {test_acc:.4f}\n")
        f.write(f"\n【分类报告】\n{report}\n")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=list(REVERSE_CLASS_MAP.values()),
                yticklabels=list(REVERSE_CLASS_MAP.values()))
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.legend()
    plt.title('损失曲线')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='训练准确率')
    plt.plot(val_accs, label='验证准确率')
    plt.legend()
    plt.title('准确率曲线')
    
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()
    
    logger.info(f"结果已保存到 {save_dir}")
    
    return test_acc

if __name__ == '__main__':
    train_experiment(model_type='resnet_se')