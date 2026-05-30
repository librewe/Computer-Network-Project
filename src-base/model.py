"""
1D-CNN 模型定义
用于加密流量分类的卷积神经网络模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrafficCNN(nn.Module):
    """1D-CNN 流量分类模型"""

    def __init__(self, input_length=100, num_classes=4, dropout_rate=0.5):
        super(TrafficCNN, self).__init__()

        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)

        self.conv4 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(128)

        self.adaptive_pool = nn.AdaptiveAvgPool1d(4)

        self.fc1 = nn.Linear(128 * 4, 256)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))

        x = self.adaptive_pool(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x


class TrafficLSTM(nn.Module):
    """LSTM 流量分类模型（用于对比）"""

    def __init__(self, input_length=100, num_classes=4, hidden_size=128,
                 num_layers=2, dropout_rate=0.5):
        super(TrafficLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True
        )

        self.fc1 = nn.Linear(hidden_size * 2, 256)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(2)

        lstm_out, (h_n, c_n) = self.lstm(x)

        h_n_forward = h_n[-2, :, :]
        h_n_backward = h_n[-1, :, :]
        hidden = torch.cat((h_n_forward, h_n_backward), dim=1)

        x = F.relu(self.fc1(hidden))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x


class TrafficMLP(nn.Module):
    """MLP 流量分类模型（用于对比）"""

    def __init__(self, input_length=100, num_classes=4, hidden_sizes=[256, 128],
                 dropout_rate=0.5):
        super(TrafficMLP, self).__init__()

        layers = []
        input_size = input_length

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_size

        self.feature_layers = nn.Sequential(*layers)
        self.classifier = nn.Linear(input_size, num_classes)

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        x = self.feature_layers(x)
        x = self.classifier(x)

        return x


def create_model(model_type='cnn', **kwargs):
    """模型工厂函数

    Args:
        model_type: 模型类型 ('cnn', 'lstm', 'mlp')
        **kwargs: 模型参数

    Returns:
        nn.Module: 创建的模型
    """
    model_type = model_type.lower()

    if model_type == 'cnn':
        return TrafficCNN(
            input_length=kwargs.get('input_length', 100),
            num_classes=kwargs.get('num_classes', 4),
            dropout_rate=kwargs.get('dropout_rate', 0.5)
        )
    elif model_type == 'lstm':
        return TrafficLSTM(
            input_length=kwargs.get('input_length', 100),
            num_classes=kwargs.get('num_classes', 4),
            hidden_size=kwargs.get('hidden_size', 128),
            num_layers=kwargs.get('num_layers', 2),
            dropout_rate=kwargs.get('dropout_rate', 0.5)
        )
    elif model_type == 'mlp':
        return TrafficMLP(
            input_length=kwargs.get('input_length', 100),
            num_classes=kwargs.get('num_classes', 4),
            hidden_sizes=kwargs.get('hidden_sizes', [256, 128]),
            dropout_rate=kwargs.get('dropout_rate', 0.5)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == '__main__':
    print("=" * 60)
    print("模型测试")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    batch_size = 8
    input_length = 100
    num_classes = 4

    test_input = torch.randn(batch_size, input_length).to(device)

    print(f"\n输入形状: {test_input.shape}")

    models = {
        'CNN': create_model('cnn', input_length=input_length, num_classes=num_classes),
        'LSTM': create_model('lstm', input_length=input_length, num_classes=num_classes),
        'MLP': create_model('mlp', input_length=input_length, num_classes=num_classes)
    }

    for name, model in models.items():
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            output = model(test_input)

        param_count = sum(p.numel() for p in model.parameters())

        print(f"\n{name} 模型:")
        print(f"  输出形状: {output.shape}")
        print(f"  参数量: {param_count:,}")

    print("\n" + "=" * 60)
    print("模型测试完成!")
    print("=" * 60)
