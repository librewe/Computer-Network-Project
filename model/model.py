"""
Model definitions for encrypted traffic classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block aligned with exp_cnn_se checkpoints."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced_channels),
            nn.ReLU(),
            nn.Linear(reduced_channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        weights = self.se(x).unsqueeze(-1)
        return x * weights


class TrafficCNN(nn.Module):
    """Baseline 1D CNN classifier."""

    def __init__(self, input_length=100, num_classes=4, dropout_rate=0.5):
        super().__init__()

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


class TrafficCNNSE(nn.Module):
    """CNN + SE attention model used by exp_cnn_se checkpoints."""

    def __init__(self, input_length=100, num_classes=4, dropout_rate=0.4):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate),
        )

        self.se = SEBlock(256, reduction=4).se
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        weights = self.se(x).unsqueeze(-1)
        x = x * weights
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


class TrafficLSTM(nn.Module):
    """LSTM classifier used for comparison experiments."""

    def __init__(self, input_length=100, num_classes=4, hidden_size=128,
                 num_layers=2, dropout_rate=0.5):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True,
        )

        self.fc1 = nn.Linear(hidden_size * 2, 256)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(2)

        _, (h_n, _) = self.lstm(x)
        hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        x = F.relu(self.fc1(hidden))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class TrafficMLP(nn.Module):
    """MLP classifier used for comparison experiments."""

    def __init__(self, input_length=100, num_classes=4, hidden_sizes=None,
                 dropout_rate=0.5):
        super().__init__()

        hidden_sizes = hidden_sizes or [256, 128]
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


def infer_model_type_from_state_dict(state_dict):
    """Infer model type from checkpoint parameter names."""
    keys = set(state_dict.keys())

    if any(key.startswith("conv_block1.") for key in keys) and any(key.startswith("se.") for key in keys):
        return "cnn_se"
    if "conv1.weight" in keys and "fc3.weight" in keys:
        return "cnn"
    if "lstm.weight_ih_l0" in keys:
        return "lstm"
    if "feature_layers.0.weight" in keys:
        return "mlp"

    raise ValueError("Unknown checkpoint format: cannot infer model type")


def create_model(model_type="cnn", **kwargs):
    """Model factory."""
    model_type = model_type.lower()

    if model_type == "cnn":
        return TrafficCNN(
            input_length=kwargs.get("input_length", 100),
            num_classes=kwargs.get("num_classes", 4),
            dropout_rate=kwargs.get("dropout_rate", 0.5),
        )
    if model_type == "cnn_se":
        return TrafficCNNSE(
            input_length=kwargs.get("input_length", 100),
            num_classes=kwargs.get("num_classes", 4),
            dropout_rate=kwargs.get("dropout_rate", 0.4),
        )
    if model_type == "lstm":
        return TrafficLSTM(
            input_length=kwargs.get("input_length", 100),
            num_classes=kwargs.get("num_classes", 4),
            hidden_size=kwargs.get("hidden_size", 128),
            num_layers=kwargs.get("num_layers", 2),
            dropout_rate=kwargs.get("dropout_rate", 0.5),
        )
    if model_type == "mlp":
        return TrafficMLP(
            input_length=kwargs.get("input_length", 100),
            num_classes=kwargs.get("num_classes", 4),
            hidden_sizes=kwargs.get("hidden_sizes", [256, 128]),
            dropout_rate=kwargs.get("dropout_rate", 0.5),
        )

    raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_input = torch.randn(8, 100).to(device)

    models = {
        "CNN": create_model("cnn", input_length=100, num_classes=4),
        "CNN_SE": create_model("cnn_se", input_length=100, num_classes=4),
        "LSTM": create_model("lstm", input_length=100, num_classes=4),
        "MLP": create_model("mlp", input_length=100, num_classes=4),
    }

    for name, model in models.items():
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            output = model(test_input)
        print(name, output.shape, sum(p.numel() for p in model.parameters()))
