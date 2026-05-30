# 配置文件
# 数据集路径配置
DATASET_PATH = 'data/ISCX-VPN/'
PCAP_FILES = [
    'data/ISCX-VPN/VPN_VoIP.pcap',
    'data/ISCX-VPN/VPN_Video.pcap',
    'data/ISCX-VPN/nonVPN_Chat.pcap',
    'data/ISCX-VPN/nonVPN_FileTransfer.pcap',
    'data/ISCX-VPN/nonVPN_Browsing.pcap',
    'data/ISCX-VPN/nonVPN_Video.pcap'
]

# 特征提取配置
MAX_PACKETS = 100  # 每个流提取的最大数据包数
MIN_PACKETS = 10   # 有效流的最小数据包数

# 数据预处理配置
NORMALIZATION_METHOD = 'minmax'  # 'minmax' 或 'zscore'
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 模型配置
MODEL_TYPE = 'cnn'
INPUT_LENGTH = 100
NUM_CLASSES = 4

# CNN模型参数
CONV_LAYERS = [
    {'out_channels': 64, 'kernel_size': 3, 'padding': 1},
    {'out_channels': 128, 'kernel_size': 3, 'padding': 1},
    {'out_channels': 256, 'kernel_size': 3, 'padding': 1}
]
POOL_SIZE = 2
FC_LAYERS = [256, 128]
DROPOUT_RATE = 0.5

# 训练配置
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

# 输出配置
OUTPUT_DIR = 'saved_models/'
MODEL_FILENAME = 'traffic_cnn_model.pth'
REPORT_FILENAME = 'evaluation_report.txt'

# 类别映射
CLASS_MAP = {
    'Video': 0,
    'Chat': 1,
    'FileTransfer': 2,
    'Web': 3
}

REVERSE_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}
