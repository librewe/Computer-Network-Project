# 配置文件
# 数据集路径配置
DATASET_PATH = 'data/'
# 按应用类型组织的数据文件（非VPN流量）
PCAP_FILES = [
    # Video类
    'data/netflix1.pcap', 'data/netflix2.pcap', 'data/netflix3.pcap', 'data/netflix4.pcap',
    'data/youtube1.pcap', 'data/youtube2.pcap', 'data/youtube3.pcap', 'data/youtube4.pcap',
    'data/youtube5.pcap', 'data/youtube6.pcap', 'data/youtubeHTML5_1.pcap',
    'data/vimeo1.pcap', 'data/vimeo2.pcap', 'data/vimeo3.pcap', 'data/vimeo4.pcap',
    'data/spotify1.pcap', 'data/spotify2.pcap', 'data/spotify3.pcap', 'data/spotify4.pcap',
    # Chat类
    'data/aim_chat_3a.pcap', 'data/aim_chat_3b.pcap',
    'data/facebook_chat_4a.pcap', 'data/facebook_chat_4b.pcap',
    'data/hangouts_chat_4a.pcap', 'data/hangout_chat_4b.pcap',
    'data/icq_chat_3a.pcap', 'data/icq_chat_3b.pcap',
    'data/skype_chat1a.pcap', 'data/skype_chat1b.pcap',
    # FileTransfer类
    'data/ftps_down_1a.pcap', 'data/ftps_down_1b.pcap',
    'data/ftps_up_2a.pcap', 'data/ftps_up_2b.pcap',
    'data/sftpDown1.pcap', 'data/sftpDown2.pcap',
    'data/sftpUp1.pcap', 'data/sftp_up_2a.pcap', 'data/sftp_up_2b.pcap',
    'data/scpDown1.pcap', 'data/scpDown2.pcap', 'data/scpDown3.pcap',
    'data/scpDown4.pcap', 'data/scpDown5.pcap', 'data/scpDown6.pcap',
    'data/scpUp1.pcap', 'data/scpUp2.pcap', 'data/scpUp3.pcap',
    'data/scpUp5.pcap', 'data/scpUp6.pcap',
    'data/skype_file1.pcap', 'data/skype_file2.pcap', 'data/skype_file3.pcap',
    # Web类 - 添加更多文件
    'data/email1a.pcap', 'data/email1b.pcap', 'data/email2a.pcap', 'data/email2b.pcap',
    'data/facebook_audio1a.pcap', 'data/facebook_audio1b.pcapng',
    'data/facebook_audio2a.pcap', 'data/facebook_audio2b.pcapng',
    'data/hangouts_audio1a.pcap', 'data/hangouts_audio1b.pcapng',
    'data/hangouts_audio2a.pcap', 'data/hangouts_audio2b.pcapng',
    'data/skype_audio1a.pcap', 'data/skype_audio1b.pcapng',
    'data/skype_audio2a.pcap', 'data/skype_audio2b.pcapng',
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
SAVED_MODEL_DIR = 'saved_models/'
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

# 数据预处理缓存配置
CACHE_DIR = 'data/cache/'
PREPROCESSED_DATA_FILE = 'preprocessed_data.npz'
