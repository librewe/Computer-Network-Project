"""
流量特征提取与预处理模块
从PCAP文件中提取流特征，并进行归一化处理
"""

import os
import struct
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlowExtractor:
    """从PCAP文件中提取流特征"""

    def __init__(self, max_packets=100):
        self.max_packets = max_packets
        self.flows = defaultdict(lambda: {
            'packets': [],
            'timestamps': [],
            'lengths_up': [],
            'lengths_down': []
        })

    def parse_pcap_header(self, f):
        """解析PCAP文件头"""
        header = f.read(24)
        if len(header) < 24:
            return None
        magic = struct.unpack('<I', header[0:4])[0]
        if magic != 0xa1b2c3d4 and magic != 0xd4c3b2a1:
            return None
        return True

    def parse_packet(self, f):
        """解析单个数据包"""
        header = f.read(16)
        if len(header) < 16:
            return None, None, None

        ts_sec, ts_usec, incl_len = struct.unpack('<III', header[0:12])

        packet_data = f.read(incl_len)
        if len(packet_data) < incl_len:
            return None, None, None

        timestamp = ts_sec + ts_usec / 1000000.0

        if len(packet_data) < 34:
            return None, None, None

        ip_header = packet_data[14:34]
        if len(ip_header) < 20:
            return None, None, None

        version = (ip_header[0] >> 4) & 0xF
        if version != 4:
            return None, None, None

        ihl = (ip_header[0] & 0xF) * 4
        protocol = ip_header[9]

        src_ip = f"{ip_header[12]}.{ip_header[13]}.{ip_header[14]}.{ip_header[15]}"
        dst_ip = f"{ip_header[16]}.{ip_header[17]}.{ip_header[18]}.{ip_header[19]}"

        if len(packet_data) < 14 + ihl + 4:
            return None, None, None

        src_port = struct.unpack('!H', packet_data[14 + ihl:14 + ihl + 2])[0]
        dst_port = struct.unpack('!H', packet_data[14 + ihl + 2:14 + ihl + 4])[0]

        return {
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'src_port': src_port,
            'dst_port': dst_port,
            'protocol': protocol,
            'length': incl_len,
            'timestamp': timestamp
        }

    def extract_flow_key(self, packet_info):
        """提取流标识符（五元组）"""
        src = f"{packet_info['src_ip']}:{packet_info['src_port']}"
        dst = f"{packet_info['dst_ip']}:{packet_info['dst_port']}"
        proto = packet_info['protocol']

        return (min(src, dst), max(src, dst), proto)

    def extract_from_pcap(self, pcap_path: str) -> Dict:
        """从PCAP文件提取流特征

        Args:
            pcap_path: PCAP文件路径

        Returns:
            Dict: 流数据字典
        """
        flows = defaultdict(lambda: {
            'packets': [],
            'timestamps': [],
            'src_ips': set(),
            'dst_ips': set(),
            'src_ports': set(),
            'dst_ports': set()
        })

        try:
            with open(pcap_path, 'rb') as f:
                if not self.parse_pcap_header(f):
                    logger.error(f"无效的PCAP文件: {pcap_path}")
                    return flows

                while True:
                    packet_info = self.parse_packet(f)
                    if packet_info is None:
                        break

                    flow_key = self.extract_flow_key(packet_info)
                    flow = flows[flow_key]

                    flow['packets'].append(packet_info['length'])
                    flow['timestamps'].append(packet_info['timestamp'])
                    flow['src_ips'].add(packet_info['src_ip'])
                    flow['dst_ips'].add(packet_info['dst_ip'])
                    flow['src_ports'].add(packet_info['src_port'])
                    flow['dst_ports'].add(packet_info['dst_port'])

                    if len(flow['packets']) > self.max_packets * 2:
                        break

        except FileNotFoundError:
            logger.error(f"文件不存在: {pcap_path}")
        except Exception as e:
            logger.error(f"解析PCAP错误: {pcap_path}, {e}")

        return flows

    def extract_flow_features(self, packet_lengths: List[int],
                             max_packets: Optional[int] = None) -> np.ndarray:
        """从数据包长度列表提取特征向量

        Args:
            packet_lengths: 数据包长度列表
            max_packets: 最大数据包数量

        Returns:
            np.ndarray: 归一化特征向量
        """
        if max_packets is None:
            max_packets = self.max_packets

        features = packet_lengths[:max_packets]
        while len(features) < max_packets:
            features.append(0)

        features = np.array(features[:max_packets], dtype=np.float32)

        max_val = np.max(features)
        if max_val > 0:
            features = features / max_val

        return features


class DataPreprocessor:
    """数据预处理器"""

    def __init__(self, max_packets=100):
        self.max_packets = max_packets
        self.label_map = {
            'Video': 0,
            'Chat': 1,
            'FileTransfer': 2,
            'Browsing': 3
        }
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}

    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """归一化特征到[0, 1]区间"""
        if isinstance(features, list):
            features = np.array(features, dtype=np.float32)

        features = np.array(features, dtype=np.float32)

        if len(features.shape) == 1:
            max_val = np.max(features)
            if max_val > 0:
                features = features / max_val
        elif len(features.shape) == 2:
            max_val = np.max(features, axis=1, keepdims=True)
            max_val = np.where(max_val > 0, max_val, 1)
            features = features / max_val

        return features

    def pad_sequences(self, features: np.ndarray,
                      max_length: Optional[int] = None) -> np.ndarray:
        """填充序列到固定长度"""
        if max_length is None:
            max_length = self.max_packets

        if len(features.shape) == 1:
            padded = np.zeros(max_length, dtype=np.float32)
            length = min(len(features), max_length)
            padded[:length] = features[:length]
            return padded

        elif len(features.shape) == 2:
            batch_size = features.shape[0]
            padded = np.zeros((batch_size, max_length), dtype=np.float32)
            for i, feat in enumerate(features):
                length = min(len(feat), max_length)
                padded[i, :length] = feat[:length]
            return padded

        return features

    def preprocess_flow(self, packet_lengths: List[int]) -> np.ndarray:
        """预处理单个流"""
        features = np.array(packet_lengths[:self.max_packets], dtype=np.float32)
        while len(features) < self.max_packets:
            features = np.append(features, 0)
        return self.normalize_features(features)

    def create_training_data(self, flows_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """从流数据创建训练数据

        Args:
            flows_data: 流数据列表，每个元素包含 'lengths' 和 'label'

        Returns:
            Tuple: (特征数组, 标签数组)
        """
        X = []
        y = []

        for flow in flows_data:
            if 'lengths' not in flow or 'label' not in flow:
                continue

            features = self.preprocess_flow(flow['lengths'])
            X.append(features)
            y.append(self.label_map.get(flow['label'], 0))

        return np.array(X), np.array(y)

    def split_train_test(self, X: np.ndarray, y: np.ndarray,
                        test_ratio: float = 0.2,
                        random_seed: int = 42) -> Tuple:
        """划分训练集和测试集"""
        np.random.seed(random_seed)
        indices = np.arange(len(X))
        np.random.shuffle(indices)

        split_idx = int(len(X) * (1 - test_ratio))
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]

        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def generate_synthetic_data(num_samples=1000, max_packets=100):
    """生成合成数据用于测试模型

    Args:
        num_samples: 样本数量
        max_packets: 最大数据包数量

    Returns:
        Tuple: (特征数组, 标签数组)
    """
    np.random.seed(42)

    X = []
    y = []

    app_patterns = {
        0: {'mean': 800, 'std': 200, 'name': 'Video'},
        1: {'mean': 100, 'std': 50, 'name': 'Chat'},
        2: {'mean': 500, 'std': 300, 'name': 'FileTransfer'},
        3: {'mean': 300, 'std': 100, 'name': 'Browsing'}
    }

    for label in range(4):
        pattern = app_patterns[label]

        for _ in range(num_samples // 4):
            num_packets = np.random.randint(20, max_packets)
            lengths = np.abs(np.random.normal(pattern['mean'], pattern['std'], num_packets))
            lengths = np.clip(lengths, 40, 1500).astype(int)

            features = lengths[:max_packets]
            while len(features) < max_packets:
                features = np.append(features, 0)

            X.append(features)
            y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    max_val = np.max(X, axis=1, keepdims=True)
    max_val = np.where(max_val > 0, max_val, 1)
    X = X / max_val

    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    return X, y


if __name__ == '__main__':
    print("=" * 60)
    print("流量特征提取与预处理模块测试")
    print("=" * 60)

    preprocessor = DataPreprocessor(max_packets=100)

    print("\n1. 测试合成数据生成...")
    X, y = generate_synthetic_data(num_samples=400)
    print(f"   生成数据形状: X={X.shape}, y={y.shape}")
    print(f"   标签分布: {np.bincount(y)}")

    print("\n2. 测试数据划分...")
    X_train, X_test, y_train, y_test = preprocessor.split_train_test(X, y)
    print(f"   训练集: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"   测试集: X_test={X_test.shape}, y_test={y_test.shape}")

    print("\n3. 特征统计:")
    print(f"   X_train 范围: [{X_train.min():.4f}, {X_train.max():.4f}]")
    print(f"   X_train 均值: {X_train.mean():.4f}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
