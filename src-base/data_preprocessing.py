"""
数据预处理模块 - 支持真实PCAP数据集处理
实现按五元组切分流、特征提取、数据清洗和归一化
"""

import os
import struct
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
import logging

from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PCAPFlowExtractor:
    """从PCAP文件提取流特征"""

    def __init__(self, max_packets=MAX_PACKETS, min_packets=MIN_PACKETS):
        self.max_packets = max_packets
        self.min_packets = min_packets
        self.flows = {}
        self.flow_labels = {}

    def parse_pcap_header(self, f):
        """解析PCAP文件头"""
        header = f.read(24)
        if len(header) < 24:
            return None

        magic = struct.unpack('<I', header[0:4])[0]
        if magic == 0xa1b2c3d4:
            self.byte_order = '<'
        elif magic == 0xd4c3b2a1:
            self.byte_order = '>'
        else:
            logger.error(f"无效的PCAP文件头: {magic}")
            return None

        version_major, version_minor = struct.unpack(f'{self.byte_order}HH', header[4:8])
        thiszone = struct.unpack(f'{self.byte_order}I', header[8:12])[0]
        sigfigs = struct.unpack(f'{self.byte_order}I', header[12:16])[0]
        snaplen = struct.unpack(f'{self.byte_order}I', header[16:20])[0]
        network = struct.unpack(f'{self.byte_order}I', header[20:24])[0]

        logger.debug(f"PCAP版本: {version_major}.{version_minor}")
        logger.debug(f"网络类型: {network}")

        return True

    def parse_packet(self, f):
        """解析单个数据包"""
        header = f.read(16)
        if len(header) < 16:
            return None

        ts_sec = struct.unpack(f'{self.byte_order}I', header[0:4])[0]
        ts_usec = struct.unpack(f'{self.byte_order}I', header[4:8])[0]
        incl_len = struct.unpack(f'{self.byte_order}I', header[8:12])[0]
        orig_len = struct.unpack(f'{self.byte_order}I', header[12:16])[0]

        packet_data = f.read(incl_len)
        if len(packet_data) < incl_len:
            return None

        timestamp = ts_sec + ts_usec / 1000000.0

        if incl_len < 14:
            return None

        ether_type = struct.unpack('!H', packet_data[12:14])[0]

        if ether_type == 0x0800:
            ip_header = packet_data[14:]
            return self.parse_ip_packet(ip_header, incl_len, timestamp)
        elif ether_type == 0x0806:
            return None
        else:
            return None

    def parse_ip_packet(self, ip_header, packet_length, timestamp):
        """解析IP数据包"""
        if len(ip_header) < 20:
            return None

        version = (ip_header[0] >> 4) & 0xF
        ihl = (ip_header[0] & 0xF) * 4

        if version != 4:
            return None

        protocol = ip_header[9]

        src_ip = f"{ip_header[12]}.{ip_header[13]}.{ip_header[14]}.{ip_header[15]}"
        dst_ip = f"{ip_header[16]}.{ip_header[17]}.{ip_header[18]}.{ip_header[19]}"

        if len(ip_header) < ihl + 4:
            return None

        if protocol == 6:
            tcp_header = ip_header[ihl:]
            if len(tcp_header) < 20:
                return None

            src_port = struct.unpack('!H', tcp_header[0:2])[0]
            dst_port = struct.unpack('!H', tcp_header[2:4])[0]
        elif protocol == 17:
            udp_header = ip_header[ihl:]
            if len(udp_header) < 8:
                return None

            src_port = struct.unpack('!H', udp_header[0:2])[0]
            dst_port = struct.unpack('!H', udp_header[2:4])[0]
        else:
            return None

        return {
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'src_port': src_port,
            'dst_port': dst_port,
            'protocol': protocol,
            'length': packet_length,
            'timestamp': timestamp
        }

    def generate_flow_key(self, packet_info):
        """生成流标识符（五元组，统一方向）"""
        src = f"{packet_info['src_ip']}:{packet_info['src_port']}"
        dst = f"{packet_info['dst_ip']}:{packet_info['dst_port']}"
        proto = packet_info['protocol']

        if src < dst:
            return (src, dst, proto)
        else:
            return (dst, src, proto)

    def extract_from_pcap(self, pcap_path: str, label: str) -> int:
        """从PCAP文件提取流特征

        Args:
            pcap_path: PCAP文件路径
            label: 流量类别标签

        Returns:
            int: 提取的流数量
        """
        if not os.path.exists(pcap_path):
            logger.warning(f"PCAP文件不存在: {pcap_path}")
            return 0

        try:
            with open(pcap_path, 'rb') as f:
                if not self.parse_pcap_header(f):
                    logger.error(f"无法解析PCAP文件头: {pcap_path}")
                    return 0

                flow_counts = 0

                while True:
                    packet_info = self.parse_packet(f)
                    if packet_info is None:
                        break

                    flow_key = self.generate_flow_key(packet_info)

                    if flow_key not in self.flows:
                        self.flows[flow_key] = {
                            'lengths': [],
                            'timestamps': [],
                            'src_ips': set(),
                            'dst_ips': set()
                        }
                        flow_counts += 1

                    flow = self.flows[flow_key]
                    flow['lengths'].append(packet_info['length'])
                    flow['timestamps'].append(packet_info['timestamp'])
                    flow['src_ips'].add(packet_info['src_ip'])
                    flow['dst_ips'].add(packet_info['dst_ip'])

                    if len(flow['lengths']) >= self.max_packets * 2:
                        pass

                self.flow_labels[label] = self.flow_labels.get(label, 0) + flow_counts
                logger.info(f"从 {pcap_path} 提取了 {flow_counts} 个流")

                return flow_counts

        except Exception as e:
            logger.error(f"解析PCAP文件错误: {pcap_path}, {e}")
            return 0

    def get_valid_flows(self):
        """获取有效流（数据包数量满足要求）"""
        valid_flows = []
        for flow_key, flow_data in self.flows.items():
            if len(flow_data['lengths']) >= self.min_packets:
                valid_flows.append({
                    'key': flow_key,
                    'lengths': flow_data['lengths'],
                    'timestamps': flow_data['timestamps']
                })
        return valid_flows


class DataPreprocessor:
    """数据预处理器"""

    def __init__(self, config):
        self.config = config
        self.max_packets = config['MAX_PACKETS']
        self.min_packets = config['MIN_PACKETS']
        self.normalization_method = config['NORMALIZATION_METHOD']
        self.class_map = config['CLASS_MAP']
        self.reverse_class_map = config['REVERSE_CLASS_MAP']

    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """归一化特征

        Args:
            features: 特征数组

        Returns:
            np.ndarray: 归一化后的特征数组
        """
        if isinstance(features, list):
            features = np.array(features, dtype=np.float32)

        if self.normalization_method == 'minmax':
            min_val = np.min(features)
            max_val = np.max(features)
            if max_val - min_val > 0:
                features = (features - min_val) / (max_val - min_val)
            else:
                features = np.zeros_like(features)
        elif self.normalization_method == 'zscore':
            mean = np.mean(features)
            std = np.std(features)
            if std > 0:
                features = (features - mean) / std
            else:
                features = np.zeros_like(features)

        return features

    def pad_and_truncate(self, lengths: List[int]) -> np.ndarray:
        """填充或截断序列到固定长度

        Args:
            lengths: 数据包长度列表

        Returns:
            np.ndarray: 固定长度的特征向量
        """
        features = np.zeros(self.max_packets, dtype=np.float32)
        actual_length = min(len(lengths), self.max_packets)
        features[:actual_length] = lengths[:actual_length]
        return features

    def clean_features(self, features: np.ndarray) -> np.ndarray:
        """清洗特征数据（处理异常值）

        Args:
            features: 特征数组

        Returns:
            np.ndarray: 清洗后的特征数组
        """
        features = np.where(features < 0, 0, features)

        max_valid_length = 1500
        features = np.where(features > max_valid_length, max_valid_length, features)

        return features

    def preprocess_flow(self, packet_lengths: List[int]) -> np.ndarray:
        """预处理单个流

        Args:
            packet_lengths: 数据包长度列表

        Returns:
            np.ndarray: 预处理后的特征向量
        """
        features = self.pad_and_truncate(packet_lengths)
        features = self.clean_features(features)
        features = self.normalize_features(features)
        return features

    def create_dataset_from_flows(self, flows: List[Dict], labels: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """从流数据创建数据集

        Args:
            flows: 流数据列表
            labels: 标签列表

        Returns:
            Tuple: (特征数组, 标签数组)
        """
        X = []
        y = []

        for flow, label in zip(flows, labels):
            features = self.preprocess_flow(flow['lengths'])
            X.append(features)
            y.append(label)

        return np.array(X), np.array(y)

    def split_dataset(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """划分数据集

        Args:
            X: 特征数组
            y: 标签数组

        Returns:
            Dict: 包含训练集、验证集、测试集的字典
        """
        indices = np.arange(len(X))
        np.random.seed(42)
        np.random.shuffle(indices)

        train_end = int(len(X) * self.config['TRAIN_RATIO'])
        val_end = train_end + int(len(X) * self.config['VAL_RATIO'])

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        return {
            'X_train': X[train_idx],
            'X_val': X[val_idx],
            'X_test': X[test_idx],
            'y_train': y[train_idx],
            'y_val': y[val_idx],
            'y_test': y[test_idx]
        }

    def load_dataset(self, pcap_files: List[str], labels: List[str]) -> Dict:
        """从PCAP文件加载并处理数据集

        Args:
            pcap_files: PCAP文件路径列表
            labels: 对应的标签列表

        Returns:
            Dict: 包含划分好的数据集
        """
        logger.info("开始从PCAP文件提取流数据...")

        extractor = PCAPFlowExtractor(
            max_packets=self.max_packets,
            min_packets=self.min_packets
        )

        for pcap_path, label in zip(pcap_files, labels):
            extractor.extract_from_pcap(pcap_path, label)

        valid_flows = extractor.get_valid_flows()
        logger.info(f"共提取 {len(valid_flows)} 个有效流")

        all_flows = []
        all_labels = []

        flow_labels = {}
        for flow_key, flow_data in extractor.flows.items():
            if len(flow_data['lengths']) >= self.min_packets:
                for label in labels:
                    if label.lower() in flow_key[0].lower() or label.lower() in flow_key[1].lower():
                        all_flows.append(flow_data)
                        all_labels.append(self.class_map.get(label, 0))
                        flow_labels[label] = flow_labels.get(label, 0) + 1
                        break

        if not all_flows:
            logger.warning("未找到足够的带标签流数据，生成合成数据...")
            return self.generate_synthetic_dataset()

        X, y = self.create_dataset_from_flows(all_flows, all_labels)

        logger.info(f"数据集形状: X={X.shape}, y={y.shape}")
        logger.info(f"标签分布: {np.bincount(y)}")

        return self.split_dataset(X, y)

    def generate_synthetic_dataset(self) -> Dict:
        """生成合成数据集用于测试"""
        logger.info("生成合成数据集...")

        np.random.seed(42)
        X = []
        y = []

        app_patterns = {
            0: {'mean': 800, 'std': 200, 'count': 300},
            1: {'mean': 100, 'std': 50, 'count': 300},
            2: {'mean': 500, 'std': 300, 'count': 300},
            3: {'mean': 300, 'std': 100, 'count': 300}
        }

        for label, pattern in app_patterns.items():
            for _ in range(pattern['count']):
                num_packets = np.random.randint(self.min_packets, self.max_packets + 1)
                lengths = np.abs(np.random.normal(pattern['mean'], pattern['std'], num_packets))
                lengths = np.clip(lengths, 40, 1500).astype(int)
                features = self.preprocess_flow(lengths)
                X.append(features)
                y.append(label)

        X = np.array(X)
        y = np.array(y)

        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

        return self.split_dataset(X, y)

    def save_dataset(self, dataset: Dict, filepath: str):
        """保存数据集到文件

        Args:
            dataset: 数据集字典
            filepath: 保存路径
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(dataset, f)
        logger.info(f"数据集已保存到: {filepath}")

    def load_processed_dataset(self, filepath: str) -> Optional[Dict]:
        """加载已处理的数据集

        Args:
            filepath: 数据文件路径

        Returns:
            Optional[Dict]: 数据集字典
        """
        if not os.path.exists(filepath):
            logger.warning(f"数据文件不存在: {filepath}")
            return None

        with open(filepath, 'rb') as f:
            dataset = pickle.load(f)

        logger.info(f"数据集加载成功: X_train={dataset['X_train'].shape}")
        return dataset


def download_dataset_guide():
    """数据集下载指南"""
    guide = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                         数据集下载指南                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. ISCX VPN-nonVPN 数据集                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ 下载地址: https://www.unb.ca/cic/datasets/vpn.html                          │
│                                                                             │
│ 文件列表:                                                                   │
│   • VPN-nonVPN.zip                                                         │
│   • VPN-nonVPN_TrafficLabels.csv                                           │
│                                                                             │
│ 解压后目录结构:                                                             │
│   data/                                                                    │
│     └── ISCX-VPN/                                                          │
│           ├── VPN_VoIP.pcap                                                │
│           ├── VPN_Video.pcap                                               │
│           ├── VPN_FileTransfer.pcap                                        │
│           ├── nonVPN_Chat.pcap                                             │
│           ├── nonVPN_FileTransfer.pcap                                     │
│           ├── nonVPN_Browsing.pcap                                         │
│           └── nonVPN_Video.pcap                                            │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. CIC-IDS2017 数据集                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ 下载地址: https://www.unb.ca/cic/datasets/ids-2017.html                    │
│                                                                             │
│ 注意: CIC-IDS2017数据集较大(约200GB),建议只下载子集                         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. 使用方法                                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1) 下载数据集并解压到 data/ISCX-VPN/ 目录                                    │
│ 2) 修改 config.py 中的 PCAP_FILES 配置                                      │
│ 3) 运行数据预处理脚本                                                        │
│                                                                             │
│ 如果不想下载真实数据集，代码会自动生成合成数据用于训练                         │
└─────────────────────────────────────────────────────────────────────────────┘
"""
    print(guide)


def main():
    """主函数 - 数据预处理演示"""
    print("=" * 70)
    print("数据预处理模块演示")
    print("=" * 70)

    download_dataset_guide()

    config = {
        'MAX_PACKETS': 100,
        'MIN_PACKETS': 10,
        'NORMALIZATION_METHOD': 'minmax',
        'TRAIN_RATIO': 0.7,
        'VAL_RATIO': 0.15,
        'TEST_RATIO': 0.15,
        'CLASS_MAP': CLASS_MAP,
        'REVERSE_CLASS_MAP': REVERSE_CLASS_MAP
    }

    preprocessor = DataPreprocessor(config)

    print("\n测试合成数据生成...")
    dataset = preprocessor.generate_synthetic_dataset()

    print(f"\n数据集划分:")
    print(f"  训练集: {dataset['X_train'].shape}")
    print(f"  验证集: {dataset['X_val'].shape}")
    print(f"  测试集: {dataset['X_test'].shape}")

    print(f"\n训练集标签分布:")
    unique, counts = np.unique(dataset['y_train'], return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {REVERSE_CLASS_MAP[u]}: {c}")

    preprocessor.save_dataset(dataset, 'data/processed_dataset.pkl')

    print("\n" + "=" * 70)
    print("数据预处理完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
