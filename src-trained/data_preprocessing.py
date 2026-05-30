"""
数据预处理模块 - 支持真实PCAP数据集处理
实现按五元组切分流、特征提取、数据清洗和归一化
"""

import os
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from scapy.all import rdpcap, IP, TCP, UDP

from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PCAPFlowExtractor:
    """从PCAP文件提取流特征（基于scapy，支持pcap/pcapng）"""

    def __init__(self, max_packets=MAX_PACKETS, min_packets=MIN_PACKETS):
        self.max_packets = max_packets
        self.min_packets = min_packets
        self.flows = {}
        self.flow_labels = {}

    def parse_packet_scapy(self, packet):
        """使用scapy解析数据包，自动支持pcap/pcapng

        Args:
            packet: scapy数据包对象

        Returns:
            dict: 数据包信息字典
        """
        if not IP in packet:
            return None

        ip_layer = packet[IP]
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        length = len(packet)
        timestamp = float(packet.time)

        protocol = ip_layer.proto

        if protocol == 6 and TCP in packet:
            src_port = packet[TCP].sport
            dst_port = packet[TCP].dport
        elif protocol == 17 and UDP in packet:
            src_port = packet[UDP].sport
            dst_port = packet[UDP].dport
        else:
            return None

        return {
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'src_port': src_port,
            'dst_port': dst_port,
            'protocol': protocol,
            'length': length,
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
        """从PCAP文件提取流特征（支持pcap/pcapng）

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
            from scapy.utils import RawPcapReader
            import time
            
            start_time = time.time()
            packet_count = 0
            flow_counts = 0
            
            # 使用流式读取，避免一次性加载全部数据包
            with RawPcapReader(pcap_path) as pcap_reader:
                for p in pcap_reader:
                    packet_bytes, metadata = p
                    packet_count += 1
                    
                    # 每处理10000个包输出一次进度
                    if packet_count % 10000 == 0:
                        elapsed = time.time() - start_time
                        logger.info(f"  正在处理 {pcap_path}: {packet_count} 个包, {elapsed:.2f}秒")
                    
                    # 快速过滤：数据包太小或不是以太网帧
                    if len(packet_bytes) < 14:
                        continue
                    
                    # 以太网帧类型（偏移12-13）
                    eth_type = (packet_bytes[12] << 8) | packet_bytes[13]
                    
                    # 只处理IPv4 (0x0800)
                    if eth_type != 0x0800:
                        continue
                    
                    # IP头部最小长度20字节
                    if len(packet_bytes) < 34:  # 14以太网 + 20IP
                        continue
                    
                    # IP版本和长度 (偏移14)
                    ip_ver_ihl = packet_bytes[14]
                    ip_version = (ip_ver_ihl >> 4) & 0x0F
                    ip_ihl = (ip_ver_ihl & 0x0F) * 4
                    
                    if ip_version != 4 or ip_ihl < 20:
                        continue
                    
                    # IP总长度（偏移16-17）
                    total_length = (packet_bytes[16] << 8) | packet_bytes[17]
                    
                    # IP协议（偏移23）
                    protocol = packet_bytes[23]
                    
                    # 只处理TCP(6)和UDP(17)
                    if protocol not in (6, 17):
                        continue
                    
                    # TCP/UDP头部至少8字节
                    transport_offset = 14 + ip_ihl
                    if len(packet_bytes) < transport_offset + 8:
                        continue
                    
                    # 端口号
                    src_port = (packet_bytes[transport_offset] << 8) | packet_bytes[transport_offset + 1]
                    dst_port = (packet_bytes[transport_offset + 2] << 8) | packet_bytes[transport_offset + 3]
                    
                    # IP地址
                    ip_start = 14
                    src_ip = f"{packet_bytes[ip_start + 12]}.{packet_bytes[ip_start + 13]}.{packet_bytes[ip_start + 14]}.{packet_bytes[ip_start + 15]}"
                    dst_ip = f"{packet_bytes[ip_start + 16]}.{packet_bytes[ip_start + 17]}.{packet_bytes[ip_start + 18]}.{packet_bytes[ip_start + 19]}"
                    
                    # 构建flow key
                    src = f"{src_ip}:{src_port}"
                    dst = f"{dst_ip}:{dst_port}"
                    if src > dst:
                        flow_key = (dst, src, protocol)
                    else:
                        flow_key = (src, dst, protocol)
                    
                    # 更新流数据
                    if flow_key not in self.flows:
                        self.flows[flow_key] = {
                            'lengths': [],
                            'timestamps': [],
                            'labels': set()
                        }
                        flow_counts += 1
                    
                    # 时间戳处理
                    ts = metadata.ts if hasattr(metadata, 'ts') else 0
                    ts_usec = metadata.ts_usec if hasattr(metadata, 'ts_usec') else 0
                    timestamp = float(ts) + float(ts_usec) / 1e6
                    
                    self.flows[flow_key]['lengths'].append(total_length)
                    self.flows[flow_key]['timestamps'].append(timestamp)
                    self.flows[flow_key]['labels'].add(label)
            
            elapsed = time.time() - start_time
            self.flow_labels[label] = self.flow_labels.get(label, 0) + flow_counts
            logger.info(f"从 {pcap_path} 提取了 {flow_counts} 个流, 共 {packet_count} 个包, 耗时 {elapsed:.2f}秒")

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


def _extract_single_pcap(args):
    """并行提取单个PCAP文件的流（顶层函数，用于multiprocessing）"""
    pcap_path, label, max_packets, min_packets = args
    
    extractor = PCAPFlowExtractor(max_packets=max_packets, min_packets=min_packets)
    flow_count = extractor.extract_from_pcap(pcap_path, label)
    
    # 只保留有效流并限制数据量（避免返回过大的数据）
    flows_data = {}
    for flow_key, flow_data in extractor.flows.items():
        if len(flow_data['lengths']) >= min_packets:
            # 截断到最大数据包数
            lengths = flow_data['lengths'][:max_packets]
            timestamps = flow_data['timestamps'][:max_packets]
            flows_data[str(flow_key)] = {
                'lengths': lengths,
                'timestamps': timestamps,
                'labels': list(flow_data['labels'])
            }
    
    logger.info(f"进程完成: {pcap_path}, 有效流: {len(flows_data)}")
    return flows_data, flow_count, label


class DataPreprocessor:
    """数据预处理器"""

    def __init__(self, config):
        self.config = config
        self.max_packets = config['MAX_PACKETS']
        self.min_packets = config['MIN_PACKETS']
        self.normalization_method = config['NORMALIZATION_METHOD']
        self.class_map = config['CLASS_MAP']
        self.reverse_class_map = config['REVERSE_CLASS_MAP']
        self.cache_dir = config.get('CACHE_DIR', 'data/cache/')
        self.preprocessed_file = config.get('PREPROCESSED_DATA_FILE', 'preprocessed_data.npz')
        # 全局归一化统计量（所有流统一尺度）
        self.global_min = None
        self.global_max = None
        self.global_mean = None
        self.global_std = None

    def compute_global_stats(self, flows: List[Dict]):
        """计算全局归一化统计量（所有流统一尺度）

        Args:
            flows: 流数据列表
        """
        all_features = []
        for flow in flows:
            features = self.pad_and_truncate(flow['lengths'])
            features = self.clean_features(features)
            all_features.append(features)

        all_features = np.array(all_features, dtype=np.float32)

        if self.normalization_method == 'minmax':
            self.global_min = np.min(all_features)
            self.global_max = np.max(all_features)
            logger.info(f"全局统计量 - min: {self.global_min}, max: {self.global_max}")
        elif self.normalization_method == 'zscore':
            self.global_mean = np.mean(all_features)
            self.global_std = np.std(all_features)
            logger.info(f"全局统计量 - mean: {self.global_mean}, std: {self.global_std}")

    def normalize_features(self, features: np.ndarray, use_global: bool = True) -> np.ndarray:
        """归一化特征

        Args:
            features: 特征数组
            use_global: 是否使用全局统计量（默认True，全局归一化）

        Returns:
            np.ndarray: 归一化后的特征数组
        """
        if isinstance(features, list):
            features = np.array(features, dtype=np.float32)

        if use_global and self.normalization_method == 'minmax':
            if self.global_min is not None and self.global_max is not None:
                if self.global_max - self.global_min > 0:
                    features = (features - self.global_min) / (self.global_max - self.global_min)
                else:
                    features = np.zeros_like(features)
            else:
                # 回退到样本内归一化（不应该发生）
                min_val = np.min(features)
                max_val = np.max(features)
                if max_val - min_val > 0:
                    features = (features - min_val) / (max_val - min_val)
                else:
                    features = np.zeros_like(features)
        elif use_global and self.normalization_method == 'zscore':
            if self.global_mean is not None and self.global_std is not None:
                if self.global_std > 0:
                    features = (features - self.global_mean) / self.global_std
                else:
                    features = np.zeros_like(features)
            else:
                mean = np.mean(features)
                std = np.std(features)
                if std > 0:
                    features = (features - mean) / std
                else:
                    features = np.zeros_like(features)
        elif self.normalization_method == 'minmax':
            # 非全局模式（保留兼容性）
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

    def preprocess_flow(self, flow_data: Dict) -> np.ndarray:
        """预处理单个流（使用更丰富的特征）

        Args:
            flow_data: 流数据字典，包含 'lengths' 和可选的 'directions', 'timestamps'

        Returns:
            np.ndarray: 预处理后的特征向量
        """
        packet_lengths = flow_data.get('lengths', [])
        
        # 获取方向信息（1表示正向，-1表示反向）
        directions = flow_data.get('directions', [1] * len(packet_lengths))
        
        # 获取时间戳信息
        timestamps = flow_data.get('timestamps', [i * 0.001 for i in range(len(packet_lengths))])
        
        # 计算时间间隔特征
        time_intervals = []
        for i in range(1, len(timestamps)):
            interval = timestamps[i] - timestamps[i-1]
            time_intervals.append(min(interval, 1.0))  # 限制最大间隔为1秒
        if not time_intervals:
            time_intervals = [0.0]
        
        # 统计特征
        lengths_np = np.array(packet_lengths)
        if len(lengths_np) > 0:
            mean_len = np.mean(lengths_np)
            std_len = np.std(lengths_np)
            max_len = np.max(lengths_np)
            min_len = np.min(lengths_np)
            ratio_up = sum(1 for d in directions if d > 0) / len(directions)
        else:
            mean_len = std_len = max_len = min_len = ratio_up = 0
        
        # 主特征：长度序列
        features_len = self.pad_and_truncate(packet_lengths)
        features_len = self.clean_features(features_len)
        features_len = self.normalize_features(features_len)
        
        # 方向特征
        features_dir = self.pad_and_truncate(directions)
        features_dir = np.array(features_dir, dtype=np.float32)
        
        # 时间间隔特征（归一化到 [0, 1]）
        features_time = self.pad_and_truncate(time_intervals)
        features_time = np.array(features_time, dtype=np.float32)
        
        # 组合特征：长度 + 方向 + 时间
        # 将方向和时间特征缩放到与长度特征相似的范围
        features_dir = (features_dir + 1) / 2  # [-1, 1] -> [0, 1]
        
        # 组合成多通道特征（长度、方向、时间）
        combined = np.stack([features_len, features_dir, features_time], axis=-1)
        
        return combined

    def create_dataset_from_flows(self, flows: List[Dict], labels: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """从流数据创建数据集

        Args:
            flows: 流数据列表
            labels: 标签列表

        Returns:
            Tuple: (特征数组, 标签数组)
        """
        # 先计算全局归一化统计量（所有流统一尺度）
        self.compute_global_stats(flows)

        X = []
        y = []

        for flow, label in zip(flows, labels):
            features = self.preprocess_flow(flow)
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

        train_end = int(len(X) * self.config.get('TRAIN_RATIO', 0.7))
        val_end = train_end + int(len(X) * self.config.get('VAL_RATIO', 0.15))

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
        
        # 并行提取PCAP文件（限制为2个进程，避免电脑卡死）
        num_workers = min(2, len(pcap_files))
        logger.info(f"使用 {num_workers} 个进程并行处理...")
        
        extractor = PCAPFlowExtractor(
            max_packets=self.max_packets,
            min_packets=self.min_packets
        )
        
        # 准备并行任务参数
        task_args = [
            (pcap_path, label, self.max_packets, self.min_packets)
            for pcap_path, label in zip(pcap_files, labels)
        ]
        
        # 并行处理
        completed_count = 0
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_extract_single_pcap, args): args for args in task_args}
            
            for future in as_completed(futures):
                try:
                    pcap_path = futures[future][0]  # 获取对应的pcap路径
                    logger.info(f"正在处理文件: {os.path.basename(pcap_path)}")
                    
                    flows_data, flow_count, label = future.result()
                    
                    # 合并结果到extractor
                    for flow_key_str, flow_data in flows_data.items():
                        try:
                            # 安全解析flow_key（避免eval的安全风险）
                            flow_key_tuple = tuple(eval(flow_key_str))
                            extractor.flows[flow_key_tuple] = flow_data
                        except:
                            # 如果解析失败，跳过这个流
                            continue
                    
                    extractor.flow_labels[label] = extractor.flow_labels.get(label, 0) + flow_count
                    
                    completed_count += 1
                    logger.info(f"完成 {completed_count}/{len(pcap_files)}: {os.path.basename(pcap_path)}, 提取了 {flow_count} 个流")
                    
                except Exception as e:
                    logger.error(f"并行处理PCAP文件出错: {e}")
                    import traceback
                    logger.error(f"详细错误: {traceback.format_exc()}")

        valid_flows = extractor.get_valid_flows()
        logger.info(f"共提取 {len(valid_flows)} 个有效流")

        all_flows = []
        all_labels = []

        flow_labels = {}
        for flow_key, flow_data in extractor.flows.items():
            if len(flow_data['lengths']) >= self.min_packets:
                # 使用流数据中存储的标签
                if 'labels' in flow_data and flow_data['labels']:
                    # 取第一个标签
                    label = list(flow_data['labels'])[0]
                    all_flows.append(flow_data)
                    all_labels.append(self.class_map.get(label, 0))
                    flow_labels[label] = flow_labels.get(label, 0) + 1

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

    def load_or_preprocess(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """加载预处理数据或重新生成

        Returns:
            Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        cache_path = os.path.join(self.cache_dir, self.preprocessed_file)
        
        # 尝试加载缓存
        cached_dataset = self.load_processed_dataset(cache_path)
        if cached_dataset is not None:
            return (
                cached_dataset['X_train'],
                cached_dataset['X_val'],
                cached_dataset['X_test'],
                cached_dataset['y_train'],
                cached_dataset['y_val'],
                cached_dataset['y_test']
            )
        
        # 尝试从PCAP文件加载
        pcap_files = self.config.get('PCAP_FILES', PCAP_FILES)
        
        # 根据文件名模式正确分配标签
        labels = []
        for f in pcap_files:
            basename = os.path.basename(f).lower()
            if any(x in basename for x in ['netflix', 'youtube', 'vimeo', 'spotify']):
                labels.append('Video')
            elif any(x in basename for x in ['chat', 'gmailchat']):
                labels.append('Chat')
            elif any(x in basename for x in ['ftps', 'sftp', 'scp', 'skype_file']):
                labels.append('FileTransfer')
            elif any(x in basename for x in ['email', 'audio']):
                labels.append('Web')
            else:
                labels.append('Video')  # 默认Video
        
        dataset = self.load_dataset(pcap_files, labels)
        
        # 保存到缓存
        self.save_dataset(dataset, cache_path)
        
        return (
            dataset['X_train'],
            dataset['X_val'],
            dataset['X_test'],
            dataset['y_train'],
            dataset['y_val'],
            dataset['y_test']
        )


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
