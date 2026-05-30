"""
在线实时推理模块
将训练好的CNN模型集成到代理服务器，实现实时流量分类
"""

import os
import sys
import torch
import torch.nn.functional as F
import threading
import time
import logging
from collections import defaultdict
from datetime import datetime

from model import TrafficCNN
from Proxy import TCPProxyServer, PacketBuffer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrafficClassifier:
    """流量分类器"""

    APP_LABELS = {
        0: 'Video',
        1: 'Chat',
        2: 'FileTransfer',
        3: 'Browsing'
    }

    def __init__(self, model_path='saved_models/best_model.pth', device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.model = TrafficCNN(input_length=100, num_classes=4)
        self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

        self.classification_history = defaultdict(list)
        self.lock = threading.Lock()

        logger.info(f"流量分类器初始化完成，使用设备: {self.device}")

    def load_model(self, model_path):
        """加载模型"""
        if not os.path.exists(model_path):
            logger.warning(f"模型文件不存在: {model_path}")
            logger.info("将使用未训练的模型（随机权重）")
            return

        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"成功加载模型: {model_path}")
            logger.info(f"模型验证准确率: {checkpoint.get('best_val_acc', 'N/A')}")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")

    def classify(self, features):
        """对流量特征进行分类

        Args:
            features: 归一化的特征向量 (list或np.array, 长度100)

        Returns:
            dict: 包含预测类别、置信度等信息
        """
        with torch.no_grad():
            if isinstance(features, list):
                features = torch.FloatTensor(features)
            elif isinstance(features, np.ndarray):
                features = torch.FloatTensor(features)

            features = features.to(self.device)

            if len(features.shape) == 1:
                features = features.unsqueeze(0)

            outputs = self.model(features)
            probabilities = F.softmax(outputs, dim=1)

            probs = probabilities[0].cpu().numpy()
            predicted_class = int(probs.argmax())
            confidence = float(probs[predicted_class])

            return {
                'class_id': predicted_class,
                'class_name': self.APP_LABELS.get(predicted_class, 'Unknown'),
                'confidence': confidence,
                'probabilities': {
                    self.APP_LABELS[i]: float(probs[i]) for i in range(len(probs))
                }
            }

    def classify_with_history(self, connection_id, packet_buffer):
        """带历史的分类，结合多次预测结果

        Args:
            connection_id: 连接ID
            packet_buffer: 数据包缓冲区

        Returns:
            dict: 分类结果
        """
        features = packet_buffer.get_normalized_features()
        result = self.classify(features)

        with self.lock:
            self.classification_history[connection_id].append(result)

            history = self.classification_history[connection_id][-5:]

            class_counts = defaultdict(int)
            total_conf = defaultdict(float)
            for r in history:
                class_counts[r['class_name']] += 1
                total_conf[r['class_name']] += r['confidence']

            final_class = max(class_counts.items(), key=lambda x: x[1])[0]
            avg_conf = total_conf[final_class] / class_counts[final_class]

            return {
                'class_name': final_class,
                'confidence': avg_conf,
                'vote_count': class_counts[final_class],
                'total_predictions': len(history)
            }


class RealtimeProxyServer(TCPProxyServer):
    """集成实时推理的代理服务器"""

    def __init__(self, classifier, listen_host='127.0.0.1',
                 listen_port=8888, max_connections=100,
                 min_packets_for_inference=10):
        super().__init__(
            listen_host=listen_host,
            listen_port=listen_port,
            max_connections=max_connections,
            inference_callback=None
        )

        self.classifier = classifier
        self.min_packets_for_inference = min_packets_for_inference
        self.recent_classifications = []
        self.classification_lock = threading.Lock()

        self.stats['app_distribution'] = defaultdict(int)
        self.stats['connection_classifications'] = {}

    def handle_client(self, client_socket, client_addr):
        """处理客户端连接，重写以添加实时推理"""
        buffer_id = f"{client_addr[0]}:{client_addr[1]}:{time.time()}"
        packet_buffer = PacketBuffer()
        self.packet_buffers[buffer_id] = packet_buffer

        remote_socket = None

        try:
            client_socket.setblocking(False)
            data = b''

            while True:
                try:
                    chunk = client_socket.recv(8192)
                    if not chunk:
                        break
                    data += chunk
                except socket.error:
                    if not data:
                        try:
                            ready, _, _ = select.select([client_socket], [], [], 0.1)
                            if not ready:
                                break
                        except:
                            break
                    else:
                        break

                if remote_socket is None and data:
                    if self.is_http_connect(data):
                        self.handle_http_connect(client_socket, data, client_addr)
                        return
                    else:
                        remote_socket = self.establish_remote_connection(data, client_addr)
                        if remote_socket is None:
                            self.send_error_response(client_socket, 502)
                            return

                if remote_socket and data:
                    remote_socket.sendall(data)
                    packet_buffer.add_packet(len(data), 'up')
                    with self.stats_lock:
                        self.stats['total_bytes_up'] += len(data)

                    self.try_infer(buffer_id, packet_buffer, client_addr)

                    data = b''

                if remote_socket:
                    try:
                        ready, _, _ = select.select([remote_socket], [], [], 0.1)
                        if ready:
                            response = remote_socket.recv(8192)
                            if response:
                                client_socket.sendall(response)
                                packet_buffer.add_packet(len(response), 'down')
                                with self.stats_lock:
                                    self.stats['total_bytes_down'] += len(response)

                                self.try_infer(buffer_id, packet_buffer, client_addr)
                            else:
                                break
                    except socket.error:
                        break

                if packet_buffer.is_ready():
                    self.try_infer(buffer_id, packet_buffer, client_addr)

        except Exception as e:
            logger.error(f"处理客户端 {client_addr} 错误: {e}")
        finally:
            if remote_socket:
                try:
                    remote_socket.close()
                except:
                    pass
            try:
                client_socket.close()
            except:
                pass
            if buffer_id in self.packet_buffers:
                del self.packet_buffers[buffer_id]

    def try_infer(self, buffer_id, packet_buffer, client_addr):
        """尝试进行推理"""
        if len(packet_buffer.packet_lengths) < self.min_packets_for_inference:
            return

        if buffer_id in self.packet_buffers:
            features = packet_buffer.get_normalized_features()
            result = self.classifier.classify(features)

            with self.classification_lock:
                self.stats['connection_classifications'][buffer_id] = {
                    'class_name': result['class_name'],
                    'confidence': result['confidence'],
                    'packet_count': len(packet_buffer.packet_lengths)
                }
                self.stats['app_distribution'][result['class_name']] += 1

                self.recent_classifications.append({
                    'timestamp': datetime.now(),
                    'class_name': result['class_name'],
                    'confidence': result['confidence'],
                    'client': f"{client_addr[0]}:{client_addr[1]}",
                    'packet_count': len(packet_buffer.packet_lengths)
                })

                if len(self.recent_classifications) > 100:
                    self.recent_classifications = self.recent_classifications[-100:]

            self.print_classification(result, client_addr)

    def print_classification(self, result, client_addr):
        """打印分类结果"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        logger.info(
            f"[{timestamp}] {client_addr[0]}:{client_addr[1]} | "
            f"Stream Type: {result['class_name']} | "
            f"Confidence: {result['confidence']:.2%} | "
            f"Probs: {', '.join([f'{k}:{v:.2%}' for k, v in result['probabilities'].items()])}"
        )


class InferenceEngine:
    """推理引擎，管理分类器和实时统计"""

    def __init__(self, model_path='saved_models/best_model.pth'):
        self.classifier = TrafficClassifier(model_path=model_path)
        self.app_stats = defaultdict(lambda: {
            'count': 0,
            'total_confidence': 0.0,
            'last_seen': None
        })
        self.lock = threading.Lock()
        self.running = True

    def update_stats(self, classification_result):
        """更新统计信息"""
        with self.lock:
            class_name = classification_result['class_name']
            confidence = classification_result['confidence']

            stats = self.app_stats[class_name]
            stats['count'] += 1
            stats['total_confidence'] += confidence
            stats['last_seen'] = datetime.now()
            stats['avg_confidence'] = stats['total_confidence'] / stats['count']

    def get_distribution(self):
        """获取应用分布"""
        with self.lock:
            total = sum(s['count'] for s in self.app_stats.values())
            if total == 0:
                return {}

            distribution = {}
            for app, stats in self.app_stats.items():
                distribution[app] = {
                    'count': stats['count'],
                    'percentage': stats['count'] / total * 100,
                    'avg_confidence': stats.get('avg_confidence', 0)
                }
            return distribution

    def get_stats_summary(self):
        """获取统计摘要"""
        with self.lock:
            return dict(self.app_stats)


def run_proxy_with_inference(model_path='saved_models/best_model.pth',
                              listen_host='127.0.0.1',
                              listen_port=8888):
    """运行带实时推理的代理服务器"""
    import argparse

    logger.info("=" * 60)
    logger.info("启动实时流量分类代理服务器")
    logger.info("=" * 60)

    classifier = TrafficClassifier(model_path=model_path)

    server = RealtimeProxyServer(
        classifier=classifier,
        listen_host=listen_host,
        listen_port=listen_port,
        max_connections=100,
        min_packets_for_inference=10
    )

    def stats_printer():
        """定期打印统计信息"""
        while server.running:
            time.sleep(10)
            stats = server.get_statistics()
            if stats['app_distribution']:
                logger.info("\n" + "=" * 40)
                logger.info("流量分布统计 (最近10秒)")
                logger.info("=" * 40)
                total = sum(stats['app_distribution'].values())
                for app, count in sorted(
                    stats['app_distribution'].items(),
                    key=lambda x: -x[1]
                ):
                    pct = count / total * 100 if total > 0 else 0
                    logger.info(f"  {app}: {count} ({pct:.1f}%)")
                logger.info("=" * 40 + "\n")

    stats_thread = threading.Thread(target=stats_printer, daemon=True)
    stats_thread.start()

    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("收到停止信号")
        server.stop()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='实时流量分类代理服务器')
    parser.add_argument('--host', default='127.0.0.1', help='监听地址')
    parser.add_argument('--port', type=int, default=8888, help='监听端口')
    parser.add_argument('--model', type=str,
                        default='saved_models/best_model.pth',
                        help='模型文件路径')

    args = parser.parse_args()

    run_proxy_with_inference(
        model_path=args.model,
        listen_host=args.host,
        listen_port=args.port
    )


if __name__ == '__main__':
    main()
