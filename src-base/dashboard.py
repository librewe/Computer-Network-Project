"""
实时流量监控看板
使用Streamlit开发Web界面，动态展示流量分类结果
"""

import os
import sys
import time
import threading
import logging
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import numpy as np

from model import TrafficCNN
from Proxy import TCPProxyServer, PacketBuffer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DashboardDataManager:
    """仪表板数据管理器"""

    def __init__(self, max_history=200):
        self.max_history = max_history
        self.lock = threading.Lock()

        self.total_connections = 0
        self.total_bytes_up = 0
        self.total_bytes_down = 0
        self.app_distribution = defaultdict(int)
        self.connection_history = []
        self.timeline_data = []
        self.recent_predictions = []

    def update_connection(self, client_addr, app_type, confidence, packet_count):
        """更新连接信息"""
        with self.lock:
            self.total_connections += 1
            self.app_distribution[app_type] += 1

            self.connection_history.append({
                'timestamp': datetime.now(),
                'client': f"{client_addr[0]}:{client_addr[1]}",
                'app_type': app_type,
                'confidence': confidence,
                'packet_count': packet_count
            })

            if len(self.connection_history) > self.max_history:
                self.connection_history = self.connection_history[-self.max_history:]

    def update_traffic(self, bytes_up, bytes_down):
        """更新流量统计"""
        with self.lock:
            self.total_bytes_up += bytes_up
            self.total_bytes_down += bytes_down

    def update_prediction(self, app_type, confidence):
        """更新预测结果"""
        with self.lock:
            self.recent_predictions.append({
                'timestamp': datetime.now(),
                'app_type': app_type,
                'confidence': confidence
            })

            if len(self.recent_predictions) > self.max_history:
                self.recent_predictions = self.recent_predictions[-self.max_history:]

            self.timeline_data.append({
                'timestamp': datetime.now(),
                'app_type': app_type
            })

            if len(self.timeline_data) > self.max_history:
                self.timeline_data = self.timeline_data[-self.max_history:]

    def get_summary(self) -> Dict:
        """获取统计摘要"""
        with self.lock:
            total = sum(self.app_distribution.values())
            return {
                'total_connections': self.total_connections,
                'total_bytes_up': self.total_bytes_up,
                'total_bytes_down': self.total_bytes_down,
                'app_distribution': dict(self.app_distribution),
                'total_predictions': total
            }

    def get_distribution_data(self) -> Dict:
        """获取分布数据"""
        with self.lock:
            total = sum(self.app_distribution.values())
            if total == 0:
                return {}

            distribution = {}
            for app, count in self.app_distribution.items():
                distribution[app] = {
                    'count': count,
                    'percentage': count / total * 100
                }
            return distribution

    def get_timeline_data(self) -> List[Dict]:
        """获取时间线数据"""
        with self.lock:
            return list(self.timeline_data)

    def get_recent_predictions(self) -> List[Dict]:
        """获取最近预测"""
        with self.lock:
            return list(self.recent_predictions[-20:])

    def get_connection_history_df(self):
        """获取连接历史DataFrame"""
        with self.lock:
            if not self.connection_history:
                return pd.DataFrame()
            df = pd.DataFrame(self.connection_history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df


class DashboardProxyServer(TCPProxyServer):
    """带仪表板支持的代理服务器"""

    def __init__(self, data_manager, listen_host='127.0.0.1',
                 listen_port=8888, max_connections=100,
                 min_packets_for_inference=10):
        super().__init__(
            listen_host=listen_host,
            listen_port=listen_port,
            max_connections=max_connections,
            inference_callback=None
        )

        self.data_manager = data_manager
        self.min_packets_for_inference = min_packets_for_inference
        self.classifier = None

    def set_classifier(self, classifier):
        """设置分类器"""
        self.classifier = classifier

    def handle_client(self, client_socket, client_addr):
        """处理客户端连接"""
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
                        self.data_manager.update_traffic(len(data), 0)

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
                                    self.data_manager.update_traffic(0, len(response))

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

        if self.classifier and buffer_id in self.packet_buffers:
            features = packet_buffer.get_normalized_features()
            result = self.classifier.classify(features)

            self.data_manager.update_connection(
                client_addr,
                result['class_name'],
                result['confidence'],
                len(packet_buffer.packet_lengths)
            )
            self.data_manager.update_prediction(
                result['class_name'],
                result['confidence']
            )

            self.print_classification(result, client_addr)

    def print_classification(self, result, client_addr):
        """打印分类结果"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] {client_addr[0]}:{client_addr[1]} | "
              f"Stream Type: {result['class_name']} | "
              f"Confidence: {result['confidence']:.2%}")


class StreamlitDashboard:
    """Streamlit仪表板"""

    def __init__(self, data_manager):
        self.data_manager = data_manager

    def run(self, port=8501):
        """运行仪表板"""
        import threading
        threading.Thread(
            target=lambda: st.set_page_config(
                page_title="流量监控看板",
                page_icon="📊",
                layout="wide"
            ),
            daemon=True
        ).start()

        st.title("📊 实时流量监控看板")
        st.markdown("基于深度学习的加密流量实时识别与监控系统")

        col1, col2, col3, col4 = st.columns(4)

        while True:
            summary = self.data_manager.get_summary()

            with col1:
                st.metric(
                    "总连接数",
                    summary['total_connections']
                )

            with col2:
                st.metric(
                    "上传流量",
                    self.format_bytes(summary['total_bytes_up'])
                )

            with col3:
                st.metric(
                    "下载流量",
                    self.format_bytes(summary['total_bytes_down'])
                )

            with col4:
                st.metric(
                    "总预测数",
                    summary['total_predictions']
                )

            st.markdown("---")

            col_left, col_right = st.columns([1, 1])

            with col_left:
                st.subheader("📈 应用类型分布")

                dist = self.data_manager.get_distribution_data()
                if dist:
                    df = pd.DataFrame([
                        {'应用类型': app, '数量': data['count'],
                         '占比': f"{data['percentage']:.1f}%"}
                        for app, data in dist.items()
                    ])
                    st.dataframe(df, use_container_width=True)

                    fig_pie = px.pie(
                        df,
                        values='数量',
                        names='应用类型',
                        title='流量类型分布',
                        color='应用类型',
                        color_discrete_map={
                            'Video': '#FF6B6B',
                            'Chat': '#4ECDC4',
                            'FileTransfer': '#45B7D1',
                            'Browsing': '#96CEB4'
                        }
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.info("暂无数据，请通过代理服务器产生流量")

            with col_right:
                st.subheader("📊 最近预测记录")

                recent = self.data_manager.get_recent_predictions()
                if recent:
                    df_recent = pd.DataFrame(recent)
                    df_recent['timestamp'] = pd.to_datetime(df_recent['timestamp']).dt.strftime('%H:%M:%S')
                    st.dataframe(df_recent, use_container_width=True)
                else:
                    st.info("暂无预测记录")

            st.markdown("---")

            timeline = self.data_manager.get_timeline_data()
            if timeline:
                st.subheader("📉 流量类型时间线")
                df_timeline = pd.DataFrame(timeline)
                df_timeline['timestamp'] = pd.to_datetime(df_timeline['timestamp'])
                df_timeline = df_timeline.set_index('timestamp')

                app_counts = df_timeline.resample('5s')['app_type'].value_counts().unstack().fillna(0)

                fig_line = go.Figure()
                for app in ['Video', 'Chat', 'FileTransfer', 'Browsing']:
                    if app in app_counts.columns:
                        fig_line.add_trace(go.Scatter(
                            x=app_counts.index,
                            y=app_counts[app],
                            mode='lines',
                            name=app,
                            stackgroup='one'
                        ))

                fig_line.update_layout(
                    title='流量类型时间分布',
                    xaxis_title='时间',
                    yaxis_title='预测数量',
                    hovermode='x unified'
                )
                st.plotly_chart(fig_line, use_container_width=True)

            time.sleep(2)
            st.rerun()

    @staticmethod
    def format_bytes(num_bytes):
        """格式化字节数"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if abs(num_bytes) < 1024.0:
                return f"{num_bytes:.2f} {unit}"
            num_bytes /= 1024.0
        return f"{num_bytes:.2f} TB"


def run_dashboard(port=8501, proxy_port=8888, model_path='saved_models/best_model.pth'):
    """运行仪表板和代理服务器"""

    import socket
    import select

    st.write("正在初始化...")

    data_manager = DashboardDataManager()

    classifier = None
    if os.path.exists(model_path):
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = TrafficCNN(input_length=100, num_classes=4)
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()

            class SimpleClassifier:
                def __init__(self, model, device):
                    self.model = model
                    self.device = device

                def classify(self, features):
                    with torch.no_grad():
                        if isinstance(features, list):
                            features = torch.FloatTensor(features)
                        elif isinstance(features, np.ndarray):
                            features = torch.FloatTensor(features)
                        features = features.to(self.device)
                        if len(features.shape) == 1:
                            features = features.unsqueeze(0)
                        outputs = self.model(features)
                        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
                        pred = int(probs.argmax())
                        labels = {0: 'Video', 1: 'Chat', 2: 'FileTransfer', 3: 'Browsing'}
                        return {
                            'class_name': labels[pred],
                            'confidence': float(probs[pred]),
                            'probabilities': {labels[i]: float(probs[i]) for i in range(4)}
                        }

            classifier = SimpleClassifier(model, device)
            st.success(f"模型加载成功: {model_path}")
        except Exception as e:
            st.error(f"模型加载失败: {e}")
    else:
        st.warning(f"模型文件不存在: {model_path}，将使用随机权重")

    proxy_server = DashboardProxyServer(
        data_manager=data_manager,
        listen_host='127.0.0.1',
        listen_port=proxy_port,
        max_connections=100,
        min_packets_for_inference=10
    )

    if classifier:
        proxy_server.set_classifier(classifier)

    proxy_thread = threading.Thread(target=proxy_server.start, daemon=True)
    proxy_thread.start()

    st.success(f"代理服务器已启动: 127.0.0.1:{proxy_port}")
    st.info("请将浏览器代理设置为 127.0.0.1:8888")

    dashboard = StreamlitDashboard(data_manager)
    dashboard.run(port=port)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='实时流量监控看板')
    parser.add_argument('--dashboard-port', type=int, default=8501,
                        help='仪表板端口')
    parser.add_argument('--proxy-port', type=int, default=8888,
                        help='代理服务器端口')
    parser.add_argument('--model', type=str,
                        default='saved_models/best_model.pth',
                        help='模型文件路径')

    args = parser.parse_args()

    run_dashboard(
        port=args.dashboard_port,
        proxy_port=args.proxy_port,
        model_path=args.model
    )


if __name__ == '__main__':
    main()
