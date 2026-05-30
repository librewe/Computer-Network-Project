"""
TCP/HTTP 代理服务器
功能：接收客户端请求，转发至目标服务器，返回响应
集成实时流量特征提取功能
"""

import socket
import threading
import select
import time
import logging
from collections import deque
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PacketBuffer:
    """数据包缓冲区，用于存储连接的前N个数据包长度"""

    def __init__(self, max_packets=100):
        self.max_packets = max_packets
        self.packet_lengths = []
        self.timestamps = []
        self.client_ip = None
        self.server_ip = None
        self.client_port = None
        self.server_port = None
        self.protocol = None
        self.start_time = None

    def add_packet(self, length, direction, timestamp=None):
        """添加数据包长度到缓冲区

        Args:
            length: 数据包长度
            direction: 'up' (客户端到服务器) 或 'down' (服务器到客户端)
            timestamp: 数据包时间戳
        """
        if timestamp is None:
            timestamp = time.time()

        if self.start_time is None:
            self.start_time = timestamp

        self.packet_lengths.append(length)
        self.timestamps.append(timestamp)

        if len(self.packet_lengths) > self.max_packets:
            self.packet_lengths = self.packet_lengths[-self.max_packets:]
            self.timestamps = self.timestamps[-self.max_packets:]

    def is_ready(self):
        """检查是否有足够的数据包进行分类"""
        return len(self.packet_lengths) >= 10

    def get_features(self):
        """获取特征向量，不足补0，超过截断"""
        features = self.packet_lengths[:self.max_packets]
        while len(features) < self.max_packets:
            features.append(0)
        return features[:self.max_packets]

    def get_normalized_features(self):
        """获取归一化后的特征向量"""
        features = self.get_features()
        if not features or max(features) == 0:
            return [0.0] * self.max_packets
        max_val = max(features)
        return [f / max_val for f in features]


class TCPProxyServer:
    """TCP代理服务器"""

    def __init__(self, listen_host='127.0.0.1', listen_port=8888,
                 max_connections=100, inference_callback=None):
        """
        初始化代理服务器

        Args:
            listen_host: 监听主机地址
            listen_port: 监听端口
            max_connections: 最大并发连接数
            inference_callback: 推理回调函数，用于实时分类
        """
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.max_connections = max_connections
        self.inference_callback = inference_callback
        self.server_socket = None
        self.running = False
        self.active_connections = {}
        self.connection_lock = threading.Lock()
        self.packet_buffers = {}
        self.stats = {
            'total_connections': 0,
            'total_bytes_up': 0,
            'total_bytes_down': 0,
            'classifications': {}
        }
        self.stats_lock = threading.Lock()

    def start(self):
        """启动代理服务器"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.listen_host, self.listen_port))
            self.server_socket.listen(self.max_connections)
            self.running = True

            logger.info(f"代理服务器启动成功!")
            logger.info(f"监听地址: {self.listen_host}:{self.listen_port}")
            logger.info(f"请将浏览器或系统代理设置为: {self.listen_host}:{self.listen_port}")
            logger.info("按 Ctrl+C 停止服务器")

            self.accept_loop()
        except Exception as e:
            logger.error(f"服务器启动失败: {e}")
            raise

    def accept_loop(self):
        """接受连接的主循环"""
        while self.running:
            try:
                client_socket, client_addr = self.server_socket.accept()
                logger.info(f"收到连接: {client_addr[0]}:{client_addr[1]}")

                with self.connection_lock:
                    self.stats['total_connections'] += 1

                client_thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_socket, client_addr),
                    daemon=True
                )
                client_thread.start()

            except Exception as e:
                if self.running:
                    logger.error(f"接受连接错误: {e}")

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

                    self.check_and_infer(buffer_id, packet_buffer)

                    if self.inference_callback:
                        self.inference_callback(buffer_id, packet_buffer)

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

                                self.check_and_infer(buffer_id, packet_buffer)

                                if self.inference_callback:
                                    self.inference_callback(buffer_id, packet_buffer)
                            else:
                                break
                    except socket.error:
                        break

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

    def is_http_connect(self, data):
        """检查是否为HTTP CONNECT请求"""
        try:
            return data.startswith(b'CONNECT')
        except:
            return False

    def handle_http_connect(self, client_socket, data, client_addr):
        """处理HTTP CONNECT隧道请求"""
        try:
            lines = data.decode('utf-8', errors='ignore').split('\r\n')
            if not lines:
                return

            request_line = lines[0]
            parts = request_line.split(' ')
            if len(parts) < 3:
                return

            host_port = parts[1]
            if ':' in host_port:
                host, port = host_port.split(':')
                port = int(port)
            else:
                host = host_port
                port = 80

            logger.info(f"HTTP CONNECT: {host}:{port}")

            remote_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            remote_socket.settimeout(10)
            remote_socket.connect((host, port))

            client_socket.sendall(b'HTTP/1.1 200 Connection Established\r\n\r\n')

            client_socket.setblocking(False)
            remote_socket.setblocking(False)

            buffer_id = f"{client_addr[0]}:{client_addr[1]}:{time.time()}"
            packet_buffer = PacketBuffer()
            self.packet_buffers[buffer_id] = packet_buffer

            self.tunnel_data(client_socket, remote_socket, buffer_id, packet_buffer)

        except Exception as e:
            logger.error(f"HTTP CONNECT处理错误: {e}")
        finally:
            try:
                client_socket.close()
            except:
                pass

    def tunnel_data(self, client_socket, remote_socket, buffer_id, packet_buffer):
        """隧道传输数据"""
        sockets = [client_socket, remote_socket]

        try:
            while True:
                ready, _, _ = select.select(sockets, [], [], 0.5)

                for sock in ready:
                    try:
                        data = sock.recv(8192)
                        if not data:
                            return

                        if sock is client_socket:
                            remote_socket.sendall(data)
                            packet_buffer.add_packet(len(data), 'up')
                            with self.stats_lock:
                                self.stats['total_bytes_up'] += len(data)
                        else:
                            client_socket.sendall(data)
                            packet_buffer.add_packet(len(data), 'down')
                            with self.stats_lock:
                                self.stats['total_bytes_down'] += len(data)

                        self.check_and_infer(buffer_id, packet_buffer)

                        if self.inference_callback:
                            self.inference_callback(buffer_id, packet_buffer)

                    except socket.error:
                        return

        except Exception as e:
            logger.error(f"隧道传输错误: {e}")
        finally:
            try:
                client_socket.close()
            except:
                pass
            try:
                remote_socket.close()
            except:
                pass
            if buffer_id in self.packet_buffers:
                del self.packet_buffers[buffer_id]

    def establish_remote_connection(self, data, client_addr):
        """建立到远程服务器的连接"""
        try:
            host, port = self.parse_host_from_request(data)
            if not host:
                return None

            logger.info(f"建立连接: {host}:{port}")

            remote_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            remote_socket.settimeout(30)
            remote_socket.connect((host, port))

            return remote_socket
        except Exception as e:
            logger.error(f"建立远程连接失败: {e}")
            return None

    def parse_host_from_request(self, data):
        """从HTTP请求中解析主机和端口"""
        try:
            lines = data.decode('utf-8', errors='ignore').split('\r\n')
            for line in lines:
                if line.lower().startswith('host:'):
                    host_port = line[5:].strip()
                    if ':' in host_port:
                        parts = host_port.split(':')
                        return parts[0], int(parts[1])
                    else:
                        return host_port, 80
            return None, None
        except:
            return None, None

    def send_error_response(self, client_socket, code):
        """发送错误响应"""
        messages = {
            502: b'HTTP/1.1 502 Bad Gateway\r\n\r\n',
            500: b'HTTP/1.1 500 Internal Server Error\r\n\r\n'
        }
        try:
            client_socket.sendall(messages.get(code, messages[500]))
        except:
            pass

    def check_and_infer(self, buffer_id, packet_buffer):
        """检查并执行推理"""
        if packet_buffer.is_ready() and self.inference_callback:
            features = packet_buffer.get_normalized_features()
            result = self.inference_callback(buffer_id, packet_buffer)
            if result:
                with self.stats_lock:
                    app_type = result if isinstance(result, str) else 'Unknown'
                    self.stats['classifications'][app_type] = \
                        self.stats['classifications'].get(app_type, 0) + 1

    def get_statistics(self):
        """获取统计信息"""
        with self.stats_lock:
            return dict(self.stats)

    def stop(self):
        """停止代理服务器"""
        logger.info("正在停止代理服务器...")
        self.running = False
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        with self.connection_lock:
            for conn in list(self.active_connections.values()):
                try:
                    conn.close()
                except:
                    pass
            self.active_connections.clear()
        logger.info("代理服务器已停止")


def main():
    """主函数 - 运行代理服务器"""
    import argparse

    parser = argparse.ArgumentParser(description='TCP/HTTP 代理服务器')
    parser.add_argument('--host', default='127.0.0.1', help='监听地址')
    parser.add_argument('--port', type=int, default=8888, help='监听端口')
    parser.add_argument('--max-connections', type=int, default=100, help='最大并发连接数')

    args = parser.parse_args()

    server = TCPProxyServer(
        listen_host=args.host,
        listen_port=args.port,
        max_connections=args.max_connections
    )

    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("收到停止信号")
        server.stop()


if __name__ == '__main__':
    main()
