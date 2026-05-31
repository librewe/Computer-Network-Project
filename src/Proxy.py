"""
TCP/HTTP proxy helpers for the dashboard demo.
"""

import logging
import select
import socket
import threading
import time


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class PacketBuffer:
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
        return len(self.packet_lengths) >= 10

    def get_features(self):
        features = self.packet_lengths[: self.max_packets]
        while len(features) < self.max_packets:
            features.append(0)
        return features[: self.max_packets]

    def get_normalized_features(self):
        features = self.get_features()
        if not features or max(features) == 0:
            return [0.0] * self.max_packets
        max_val = max(features)
        return [feature / max_val for feature in features]


class TCPProxyServer:
    def __init__(self, listen_host="127.0.0.1", listen_port=8888, max_connections=100, inference_callback=None):
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
            "total_connections": 0,
            "total_bytes_up": 0,
            "total_bytes_down": 0,
            "classifications": {},
        }
        self.stats_lock = threading.Lock()

    def start(self):
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.settimeout(1.0)
            self.server_socket.bind((self.listen_host, self.listen_port))
            self.server_socket.listen(self.max_connections)
            self.running = True

            logger.info("代理服务器启动成功!")
            logger.info("监听地址: %s:%s", self.listen_host, self.listen_port)
            logger.info("请将浏览器或系统代理设置为: %s:%s", self.listen_host, self.listen_port)
            logger.info("按 Ctrl+C 停止服务器")

            self.accept_loop()
        except Exception as exc:
            logger.error("服务器启动失败: %s", exc)
            raise

    def accept_loop(self):
        while self.running:
            try:
                client_socket, client_addr = self.server_socket.accept()
                logger.info("收到连接: %s:%s", client_addr[0], client_addr[1])

                with self.connection_lock:
                    self.stats["total_connections"] += 1

                client_thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_socket, client_addr),
                    daemon=True,
                )
                client_thread.start()
            except socket.timeout:
                continue
            except OSError:
                break
            except Exception as exc:
                if self.running:
                    logger.error("接受连接错误: %s", exc)

    def handle_client(self, client_socket, client_addr):
        buffer_id = f"{client_addr[0]}:{client_addr[1]}:{time.time()}"
        packet_buffer = PacketBuffer()
        self.packet_buffers[buffer_id] = packet_buffer
        remote_socket = None

        try:
            client_socket.setblocking(False)
            data = b""

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
                        except Exception:
                            break
                    else:
                        break

                if remote_socket is None and data:
                    if self.is_http_connect(data):
                        self.handle_http_connect(client_socket, data, client_addr)
                        return
                    remote_socket = self.establish_remote_connection(data, client_addr)
                    if remote_socket is None:
                        self.send_error_response(client_socket, 502)
                        return

                if remote_socket and data:
                    remote_socket.sendall(data)
                    packet_buffer.add_packet(len(data), "up")
                    with self.stats_lock:
                        self.stats["total_bytes_up"] += len(data)

                    self.check_and_infer(buffer_id, packet_buffer)
                    if self.inference_callback:
                        self.inference_callback(buffer_id, packet_buffer)
                    data = b""

                if remote_socket:
                    try:
                        ready, _, _ = select.select([remote_socket], [], [], 0.1)
                        if ready:
                            response = remote_socket.recv(8192)
                            if response:
                                client_socket.sendall(response)
                                packet_buffer.add_packet(len(response), "down")
                                with self.stats_lock:
                                    self.stats["total_bytes_down"] += len(response)

                                self.check_and_infer(buffer_id, packet_buffer)
                                if self.inference_callback:
                                    self.inference_callback(buffer_id, packet_buffer)
                            else:
                                break
                    except socket.error:
                        break
        except Exception as exc:
            logger.error("处理客户端 %s 错误: %s", client_addr, exc)
        finally:
            if remote_socket:
                try:
                    remote_socket.close()
                except Exception:
                    pass
            try:
                client_socket.close()
            except Exception:
                pass
            self.packet_buffers.pop(buffer_id, None)

    def is_http_connect(self, data):
        try:
            return data.startswith(b"CONNECT")
        except Exception:
            return False

    def handle_http_connect(self, client_socket, data, client_addr):
        remote_socket = None
        try:
            lines = data.decode("utf-8", errors="ignore").split("\r\n")
            if not lines:
                return

            request_line = lines[0]
            parts = request_line.split(" ")
            if len(parts) < 3:
                return

            host_port = parts[1]
            if ":" in host_port:
                host, port = host_port.split(":", 1)
                port = int(port)
            else:
                host = host_port
                port = 80

            logger.info("HTTP CONNECT: %s:%s", host, port)

            remote_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            remote_socket.settimeout(10)
            remote_socket.connect((host, port))

            client_socket.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            client_socket.setblocking(False)
            remote_socket.setblocking(False)

            buffer_id = f"{client_addr[0]}:{client_addr[1]}:{time.time()}"
            packet_buffer = PacketBuffer()
            self.packet_buffers[buffer_id] = packet_buffer

            self.tunnel_data(client_socket, remote_socket, buffer_id, packet_buffer)
        except Exception as exc:
            logger.error("HTTP CONNECT处理错误: %s", exc)
        finally:
            try:
                client_socket.close()
            except Exception:
                pass

    def tunnel_data(self, client_socket, remote_socket, buffer_id, packet_buffer):
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
                            packet_buffer.add_packet(len(data), "up")
                            with self.stats_lock:
                                self.stats["total_bytes_up"] += len(data)
                        else:
                            client_socket.sendall(data)
                            packet_buffer.add_packet(len(data), "down")
                            with self.stats_lock:
                                self.stats["total_bytes_down"] += len(data)

                        self.check_and_infer(buffer_id, packet_buffer)
                        if self.inference_callback:
                            self.inference_callback(buffer_id, packet_buffer)
                    except socket.error:
                        return
        except Exception as exc:
            logger.error("隧道传输错误: %s", exc)
        finally:
            try:
                client_socket.close()
            except Exception:
                pass
            try:
                remote_socket.close()
            except Exception:
                pass
            self.packet_buffers.pop(buffer_id, None)

    def establish_remote_connection(self, data, client_addr):
        try:
            host, port = self.parse_host_from_request(data)
            if not host:
                return None

            logger.info("建立连接: %s:%s", host, port)

            remote_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            remote_socket.settimeout(30)
            remote_socket.connect((host, port))
            return remote_socket
        except Exception as exc:
            logger.error("建立远程连接失败: %s", exc)
            return None

    def parse_host_from_request(self, data):
        try:
            lines = data.decode("utf-8", errors="ignore").split("\r\n")
            for line in lines:
                if line.lower().startswith("host:"):
                    host_port = line[5:].strip()
                    if ":" in host_port:
                        host, port = host_port.split(":", 1)
                        return host, int(port)
                    return host_port, 80
            return None, None
        except Exception:
            return None, None

    def send_error_response(self, client_socket, code):
        messages = {
            502: b"HTTP/1.1 502 Bad Gateway\r\n\r\n",
            500: b"HTTP/1.1 500 Internal Server Error\r\n\r\n",
        }
        try:
            client_socket.sendall(messages.get(code, messages[500]))
        except Exception:
            pass

    def check_and_infer(self, buffer_id, packet_buffer):
        if packet_buffer.is_ready() and self.inference_callback:
            result = self.inference_callback(buffer_id, packet_buffer)
            if result:
                with self.stats_lock:
                    app_type = result if isinstance(result, str) else "Unknown"
                    self.stats["classifications"][app_type] = self.stats["classifications"].get(app_type, 0) + 1

    def get_statistics(self):
        with self.stats_lock:
            return dict(self.stats)

    def stop(self):
        logger.info("正在停止代理服务器...")
        self.running = False
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass
            self.server_socket = None
        with self.connection_lock:
            for conn in list(self.active_connections.values()):
                try:
                    conn.close()
                except Exception:
                    pass
            self.active_connections.clear()
        logger.info("代理服务器已停止")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="TCP/HTTP proxy server")
    parser.add_argument("--host", default="127.0.0.1", help="Listen address")
    parser.add_argument("--port", type=int, default=8888, help="Listen port")
    parser.add_argument("--max-connections", type=int, default=100, help="Max connections")
    args = parser.parse_args()

    server = TCPProxyServer(
        listen_host=args.host,
        listen_port=args.port,
        max_connections=args.max_connections,
    )

    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("收到停止信号")
        server.stop()


if __name__ == "__main__":
    main()
