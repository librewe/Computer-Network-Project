"""
Streamlit dashboard for the traffic proxy demo.
"""

import atexit
import logging
import os
import select
import socket
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from model import create_model, infer_model_type_from_state_dict
from Proxy import PacketBuffer, TCPProxyServer
from runtime_config import get_runtime_model_path, load_runtime_config


RUNTIME_CONFIG = load_runtime_config()
LABELS = {int(k): v for k, v in RUNTIME_CONFIG["labels"].items()}
APP_ORDER = list(RUNTIME_CONFIG["dashboard"]["app_order"])
COLOR_MAP = dict(RUNTIME_CONFIG["dashboard"]["color_map"])
REFRESH_SECONDS = int(RUNTIME_CONFIG["dashboard"].get("refresh_seconds", 2))
ACTIVE_WINDOW_SECONDS = max(REFRESH_SECONDS * 2, 5)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DashboardDataManager:
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
        self.last_activity_at = None

    def update_connection(self, client_addr, app_type, confidence, packet_count):
        with self.lock:
            now = datetime.now()
            self.total_connections += 1
            self.app_distribution[app_type] += 1
            self.connection_history.append(
                {
                    "timestamp": now,
                    "client": f"{client_addr[0]}:{client_addr[1]}",
                    "app_type": app_type,
                    "confidence": confidence,
                    "packet_count": packet_count,
                }
            )
            self.connection_history = self.connection_history[-self.max_history:]
            self.last_activity_at = now

    def update_traffic(self, bytes_up, bytes_down):
        with self.lock:
            if bytes_up > 0 or bytes_down > 0:
                self.last_activity_at = datetime.now()
            self.total_bytes_up += bytes_up
            self.total_bytes_down += bytes_down

    def update_prediction(self, app_type, confidence):
        with self.lock:
            now = datetime.now()
            self.recent_predictions.append(
                {
                    "timestamp": now,
                    "app_type": app_type,
                    "confidence": confidence,
                }
            )
            self.recent_predictions = self.recent_predictions[-self.max_history:]
            self.timeline_data.append({"timestamp": now, "app_type": app_type})
            self.timeline_data = self.timeline_data[-self.max_history:]
            self.last_activity_at = now

    def get_summary(self) -> Dict:
        with self.lock:
            total = sum(self.app_distribution.values())
            return {
                "total_connections": self.total_connections,
                "total_bytes_up": self.total_bytes_up,
                "total_bytes_down": self.total_bytes_down,
                "app_distribution": dict(self.app_distribution),
                "total_predictions": total,
            }

    def get_distribution_data(self) -> Dict:
        with self.lock:
            total = sum(self.app_distribution.values())
            if total == 0:
                return {}
            return {
                app: {"count": count, "percentage": count / total * 100}
                for app, count in self.app_distribution.items()
            }

    def get_timeline_data(self) -> List[Dict]:
        with self.lock:
            return list(self.timeline_data)

    def get_recent_predictions(self) -> List[Dict]:
        with self.lock:
            return list(self.recent_predictions[-20:])

    def get_connection_history_df(self):
        with self.lock:
            if not self.connection_history:
                return pd.DataFrame()
            df = pd.DataFrame(self.connection_history)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df

    def has_recent_activity(self, active_window_seconds):
        with self.lock:
            if self.last_activity_at is None:
                return False
            return (datetime.now() - self.last_activity_at).total_seconds() <= active_window_seconds


class DashboardProxyServer(TCPProxyServer):
    def __init__(
        self,
        data_manager,
        listen_host="127.0.0.1",
        listen_port=8888,
        max_connections=100,
        min_packets_for_inference=10,
    ):
        super().__init__(
            listen_host=listen_host,
            listen_port=listen_port,
            max_connections=max_connections,
            inference_callback=None,
        )
        self.data_manager = data_manager
        self.min_packets_for_inference = min_packets_for_inference
        self.classifier = None
        self.reported_buffers = set()

    def set_classifier(self, classifier):
        self.classifier = classifier

    def try_infer(self, buffer_id, packet_buffer, client_addr, force=False):
        packet_count = len(packet_buffer.packet_lengths)
        if buffer_id in self.reported_buffers:
            return
        if packet_count == 0:
            return
        if not force and packet_count < self.min_packets_for_inference:
            return
        if not self.classifier or buffer_id not in self.packet_buffers:
            return

        result = self.classifier.classify(packet_buffer.get_normalized_features())
        self.data_manager.update_connection(
            client_addr,
            result["class_name"],
            result["confidence"],
            packet_count,
        )
        self.data_manager.update_prediction(result["class_name"], result["confidence"])
        self.reported_buffers.add(buffer_id)
        logger.info(
            "Dashboard classified %s:%s as %s (confidence=%.2f, packets=%s)",
            client_addr[0],
            client_addr[1],
            result["class_name"],
            result["confidence"],
            packet_count,
        )

    def handle_client(self, client_socket, client_addr):
        buffer_id = f"{client_addr[0]}:{client_addr[1]}:{datetime.now().timestamp()}"
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
                        self.handle_http_connect_dashboard(client_socket, data, client_addr)
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
                    self.data_manager.update_traffic(len(data), 0)
                    self.try_infer(buffer_id, packet_buffer, client_addr)
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
                                self.data_manager.update_traffic(0, len(response))
                                self.try_infer(buffer_id, packet_buffer, client_addr)
                            else:
                                break
                    except socket.error:
                        break
        finally:
            self.try_infer(buffer_id, packet_buffer, client_addr, force=True)
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
            self.reported_buffers.discard(buffer_id)

    def handle_http_connect_dashboard(self, client_socket, data, client_addr):
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
                port = 443

            logger.info("HTTP CONNECT: %s:%s", host, port)

            remote_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            remote_socket.settimeout(10)
            remote_socket.connect((host, port))
            client_socket.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")

            client_socket.setblocking(False)
            remote_socket.setblocking(False)

            buffer_id = f"{client_addr[0]}:{client_addr[1]}:{datetime.now().timestamp()}"
            packet_buffer = PacketBuffer()
            self.packet_buffers[buffer_id] = packet_buffer
            self.tunnel_data_dashboard(client_socket, remote_socket, buffer_id, packet_buffer, client_addr)
        except Exception as exc:
            self.send_error_response(client_socket, 502)
            logger.error("HTTP CONNECT dashboard error: %s", exc)
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

    def tunnel_data_dashboard(self, client_socket, remote_socket, buffer_id, packet_buffer, client_addr):
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
                            self.data_manager.update_traffic(len(data), 0)
                        else:
                            client_socket.sendall(data)
                            packet_buffer.add_packet(len(data), "down")
                            with self.stats_lock:
                                self.stats["total_bytes_down"] += len(data)
                            self.data_manager.update_traffic(0, len(data))

                        self.try_infer(buffer_id, packet_buffer, client_addr)
                    except socket.error:
                        return
        finally:
            self.try_infer(buffer_id, packet_buffer, client_addr, force=True)
            try:
                client_socket.close()
            except Exception:
                pass
            try:
                remote_socket.close()
            except Exception:
                pass
            self.packet_buffers.pop(buffer_id, None)
            self.reported_buffers.discard(buffer_id)


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
            return {
                "class_name": LABELS[pred],
                "confidence": float(probs[pred]),
                "probabilities": {LABELS[i]: float(probs[i]) for i in range(len(probs))},
            }


def load_classifier(model_path):
    if not os.path.exists(model_path):
        return None, f"Model file not found: {model_path}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model_type = infer_model_type_from_state_dict(state_dict)
    model = create_model(
        model_type,
        input_length=int(RUNTIME_CONFIG["model"]["input_length"]),
        num_classes=int(RUNTIME_CONFIG["model"]["num_classes"]),
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return SimpleClassifier(model, device), None


def format_bytes(num_bytes):
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"


def render_status_cards(runtime, data_manager):
    traffic_detected = data_manager.has_recent_activity(ACTIVE_WINDOW_SECONDS)
    proxy_text = (
        f"connected {RUNTIME_CONFIG['proxy']['host']}:{runtime['proxy_port']}"
        if traffic_detected
        else f"Please set your network proxy to {RUNTIME_CONFIG['proxy']['host']}:{runtime['proxy_port']}"
    )

    cols = st.columns(3)

    with cols[0]:
        if runtime["load_error"]:
            st.error(f"Model · \n{runtime['load_error']}")
        else:
            st.success(f"Model · \n{os.path.basename(runtime['model_path'])}")

    with cols[1]:
        st.success(f"Service · listening on \n{RUNTIME_CONFIG['proxy']['host']}:{runtime['proxy_port']}") #

    with cols[2]:
        if traffic_detected:
            st.success(f"Proxy · \n{proxy_text}")
        else:
            st.warning(f"Proxy · \n{proxy_text}")


@st.cache_resource(show_spinner=False)
def get_dashboard_runtime(proxy_port, model_path):
    resolved_model_path = model_path or get_runtime_model_path()
    data_manager = DashboardDataManager()
    classifier, load_error = load_classifier(resolved_model_path)

    proxy_server = DashboardProxyServer(
        data_manager=data_manager,
        listen_host=RUNTIME_CONFIG["proxy"]["host"],
        listen_port=proxy_port,
        max_connections=int(RUNTIME_CONFIG["proxy"]["max_connections"]),
        min_packets_for_inference=int(RUNTIME_CONFIG["proxy"]["min_packets_for_inference"]),
    )
    if classifier:
        proxy_server.set_classifier(classifier)

    proxy_thread = threading.Thread(target=proxy_server.start, daemon=True)
    proxy_thread.start()

    logger.info("Dashboard runtime initialized on %s:%s", RUNTIME_CONFIG["proxy"]["host"], proxy_port)
    return {
        "data_manager": data_manager,
        "classifier": classifier,
        "load_error": load_error,
        "model_path": resolved_model_path,
        "proxy_port": proxy_port,
        "proxy_server": proxy_server,
    }


def stop_cached_runtime():
    runtime = st.session_state.get("dashboard_runtime")
    if runtime and runtime.get("proxy_server"):
        runtime["proxy_server"].stop()


atexit.register(stop_cached_runtime)


@st.fragment(run_every=REFRESH_SECONDS)
def render_live_dashboard(data_manager, proxy_port):
    runtime = st.session_state["dashboard_runtime"]
    summary = data_manager.get_summary()
    render_status_cards(runtime, data_manager)
    st.markdown("")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Connections", summary["total_connections"])
    col2.metric("Upload", format_bytes(summary["total_bytes_up"]))
    col3.metric("Download", format_bytes(summary["total_bytes_down"]))
    col4.metric("Predictions", summary["total_predictions"])

    st.markdown("---")
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Application Distribution")
        dist = data_manager.get_distribution_data()
        if dist:
            df = pd.DataFrame(
                [{"Application": app, "Count": item["count"], "Share": f"{item['percentage']:.1f}%"} for app, item in dist.items()]
            )
            st.dataframe(df, width="stretch")
            fig_pie = px.pie(
                df,
                values="Count",
                names="Application",
                title="Prediction Distribution",
                color="Application",
                color_discrete_map=COLOR_MAP,
            )
            st.plotly_chart(fig_pie, width="stretch")
        else:
            st.info("No traffic has been recorded yet. Generate traffic through the proxy and wait for auto-refresh.")

    with right:
        st.subheader("Recent Predictions")
        recent = data_manager.get_recent_predictions()
        if recent:
            df_recent = pd.DataFrame(recent)
            df_recent["timestamp"] = pd.to_datetime(df_recent["timestamp"]).dt.strftime("%H:%M:%S")
            st.dataframe(df_recent, width="stretch")
        else:
            st.info("No prediction records yet.")

    st.markdown("---")
    st.subheader("Recent Connections")
    connection_history_df = data_manager.get_connection_history_df()
    if not connection_history_df.empty:
        history_view = connection_history_df.sort_values("timestamp", ascending=False).copy()
        history_view["timestamp"] = history_view["timestamp"].dt.strftime("%H:%M:%S")
        st.dataframe(history_view.head(10), width="stretch")
    else:
        st.info("No connection records yet.")

    st.markdown("---")
    timeline = data_manager.get_timeline_data()
    if timeline:
        st.subheader("Prediction Timeline")
        df_timeline = pd.DataFrame(timeline)
        df_timeline["timestamp"] = pd.to_datetime(df_timeline["timestamp"])
        df_timeline = df_timeline.set_index("timestamp")
        app_counts = df_timeline.resample("5s")["app_type"].value_counts().unstack().fillna(0)

        fig_line = go.Figure()
        for app in APP_ORDER:
            if app in app_counts.columns:
                fig_line.add_trace(
                    go.Scatter(
                        x=app_counts.index,
                        y=app_counts[app],
                        mode="lines",
                        name=app,
                        stackgroup="one",
                    )
                )
        fig_line.update_layout(
            title="Predictions Over Time",
            xaxis_title="Time",
            yaxis_title="Prediction Count",
            hovermode="x unified",
        )
        st.plotly_chart(fig_line, width="stretch")

    st.caption(
        f"Proxy listening on {RUNTIME_CONFIG['proxy']['host']}:{proxy_port}. "
        f"The page auto-refreshes every {REFRESH_SECONDS} seconds."
    )


def render_dashboard(data_manager, proxy_port):
    st.set_page_config(page_title="Traffic Monitor Dashboard", page_icon="TD", layout="wide")
    st.title("Traffic Monitor Dashboard")
    st.markdown("Encrypted traffic proxy demo with real-time classification statistics.")
    render_live_dashboard(data_manager, proxy_port)


def run_dashboard(port=8501, proxy_port=8888, model_path=None):
    runtime = get_dashboard_runtime(proxy_port, model_path or get_runtime_model_path())
    st.session_state["dashboard_runtime"] = runtime
    render_dashboard(runtime["data_manager"], runtime["proxy_port"])


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Traffic monitoring dashboard")
    parser.add_argument("--dashboard-port", type=int, default=int(RUNTIME_CONFIG["dashboard"]["port"]))
    parser.add_argument("--proxy-port", type=int, default=int(RUNTIME_CONFIG["proxy"]["port"]))
    parser.add_argument("--model", type=str, default=get_runtime_model_path())
    args = parser.parse_args()

    run_dashboard(
        port=args.dashboard_port,
        proxy_port=args.proxy_port,
        model_path=args.model,
    )


if __name__ == "__main__":
    main()
