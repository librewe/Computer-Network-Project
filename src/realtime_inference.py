"""
Real-time inference helpers for the traffic proxy.
"""

import os
import threading
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from model import create_model, infer_model_type_from_state_dict
from Proxy import TCPProxyServer
from runtime_config import get_runtime_model_path, load_runtime_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrafficClassifier:
    def __init__(self, model_path=None, device=None):
        runtime_config = load_runtime_config()

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.app_labels = {int(k): v for k, v in runtime_config['labels'].items()}
        self.input_length = int(runtime_config['model']['input_length'])
        self.num_classes = int(runtime_config['model']['num_classes'])
        self.model = None
        self.model_type = None
        self.model_loaded = False
        self.load_error = None
        self.classification_history = {}
        self.lock = threading.Lock()

        self.load_model(model_path or get_runtime_model_path())
        logger.info(f"流量分类器初始化完成，使用设备: {self.device}")

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            self.load_error = f"Model file not found: {model_path}"
            logger.warning(self.load_error)
            return

        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

            self.model_type = infer_model_type_from_state_dict(state_dict)
            self.model = create_model(
                self.model_type,
                input_length=self.input_length,
                num_classes=self.num_classes
            )
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            self.load_error = None

            logger.info(f"成功加载模型: {model_path}")
            logger.info(f"模型类型: {self.model_type}")
            if isinstance(checkpoint, dict) and 'best_val_acc' in checkpoint:
                logger.info(f"模型验证准确率: {checkpoint.get('best_val_acc')}")
        except Exception as exc:
            self.load_error = str(exc)
            self.model_loaded = False
            logger.error(f"加载模型失败: {exc}")

    def classify(self, features):
        if self.model is None:
            raise RuntimeError(f"Model is not loaded: {self.load_error}")

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
                'class_name': self.app_labels.get(predicted_class, 'Unknown'),
                'confidence': confidence,
                'probabilities': {
                    self.app_labels[i]: float(probs[i]) for i in range(len(probs))
                }
            }

    def get_statistics(self):
        with self.lock:
            return dict(self.classification_history)


class InferenceProxyServer(TCPProxyServer):
    def __init__(self, listen_host='127.0.0.1', listen_port=8888,
                 max_connections=100, model_path=None):
        super().__init__(listen_host, listen_port, max_connections)
        self.classifier = TrafficClassifier(model_path)
        self.connection_classifications = {}
        self.classification_callback = None

    def set_classification_callback(self, callback):
        self.classification_callback = callback


def main():
    import argparse

    runtime_config = load_runtime_config()
    proxy_config = runtime_config['proxy']

    parser = argparse.ArgumentParser(description='实时流量分类代理服务器')
    parser.add_argument('--port', type=int, default=int(proxy_config['port']), help='监听端口')
    parser.add_argument('--model-path', type=str, default=get_runtime_model_path(), help='模型文件路径')
    args = parser.parse_args()

    server = InferenceProxyServer(
        listen_host=proxy_config['host'],
        listen_port=args.port,
        max_connections=int(proxy_config['max_connections']),
        model_path=args.model_path
    )

    def on_classification(client_addr, result, packet_count):
        logger.info(
            f"[分类结果] {client_addr}: {result['class_name']} "
            f"(置信度: {result['confidence']:.2f}, 数据包数: {packet_count})"
        )

    server.set_classification_callback(on_classification)

    try:
        server.start()
    except KeyboardInterrupt:
        server.stop()


if __name__ == '__main__':
    main()
