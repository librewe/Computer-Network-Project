import os
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src-trained"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from runtime_config import get_runtime_model_path

TEST_MODEL_PATH = Path(get_runtime_model_path())


class TestProjectStructure(unittest.TestCase):
    def test_core_files_exist(self):
        required_files = [
            "Proxy.py",
            "realtime_inference.py",
            "dashboard.py",
            "model.py",
            "config.py",
        ]
        for filename in required_files:
            with self.subTest(filename=filename):
                self.assertTrue((SRC_DIR / filename).exists(), f"Missing file: {filename}")

    def test_model_weight_exists(self):
        self.assertTrue(TEST_MODEL_PATH.exists(), f"Missing test model: {TEST_MODEL_PATH}")


class TestProxyLayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from Proxy import PacketBuffer, TCPProxyServer

        cls.PacketBuffer = PacketBuffer
        cls.TCPProxyServer = TCPProxyServer

    def test_packet_buffer_padding_and_normalization(self):
        buffer = self.PacketBuffer(max_packets=8)
        for length in [100, 300, 500]:
            buffer.add_packet(length, "up")

        features = buffer.get_features()
        normalized = buffer.get_normalized_features()

        self.assertEqual(len(features), 8)
        self.assertEqual(features[:3], [100, 300, 500])
        self.assertEqual(features[3:], [0, 0, 0, 0, 0])
        self.assertAlmostEqual(normalized[0], 0.2, places=5)
        self.assertAlmostEqual(normalized[1], 0.6, places=5)
        self.assertAlmostEqual(normalized[2], 1.0, places=5)

    def test_http_request_host_parsing(self):
        server = self.TCPProxyServer(listen_host="127.0.0.1", listen_port=18888, max_connections=5)
        request = b"GET / HTTP/1.1\r\nHost: example.com:8080\r\n\r\n"
        host, port = server.parse_host_from_request(request)
        self.assertEqual(host, "example.com")
        self.assertEqual(port, 8080)


class TestRealtimeInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from realtime_inference import TrafficClassifier

        cls.TrafficClassifier = TrafficClassifier

    def test_classifier_outputs_expected_shape(self):
        classifier = self.TrafficClassifier(model_path=str(TEST_MODEL_PATH))
        self.assertTrue(classifier.model_loaded, msg=classifier.load_error)
        features = np.linspace(0.0, 1.0, 100, dtype=np.float32)

        result = classifier.classify(features)

        self.assertIn("class_name", result)
        self.assertIn("confidence", result)
        self.assertIn("probabilities", result)
        self.assertEqual(set(result["probabilities"].keys()), {"Video", "Chat", "FileTransfer", "Web"})
        self.assertTrue(0.0 <= result["confidence"] <= 1.0)


class TestDashboardDataFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dashboard import DashboardDataManager

        cls.DashboardDataManager = DashboardDataManager

    def test_data_manager_accumulates_metrics(self):
        manager = self.DashboardDataManager(max_history=10)

        manager.update_traffic(1024, 2048)
        manager.update_connection(("127.0.0.1", 50000), "Video", 0.91, 18)
        manager.update_prediction("Video", 0.91)

        summary = manager.get_summary()
        dist = manager.get_distribution_data()
        recent = manager.get_recent_predictions()
        history_df = manager.get_connection_history_df()

        self.assertEqual(summary["total_connections"], 1)
        self.assertEqual(summary["total_bytes_up"], 1024)
        self.assertEqual(summary["total_bytes_down"], 2048)
        self.assertEqual(summary["total_predictions"], 1)
        self.assertIn("Video", dist)
        self.assertEqual(dist["Video"]["count"], 1)
        self.assertEqual(len(recent), 1)
        self.assertEqual(len(history_df), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
