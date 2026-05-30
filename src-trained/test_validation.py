"""
测试验证方案 - 代理底层开发与流量特征提取模块

测试内容：
1. feature_extraction.py 单元测试
2. Proxy.py 单元测试
3. 集成测试（两个模块交互）
4. 性能测试
5. 异常处理测试

执行方式：
python test_validation.py
"""

import os
import sys
import time
import socket
import threading
import struct
import unittest
import tempfile
import logging
from io import BytesIO
from unittest.mock import Mock, patch, MagicMock

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestFlowExtractor(unittest.TestCase):
    """FlowExtractor单元测试"""

    def setUp(self):
        """测试前准备"""
        sys.path.insert(0, r'd:\解压\大二下\计网\Project')
        from feature_extraction import FlowExtractor
        self.extractor = FlowExtractor(max_packets=100)

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.extractor.max_packets, 100)
        self.assertEqual(len(self.extractor.flows), 0)

    def test_parse_pcap_header_valid(self):
        """测试解析有效PCAP文件头"""
        magic = struct.pack('<I', 0xa1b2c3d4)
        version = struct.pack('<HH', 2, 4)
        thiszone = struct.pack('<I', 0)
        sigfigs = struct.pack('<I', 0)
        snaplen = struct.pack('<I', 65535)
        network = struct.pack('<I', 1)
        header = magic + version + thiszone + sigfigs + snaplen + network

        f = BytesIO(header)
        result = self.extractor.parse_pcap_header(f)
        self.assertIsNotNone(result)

    def test_parse_pcap_header_invalid_magic(self):
        """测试解析无效MAGIC的PCAP文件头"""
        invalid_header = struct.pack('<I', 0x12345678) + b'\x00' * 20
        f = BytesIO(invalid_header)
        result = self.extractor.parse_pcap_header(f)
        self.assertIsNone(result)

    def test_parse_pcap_header_truncated(self):
        """测试解析截断的PCAP文件头"""
        f = BytesIO(b'\x00\x00\x00')
        result = self.extractor.parse_pcap_header(f)
        self.assertIsNone(result)

    def test_extract_flow_key(self):
        """测试五元组流键生成"""
        packet_info = {
            'src_ip': '192.168.1.100',
            'dst_ip': '192.168.1.1',
            'src_port': 54321,
            'dst_port': 80,
            'protocol': 6
        }
        flow_key = self.extractor.extract_flow_key(packet_info)

        result_src = flow_key[0]
        result_dst = flow_key[1]
        result_proto = flow_key[2]

        self.assertEqual(result_proto, 6)

        src_ip_str = packet_info['src_ip']
        dst_ip_str = packet_info['dst_ip']
        src_port_str = str(packet_info['src_port'])
        dst_port_str = str(packet_info['dst_port'])

        src_endpoint = f"{src_ip_str}:{src_port_str}"
        dst_endpoint = f"{dst_ip_str}:{dst_port_str}"

        expected_first = min(src_endpoint, dst_endpoint)
        expected_second = max(src_endpoint, dst_endpoint)

        self.assertEqual(result_src, expected_first)
        self.assertEqual(result_dst, expected_second)

    def test_extract_flow_key_reversed(self):
        """测试五元组流键生成（源和目标颠倒）"""
        packet_info1 = {
            'src_ip': '192.168.1.100',
            'dst_ip': '192.168.1.1',
            'src_port': 80,
            'dst_port': 54321,
            'protocol': 6
        }
        packet_info2 = {
            'src_ip': '192.168.1.1',
            'dst_ip': '192.168.1.100',
            'src_port': 54321,
            'dst_port': 80,
            'protocol': 6
        }

        key1 = self.extractor.extract_flow_key(packet_info1)
        key2 = self.extractor.extract_flow_key(packet_info2)

        self.assertEqual(key1, key2)

    def test_extract_from_nonexistent_file(self):
        """测试处理不存在的文件"""
        flows = self.extractor.extract_from_pcap('nonexistent.pcap')
        self.assertEqual(len(flows), 0)

    def test_extract_flow_features_padding(self):
        """测试特征提取 - 填充"""
        packet_lengths = [100, 200, 300]
        features = self.extractor.extract_flow_features(packet_lengths, max_packets=10)

        self.assertEqual(len(features), 10)
        max_val = max(packet_lengths)
        self.assertAlmostEqual(features[0], 100 / max_val, places=5)
        self.assertAlmostEqual(features[1], 200 / max_val, places=5)
        self.assertAlmostEqual(features[2], 300 / max_val, places=5)
        for i in range(3, 10):
            self.assertEqual(features[i], 0.0)

    def test_extract_flow_features_truncation(self):
        """测试特征提取 - 截断"""
        packet_lengths = list(range(150))
        features = self.extractor.extract_flow_features(packet_lengths, max_packets=50)

        self.assertEqual(len(features), 50)
        self.assertEqual(features[0], 0.0)
        self.assertEqual(features[49], 149/149)

    def test_extract_flow_features_empty(self):
        """测试特征提取 - 空列表"""
        features = self.extractor.extract_flow_features([], max_packets=10)

        self.assertEqual(len(features), 10)
        self.assertTrue(all(f == 0.0 for f in features))


class TestDataPreprocessor(unittest.TestCase):
    """DataPreprocessor单元测试"""

    def setUp(self):
        sys.path.insert(0, r'd:\解压\大二下\计网\Project')
        from feature_extraction import DataPreprocessor
        self.processor = DataPreprocessor(max_packets=100)

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.processor.max_packets, 100)
        self.assertEqual(len(self.processor.label_map), 4)

    def test_normalize_features_list(self):
        """测试归一化 - 列表输入"""
        features = [100, 200, 300, 400]
        normalized = self.processor.normalize_features(features)

        self.assertEqual(normalized[0], 0.25)
        self.assertEqual(normalized[1], 0.5)
        self.assertEqual(normalized[2], 0.75)
        self.assertEqual(normalized[3], 1.0)

    def test_normalize_features_array(self):
        """测试归一化 - 数组输入"""
        import numpy as np
        features = np.array([100, 200, 300, 400], dtype=np.float32)
        normalized = self.processor.normalize_features(features)

        self.assertAlmostEqual(normalized[0], 0.25, places=5)
        self.assertAlmostEqual(normalized[-1], 1.0, places=5)

    def test_normalize_features_zeros(self):
        """测试归一化 - 全零输入"""
        import numpy as np
        features = np.array([0, 0, 0], dtype=np.float32)
        normalized = self.processor.normalize_features(features)

        self.assertTrue(all(f == 0.0 for f in normalized))

    def test_pad_sequences_1d(self):
        """测试序列填充 - 一维"""
        import numpy as np
        features = np.array([1, 2, 3], dtype=np.float32)
        padded = self.processor.pad_sequences(features, max_length=10)

        self.assertEqual(len(padded), 10)
        self.assertTrue(np.array_equal(padded[:3], np.array([1, 2, 3])))
        self.assertTrue(np.array_equal(padded[3:], np.zeros(7)))

    def test_pad_sequences_2d(self):
        """测试序列填充 - 二维"""
        import numpy as np
        features = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        padded = self.processor.pad_sequences(features, max_length=5)

        self.assertEqual(padded.shape, (3, 5))
        self.assertTrue(np.array_equal(padded[0, :2], np.array([1, 2])))

    def test_preprocess_flow(self):
        """测试流预处理"""
        packet_lengths = [100] * 50 + [200] * 50
        features = self.processor.preprocess_flow(packet_lengths)

        self.assertEqual(len(features), 100)
        self.assertEqual(features[0], 0.5)
        self.assertEqual(features[49], 0.5)
        self.assertEqual(features[50], 1.0)
        self.assertEqual(features[99], 1.0)

    def test_split_train_test(self):
        """测试数据集划分"""
        import numpy as np
        X = np.random.rand(100, 50)
        y = np.random.randint(0, 4, 100)

        X_train, X_test, y_train, y_test = self.processor.split_train_test(X, y, test_ratio=0.2)

        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_test), 20)
        self.assertEqual(len(y_train), 80)
        self.assertEqual(len(y_test), 20)


class TestPacketBuffer(unittest.TestCase):
    """PacketBuffer单元测试"""

    def setUp(self):
        sys.path.insert(0, r'd:\解压\大二下\计网\Project')
        from Proxy import PacketBuffer
        self.buffer = PacketBuffer(max_packets=100)

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.buffer.max_packets, 100)
        self.assertEqual(len(self.buffer.packet_lengths), 0)
        self.assertIsNone(self.buffer.start_time)

    def test_add_packet(self):
        """测试添加数据包"""
        self.buffer.add_packet(100, 'up', timestamp=1000.0)
        self.buffer.add_packet(200, 'down', timestamp=1001.0)

        self.assertEqual(len(self.buffer.packet_lengths), 2)
        self.assertEqual(self.buffer.packet_lengths[0], 100)
        self.assertEqual(self.buffer.packet_lengths[1], 200)
        self.assertEqual(self.buffer.start_time, 1000.0)

    def test_add_packet_overflow(self):
        """测试添加数据包 - 溢出"""
        for i in range(150):
            self.buffer.add_packet(i + 1, 'up')

        self.assertEqual(len(self.buffer.packet_lengths), 100)
        self.assertEqual(self.buffer.packet_lengths[0], 51)
        self.assertEqual(self.buffer.packet_lengths[-1], 150)

    def test_is_ready(self):
        """测试就绪检查"""
        self.assertFalse(self.buffer.is_ready())

        for i in range(10):
            self.buffer.add_packet(100, 'up')

        self.assertTrue(self.buffer.is_ready())

    def test_get_features(self):
        """测试获取特征"""
        for i in range(50):
            self.buffer.add_packet((i + 1) * 10, 'up')

        features = self.buffer.get_features()

        self.assertEqual(len(features), 100)
        self.assertEqual(features[0], 10)
        self.assertEqual(features[49], 500)
        for i in range(50, 100):
            self.assertEqual(features[i], 0)

    def test_get_normalized_features(self):
        """测试获取归一化特征"""
        for i in range(5):
            self.buffer.add_packet((i + 1) * 100, 'up')

        features = self.buffer.get_normalized_features()

        self.assertAlmostEqual(features[0], 0.2, places=5)
        self.assertAlmostEqual(features[1], 0.4, places=5)
        self.assertAlmostEqual(features[4], 1.0, places=5)

    def test_get_normalized_features_empty(self):
        """测试获取归一化特征 - 空缓冲区"""
        features = self.buffer.get_normalized_features()

        self.assertTrue(all(f == 0.0 for f in features))


class TestTCPProxyServer(unittest.TestCase):
    """TCPProxyServer单元测试"""

    def setUp(self):
        sys.path.insert(0, r'd:\解压\大二下\计网\Project')
        from Proxy import TCPProxyServer
        self.server = TCPProxyServer(
            listen_host='127.0.0.1',
            listen_port=18888,
            max_connections=10
        )

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.server.listen_host, '127.0.0.1')
        self.assertEqual(self.server.listen_port, 18888)
        self.assertEqual(self.server.max_connections, 10)
        self.assertFalse(self.server.running)

    def test_is_http_connect_true(self):
        """测试HTTP CONNECT检测 - 真"""
        self.assertTrue(self.server.is_http_connect(b'CONNECT example.com:443 HTTP/1.1'))

    def test_is_http_connect_false(self):
        """测试HTTP CONNECT检测 - 假"""
        self.assertFalse(self.server.is_http_connect(b'GET / HTTP/1.1'))
        self.assertFalse(self.server.is_http_connect(b'POST / HTTP/1.1'))
        self.assertFalse(self.server.is_http_connect(b''))

    def test_parse_host_from_request(self):
        """测试从请求解析主机"""
        request = b'GET http://example.com/index.html HTTP/1.1\r\nHost: example.com\r\n\r\n'
        host, port = self.server.parse_host_from_request(request)

        self.assertEqual(host, 'example.com')
        self.assertEqual(port, 80)

    def test_parse_host_from_request_with_port(self):
        """测试从请求解析主机（带端口）"""
        request = b'GET /index.html HTTP/1.1\r\nHost: example.com:8080\r\n\r\n'
        host, port = self.server.parse_host_from_request(request)

        self.assertEqual(host, 'example.com')
        self.assertEqual(port, 8080)

    def test_parse_host_from_request_no_host(self):
        """测试从请求解析主机 - 无Host头"""
        request = b'GET /index.html HTTP/1.1\r\n\r\n'
        host, port = self.server.parse_host_from_request(request)

        self.assertIsNone(host)
        self.assertIsNone(port)

    def test_get_statistics(self):
        """测试获取统计信息"""
        stats = self.server.get_statistics()

        self.assertIn('total_connections', stats)
        self.assertIn('total_bytes_up', stats)
        self.assertIn('total_bytes_down', stats)
        self.assertIn('classifications', stats)

    def test_send_error_response(self):
        """测试发送错误响应"""
        mock_socket = MagicMock()
        self.server.send_error_response(mock_socket, 502)
        mock_socket.sendall.assert_called_once()


class TestIntegration(unittest.TestCase):
    """集成测试"""

    def test_flow_extractor_to_proxy_integration(self):
        """测试FlowExtractor到Proxy的集成"""
        sys.path.insert(0, r'd:\解压\大二下\计网\Project')
        from feature_extraction import DataPreprocessor
        from Proxy import PacketBuffer

        processor = DataPreprocessor(max_packets=100)

        packet_lengths = [100] * 50 + [200] * 50
        features = processor.preprocess_flow(packet_lengths)

        buffer = PacketBuffer(max_packets=100)
        for length in packet_lengths:
            buffer.add_packet(length, 'up')

        buffer_features = buffer.get_normalized_features()

        self.assertEqual(len(features), len(buffer_features))
        self.assertTrue(abs(features[-1] - buffer_features[-1]) < 0.01)

    def test_data_pipeline_consistency(self):
        """测试数据管道一致性"""
        sys.path.insert(0, r'd:\解压\大二下\计网\Project')
        from feature_extraction import FlowExtractor, DataPreprocessor

        extractor = FlowExtractor(max_packets=100)
        processor = DataPreprocessor(max_packets=100)

        packet_lengths = [150] * 30 + [500] * 30 + [1000] * 40

        features1 = extractor.extract_flow_features(packet_lengths)
        features2 = processor.preprocess_flow(packet_lengths)

        self.assertEqual(len(features1), len(features2))

        max_val = max(packet_lengths)
        expected_normalized = [min(pl / max_val, 1.0) for pl in packet_lengths[:100]]

        self.assertTrue(abs(features1[0] - features2[0]) < 0.01)


class TestPerformance(unittest.TestCase):
    """性能测试"""

    def test_flow_extraction_performance(self):
        """测试流提取性能"""
        sys.path.insert(0, r'd:\解压\大二下\计网\Project')
        from feature_extraction import FlowExtractor

        extractor = FlowExtractor(max_packets=100)

        start_time = time.time()
        for _ in range(1000):
            packet_lengths = list(range(100))
            features = extractor.extract_flow_features(packet_lengths)
        elapsed = time.time() - start_time

        logger.info(f"流提取1000次耗时: {elapsed:.4f}秒")
        self.assertLess(elapsed, 5.0, "流提取性能测试失败：耗时过长")

    def test_normalization_performance(self):
        """测试归一化性能"""
        sys.path.insert(0, r'd:\解压\大二下\计网\Project')
        from feature_extraction import DataPreprocessor
        import numpy as np

        processor = DataPreprocessor(max_packets=100)

        start_time = time.time()
        for _ in range(1000):
            features = np.random.rand(100) * 1000
            normalized = processor.normalize_features(features)
        elapsed = time.time() - start_time

        logger.info(f"归一化1000次耗时: {elapsed:.4f}秒")
        self.assertLess(elapsed, 2.0, "归一化性能测试失败：耗时过长")

    def test_packet_buffer_operations(self):
        """测试PacketBuffer操作性能"""
        sys.path.insert(0, r'd:\解压\大二下\计网\Project')
        from Proxy import PacketBuffer

        buffer = PacketBuffer(max_packets=100)

        start_time = time.time()
        for _ in range(10000):
            buffer.add_packet(100, 'up')
            if len(buffer.packet_lengths) > 100:
                buffer.packet_lengths = []
        elapsed = time.time() - start_time

        logger.info(f"PacketBuffer操作10000次耗时: {elapsed:.4f}秒")
        self.assertLess(elapsed, 3.0, "PacketBuffer性能测试失败：耗时过长")


class TestExceptionHandling(unittest.TestCase):
    """异常处理测试"""

    def test_flow_extractor_invalid_pcap(self):
        """测试处理无效PCAP文件"""
        sys.path.insert(0, r'd:\解压\大二下\计网\Project')
        from feature_extraction import FlowExtractor

        extractor = FlowExtractor()

        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pcap') as f:
            f.write(b'invalid pcap data')
            temp_path = f.name

        try:
            flows = extractor.extract_from_pcap(temp_path)
            self.assertEqual(len(flows), 0)
        finally:
            os.unlink(temp_path)

    def test_flow_extractor_permission_error(self):
        """测试文件权限错误"""
        sys.path.insert(0, r'd:\解压\大二下\计网\Project')
        from feature_extraction import FlowExtractor

        extractor = FlowExtractor()
        flows = extractor.extract_from_pcap('/root/protected.pcap')
        self.assertEqual(len(flows), 0)

    def test_proxy_server_address_in_use(self):
        """测试代理服务器地址占用"""
        sys.path.insert(0, r'd:\解压\大二下\计网\Project')
        from Proxy import TCPProxyServer

        server1 = TCPProxyServer(listen_port=18889)
        try:
            server1.server_socket = socket.socket()
            server1.server_socket.bind(('127.0.0.1', 18889))
            server1.server_socket.listen(1)

            server2 = TCPProxyServer(listen_port=18889)
            with self.assertRaises(Exception):
                server2.start()
        finally:
            server1.server_socket.close()

    def test_proxy_server_invalid_callback(self):
        """测试无效回调函数"""
        sys.path.insert(0, r'd:\解压\大二下\计网\Project')
        from Proxy import TCPProxyServer

        def bad_callback(*args):
            raise Exception("Callback error")

        server = TCPProxyServer(
            listen_port=18890,
            inference_callback=bad_callback
        )

        self.assertIsNotNone(server.inference_callback)


def run_tests():
    """运行所有测试"""
    print("=" * 70)
    print("代理底层开发与流量特征提取模块 - 验证测试")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestFlowExtractor))
    suite.addTests(loader.loadTestsFromTestCase(TestDataPreprocessor))
    suite.addTests(loader.loadTestsFromTestCase(TestPacketBuffer))
    suite.addTests(loader.loadTestsFromTestCase(TestTCPProxyServer))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestExceptionHandling))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    print(f"测试用例数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✅ 所有测试通过！模块功能正常。")
        return 0
    else:
        print("\n❌ 部分测试失败，请检查上述错误信息。")
        return 1


if __name__ == '__main__':
    sys.exit(run_tests())
