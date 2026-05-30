# Wireshark 使用详细指南

## 📖 目录

1. [Wireshark 安装](#1-wireshark-安装)
2. [界面介绍](#2-界面介绍)
3. [打开PCAP文件](#3-打开pcap文件)
4. [流量过滤](#4-流量过滤)
5. [查看数据包详情](#5-查看数据包详情)
6. [统计功能](#6-统计功能)
7. [导出数据](#7-导出数据)
8. [验证数据一致性](#8-验证数据一致性)

---

## 1. Wireshark 安装

### 步骤1：下载Wireshark
1. 打开浏览器访问：https://www.wireshark.org/download.html
2. 根据操作系统选择对应的安装包：
   - Windows: 下载 `Wireshark-win64-x.x.x.exe`
   - macOS: 下载 `Wireshark-x.x.x.dmg`
   - Linux: 使用包管理器安装

### 步骤2：安装Wireshark
1. 双击下载的安装文件
2. 按照安装向导完成安装
3. 安装过程中会提示安装WinPcap/Npcap，**务必勾选安装**，这是抓包所需的驱动

### 步骤3：启动Wireshark
1. 从开始菜单或桌面快捷方式启动Wireshark
2. 首次启动可能会提示选择网卡

---

## 2. 界面介绍

Wireshark界面分为5个主要区域：

```
┌─────────────────────────────────────────────────────────────┐
│  菜单栏 (File, Edit, View, Go, Capture, Analyze, ...)      │
├─────────────────────────────────────────────────────────────┤
│  工具栏 (常用快捷按钮)                                       │
├─────────────────────────────────────────────────────────────┤
│  过滤栏 (Filter: 输入过滤条件)                              │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  数据包列表 (Packet List)                                   │
│  ┌────┬───────────┬────────┬──────────┬──────────────────┐  │
│  │No. │ Time      │ Source │ Destination│ Protocol │ Info │  │
│  ├────┼───────────┼────────┼──────────┼──────────────────┤  │
│  │1   │ 0.000000  │ 192... │ 10.0.0.1  │ TCP      │ ...  │  │
│  │2   │ 0.000123  │ 10.0... │ 192.168.. │ HTTP     │ ...  │  │
│  └────┴───────────┴────────┴──────────┴──────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  数据包详情 (Packet Details)                                │
│  Frame 1: 60 bytes on wire (480 bits), ...                 │
│  ▶ Ethernet II, Src: ..., Dst: ...                         │
│  ▶ Internet Protocol Version 4, Src: ..., Dst: ...        │
│  ▶ Transmission Control Protocol, Src Port: ...            │
│  ▶ Hypertext Transfer Protocol                            │
├─────────────────────────────────────────────────────────────┤
│  十六进制视图 (Packet Bytes)                                │
│  0000  00 11 22 33 44 55 00 aa bb cc dd ee ff 08 00 ...  │
└─────────────────────────────────────────────────────────────┘
```

| 区域 | 功能 |
|------|------|
| 菜单栏 | 所有操作命令入口 |
| 工具栏 | 常用快捷操作（抓包开始/停止、保存等） |
| 过滤栏 | 输入过滤条件筛选数据包 |
| 数据包列表 | 显示所有捕获的数据包摘要 |
| 数据包详情 | 显示选中数据包的分层详细信息 |
| 十六进制视图 | 数据包的原始十六进制数据 |

---

## 3. 打开PCAP文件

### 步骤1：准备数据集
1. 下载ISCX VPN-nonVPN数据集并解压
2. 找到PCAP文件（如 `VPN_Video.pcap`, `nonVPN_Chat.pcap` 等）

### 步骤2：打开文件
1. 启动Wireshark
2. 点击菜单栏 **File** → **Open**
   - 或使用快捷键 `Ctrl+O`
3. 在文件选择对话框中导航到数据集目录
4. 选择要打开的PCAP文件（可多选）
5. 点击 **Open**

### 步骤3：查看文件信息
打开后会自动显示数据包列表，可以看到：
- 数据包编号
- 时间戳
- 源地址和目的地址
- 协议类型
- 简要信息

---

## 4. 流量过滤

### 4.1 基本过滤语法

| 过滤类型 | 语法示例 | 说明 |
|----------|----------|------|
| 按协议 | `tcp` | 只显示TCP数据包 |
| | `udp` | 只显示UDP数据包 |
| | `http` | 只显示HTTP数据包 |
| | `tls` | 只显示TLS/SSL数据包 |
| 按IP地址 | `ip.src == 192.168.1.100` | 源IP为指定地址 |
| | `ip.dst == 192.168.1.1` | 目的IP为指定地址 |
| | `ip.addr == 192.168.1.100` | 源或目的IP为指定地址 |
| 按端口 | `tcp.port == 80` | TCP端口为80 |
| | `udp.port == 53` | UDP端口为53 |
| | `tcp.srcport == 8080` | 源端口为8080 |
| 按包长度 | `frame.len > 1000` | 包长度大于1000字节 |
| | `frame.len < 50` | 包长度小于50字节 |
| 组合条件 | `tcp && ip.src == 192.168.1.100` | TCP且源IP匹配 |
| | `(tcp.port == 80) || (tcp.port == 443)` | 端口80或443 |

### 4.2 按应用类型过滤

#### 视频流量过滤
```
# YouTube视频流量
tcp && (ip.dst contains "google" || ip.src contains "google") && frame.len > 500

# 通用视频流量特征
(tcp && frame.len > 500 && frame.len < 1500) || udp && frame.len > 500
```

#### 聊天流量过滤
```
# 聊天软件通常使用小数据包
tcp && frame.len < 200 && (tcp.port == 443 || tcp.port == 80)

# Facebook聊天
ip.addr contains "facebook" && frame.len < 200
```

#### 文件传输流量过滤
```
# FTP流量
ftp || ftp-data

# 大文件传输特征
tcp && frame.len > 1000 && tcp.flags.push == 1
```

#### 网页浏览流量
```
# HTTP流量
http

# HTTPS流量
tls && tcp.port == 443
```

### 4.3 保存过滤条件
1. 在过滤栏输入过滤条件
2. 点击过滤栏右侧的 **Save** 按钮
3. 输入过滤器名称（如 "Video Traffic"）
4. 点击 **OK** 保存

---

## 5. 查看数据包详情

### 5.1 查看五元组信息

选中一个数据包，在数据包详情面板中展开：

```
Frame 1: 60 bytes on wire (480 bits), 60 bytes captured (480 bits)
├─ Ethernet II, Src: 00:11:22:33:44:55, Dst: aa:bb:cc:dd:ee:ff
├─ Internet Protocol Version 4, Src: 192.168.1.100, Dst: 203.0.113.50
│  ├─ Version: 4
│  ├─ Header Length: 20 bytes
│  ├─ Differentiated Services Field: 0x00
│  ├─ Total Length: 46
│  ├─ Identification: 0x1234
│  ├─ Flags: 0x02 (Don't Fragment)
│  ├─ Fragment Offset: 0
│  ├─ Time to Live: 64
│  ├─ Protocol: TCP (6)          ← 传输层协议
│  ├─ Header Checksum: 0xabcd [validation disabled]
│  ├─ Source Address: 192.168.1.100    ← 源IP地址
│  └─ Destination Address: 203.0.113.50 ← 目的IP地址
├─ Transmission Control Protocol, Src Port: 54321, Dst Port: 80
│  ├─ Source Port: 54321    ← 源端口
│  ├─ Destination Port: 80  ← 目的端口
│  ├─ Sequence Number: 1234567890
│  ├─ Acknowledgment Number: 0
│  ├─ Header Length: 20 bytes
│  └─ Flags: 0x02 (SYN)
└─ (No payload)
```

**五元组信息**：
- 源IP地址 (Source Address): `192.168.1.100`
- 目的IP地址 (Destination Address): `203.0.113.50`
- 源端口 (Source Port): `54321`
- 目的端口 (Destination Port): `80`
- 传输层协议 (Protocol): `TCP (6)`

### 5.2 查看包长度

包长度信息在Frame层：
```
Frame 1: 60 bytes on wire (480 bits), 60 bytes captured (480 bits)
```

### 5.3 查看TCP流

1. 右键点击数据包列表中的任意TCP数据包
2. 选择 **Follow** → **TCP Stream**
3. 会弹出一个新窗口，显示整个TCP流的内容
4. 可以选择显示格式（ASCII, EBCDIC, Hex Dump等）

---

## 6. 统计功能

### 6.1 基本统计

1. 点击菜单栏 **Statistics** → **Summary**
2. 会显示捕获文件的基本信息：
   - 数据包总数
   - 文件大小
   - 捕获时间范围
   - 平均包大小
   - 吞吐量等

### 6.2 协议分层统计

1. 点击菜单栏 **Statistics** → **Protocol Hierarchy**
2. 显示各层协议的数据包数量和百分比
3. 可以看到TCP、UDP、HTTP、TLS等协议的分布

### 6.3 会话统计

1. 点击菜单栏 **Statistics** → **Conversations**
2. 选择 **TCP** 标签页
3. 显示所有TCP会话的统计信息：
   - 源地址和端口
   - 目的地址和端口
   - 数据包数量
   - 字节数
   - 持续时间

### 6.4 流量图

1. 点击菜单栏 **Statistics** → **IO Graph**
2. 可以看到流量随时间的变化趋势
3. 可以设置时间间隔和显示单位

---

## 7. 导出数据

### 7.1 导出筛选后的数据包

1. 输入过滤条件筛选数据包
2. 选中要导出的数据包（可多选）
3. 右键点击 → **Export Specified Packets**
4. 在对话框中：
   - 选择导出范围（选中的数据包或所有显示的数据包）
   - 选择输出格式（PCAP, PCAP-NG等）
   - 指定输出文件路径
5. 点击 **Save**

### 7.2 导出为CSV

1. 选择要导出的数据包
2. 点击菜单栏 **File** → **Export Packet Dissections** → **As CSV...**
3. 设置输出选项：
   - 选择导出的字段
   - 设置分隔符
4. 指定输出文件路径
5. 点击 **Save**

### 7.3 导出TCP流

1. 右键点击TCP数据包 → **Follow** → **TCP Stream**
2. 在弹出的窗口中点击 **Save As...**
3. 选择保存路径和文件名
4. 点击 **Save**

---

## 8. 验证数据一致性

### 8.1 验证脚本处理结果

**步骤1：提取特征**
1. 使用Wireshark打开PCAP文件
2. 应用过滤条件筛选特定流
3. 记录该流的前N个数据包长度

**步骤2：运行预处理脚本**
```bash
python data_preprocessing.py
```

**步骤3：对比结果**
1. 将Wireshark中记录的包长度与脚本输出的特征进行对比
2. 验证归一化后的值是否正确
3. 检查填充和截断是否符合预期

### 8.2 验证模型输入

```python
# 示例：验证单个流的特征提取
from data_preprocessing import DataPreprocessor

config = {
    'MAX_PACKETS': 100,
    'MIN_PACKETS': 10,
    'NORMALIZATION_METHOD': 'minmax',
    'CLASS_MAP': {'Video': 0, 'Chat': 1, 'FileTransfer': 2, 'Web': 3},
    'REVERSE_CLASS_MAP': {0: 'Video', 1: 'Chat', 2: 'FileTransfer', 3: 'Web'}
}

preprocessor = DataPreprocessor(config)

# 手动输入从Wireshark中记录的包长度
packet_lengths = [60, 1500, 1480, 60, 1500, ...]  # 从Wireshark获取

features = preprocessor.preprocess_flow(packet_lengths)
print(f"特征长度: {len(features)}")
print(f"特征范围: [{features.min():.4f}, {features.max():.4f}]")
print(f"特征值: {features[:10]}...")
```

### 8.3 常见问题排查

| 问题 | 可能原因 | 解决方法 |
|------|---------|---------|
| 包长度不匹配 | 脚本和Wireshark的统计方式不同 | 确认两者都使用相同的长度定义 |
| 流切分不一致 | 五元组定义不同 | 确保脚本和Wireshark使用相同的五元组定义 |
| 归一化结果差异 | 归一化方法不同 | 确认使用相同的归一化方法（Min-Max或Z-score） |

---

## 📝 常用快捷键

| 快捷键 | 功能 |
|--------|------|
| `Ctrl+O` | 打开文件 |
| `Ctrl+S` | 保存文件 |
| `Ctrl+F` | 查找数据包 |
| `Ctrl+E` | 开始/停止抓包 |
| `Ctrl+D` | 清空捕获 |
| `Ctrl+L` | 清空过滤栏 |
| `Enter` | 选中数据包 |
| `Esc` | 取消选择 |

---

## 📌 实用技巧

1. **使用颜色规则**：点击菜单栏 **View** → **Coloring Rules**，可以为不同协议设置不同颜色
2. **使用专家信息**：点击数据包详情面板中的 **Expert Info** 标签，可以查看警告和错误信息
3. **使用注释**：右键点击数据包 → **Add Comment**，可以为数据包添加注释
4. **使用书签**：右键点击数据包 → **Add Bookmark**，可以标记重要的数据包

---

如有任何问题，请参考Wireshark官方文档：https://www.wireshark.org/docs/
