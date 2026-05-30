# 基于深度学习的加密流量实时识别与监控系统

## 📁 项目文件结构

```
d:\解压\大二下\计网\Project\
├── config.py                    # 配置文件（数据集路径、模型参数等）
├── Proxy.py                     # TCP/HTTP代理服务器主程序
├── feature_extraction.py        # 流量特征提取模块
├── data_preprocessing.py        # 数据集预处理脚本（支持真实PCAP）
├── train_model.py               # 1D-CNN模型训练脚本
├── model.py                     # 模型定义（CNN/LSTM/MLP）
├── realtime_inference.py        # 在线实时推理模块
├── dashboard.py                 # Streamlit实时监控看板
├── visualization.py             # 模型可视化与对比实验
├── jitter_test.py               # 对抗网络抖动测试
├── requirements.txt             # 依赖包列表
├── WIRESHARK_GUIDE.md           # Wireshark详细使用指南
├── README.md                    # 项目说明文档
├── saved_models/                # 保存的训练模型
└── data/                        # 数据目录（PCAP文件和处理后数据）
```

---

## 🎯 功能概览

### 基础功能 (必做)
1. ✅ **TCP/HTTP代理服务器** - 转发网络流量
2. ✅ **流量特征提取** - 按五元组切分流，提取包长度特征
3. ✅ **数据预处理** - 清洗、归一化（Min-Max/Z-score）
4. ✅ **离线模型训练** - 1D-CNN模型训练与评估

### 扩展功能
- ✅ **在线实时推理** (25分) - 实时分类流量
- ✅ **实时流量监控看板** (25分) - Web界面展示
- ✅ **模型可视化与对比** (10分) - 训练曲线、混淆矩阵
- ✅ **对抗网络抖动测试** (10分) - 抗干扰能力评估

---

## 🚀 详细操作指南

### 第一步：安装依赖

1. 打开**命令提示符** (Win+R，输入 cmd)

2. 进入项目目录：
```bash
cd d:\解压\大二下\计网\Project
```

3. 安装Python依赖包：
```bash
pip install -r requirements.txt
```

如果安装速度慢，使用国内镜像：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 第二步：下载数据集（可选但推荐）

#### 方式A：使用真实数据集（推荐）

1. **下载ISCX VPN-nonVPN数据集**：
   - 访问：https://www.unb.ca/cic/datasets/vpn.html
   - 下载 `VPN-nonVPN.zip` 和 `VPN-nonVPN_TrafficLabels.csv`

2. **解压到指定目录**：
```
data/
└── ISCX-VPN/
      ├── VPN_Video.pcap
      ├── VPN_FileTransfer.pcap
      ├── nonVPN_Chat.pcap
      ├── nonVPN_FileTransfer.pcap
      ├── nonVPN_Browsing.pcap
      └── nonVPN_Video.pcap
```

3. **配置数据集路径**：
   - 编辑 `config.py` 文件
   - 修改 `DATASET_PATH` 和 `PCAP_FILES` 配置

#### 方式B：使用合成数据（自动生成）

如果不下载真实数据集，代码会自动生成合成数据用于训练。

### 第三步：数据预处理

```bash
python data_preprocessing.py
```

功能说明：
- 按五元组（源IP、目的IP、源端口、目的端口、协议）切分流
- 提取前N个数据包长度特征（N可配置，默认100）
- 清洗异常值（小于0或大于1500的长度）
- 归一化处理（Min-Max或Z-score）
- 划分数据集（训练集70%、验证集15%、测试集15%）

### 第四步：训练模型

```bash
python train_model.py --epochs 50 --batch-size 64 --lr 0.001
```

参数说明：
- `--epochs`: 训练轮数（默认50）
- `--batch-size`: 批量大小（默认64）
- `--lr`: 学习率（默认0.001）
- `--max-packets`: 每个流提取的最大数据包数（默认100）

训练完成后生成：
- `saved_models/best_model.pth` - 最佳模型权重
- `saved_models/final_model.pth` - 最终模型权重
- `saved_models/training_curves.png` - 损失/准确率曲线
- `saved_models/confusion_matrix.png` - 混淆矩阵热力图
- `saved_models/evaluation_report.txt` - 评估报告（准确率、精确率、召回率、F1）

### 第五步：测试代理服务器

#### 方式A：使用基础代理服务器
```bash
python Proxy.py --port 8888
```

#### 方式B：使用带实时推理的代理服务器
```bash
python realtime_inference.py --port 8888
```

#### 配置浏览器代理

**Chrome浏览器**：
1. 设置 → 系统 → 代理设置 → 局域网设置
2. 勾选"为LAN使用代理服务器"
3. 地址：`127.0.0.1`，端口：`8888`

**Firefox浏览器**：
1. 设置 → 常规 → 网络设置 → 设置
2. 选择"手动代理配置"
3. HTTP代理：`127.0.0.1`，端口：`8888`

### 第六步：启动实时监控看板

**需要同时运行两个终端**：

#### 终端1：启动代理服务器
```bash
python realtime_inference.py --port 8888
```

#### 终端2：启动Streamlit看板
```bash
streamlit run dashboard.py --server.port 8501
```

然后打开浏览器访问：http://localhost:8501

---

## 📊 模型架构

### 1D-CNN模型结构

```
输入层 (100维)
    ↓
Conv1D(64, kernel_size=3) → ReLU → MaxPool1D(2)
    ↓
Conv1D(128, kernel_size=3) → ReLU → MaxPool1D(2)
    ↓
Conv1D(256, kernel_size=3) → ReLU → MaxPool1D(2)
    ↓
Flatten
    ↓
Dense(256) → ReLU → Dropout(0.5)
    ↓
Dense(128) → ReLU → Dropout(0.5)
    ↓
输出层 (4类) → Softmax
```

### 类别映射

| 类别ID | 应用类型 | 特点 |
|--------|---------|------|
| 0 | Video（视频） | 大数据包、持续传输 |
| 1 | Chat（聊天） | 小数据包、间歇性传输 |
| 2 | FileTransfer（文件传输） | 中大型数据包 |
| 3 | Web（网页浏览） | 混合大小包 |

---

## ⚙️ 配置文件说明

`config.py` 文件包含以下配置项：

```python
# 数据集路径
DATASET_PATH = 'data/ISCX-VPN/'
PCAP_FILES = [...]

# 特征提取参数
MAX_PACKETS = 100      # 每个流提取的最大数据包数
MIN_PACKETS = 10       # 有效流的最小数据包数

# 数据预处理
NORMALIZATION_METHOD = 'minmax'  # 'minmax' 或 'zscore'
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 模型参数
INPUT_LENGTH = 100
NUM_CLASSES = 4
DROPOUT_RATE = 0.5

# 训练参数
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
```

---

## 📈 评估指标说明

训练完成后会输出以下指标：

| 指标 | 说明 |
|------|------|
| **Accuracy（准确率）** | 正确分类的样本占总样本的比例 |
| **Precision（精确率）** | 预测为某类的样本中真正属于该类的比例 |
| **Recall（召回率）** | 真正属于某类的样本被正确预测的比例 |
| **F1 Score** | 精确率和召回率的调和平均数 |

---

## 🔍 Wireshark使用指南

详细的Wireshark使用说明请参考 `WIRESHARK_GUIDE.md` 文件，包括：
- 安装和界面介绍
- 打开PCAP文件
- 流量过滤方法
- 查看数据包详情（五元组、包长度）
- 统计功能使用
- 数据导出
- 验证脚本处理结果的一致性

---

## ⚠️ 常见问题

### 1. 安装依赖失败
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. 模型文件不存在
- 运行 `python train_model.py` 训练模型
- 或者代码会自动使用随机权重（准确率较低）

### 3. 代理服务器无法启动
- 检查端口是否被占用：`netstat -an | findstr 8888`
- 尝试更换端口：`python Proxy.py --port 8889`

### 4. 浏览器无法通过代理上网
- 确认代理配置正确
- 确认代理服务器正在运行
- 检查防火墙设置

---

## 📝 团队分工建议

| 任务 | 人数 | 负责文件 |
|------|------|---------|
| 代理底层开发 | 1-2人 | Proxy.py, realtime_inference.py |
| 数据预处理 | 1人 | data_preprocessing.py |
| 模型训练 | 1人 | train_model.py, model.py |
| 可视化与前端 | 1人 | dashboard.py, visualization.py |
| 测试与文档 | 1人 | jitter_test.py, 实验报告 |

---

## 📞 技术支持

如有问题，请检查：
1. Python版本（建议3.8+）
2. 依赖包是否完整安装
3. 端口是否被占用
4. 浏览器代理是否正确配置

---

**项目评分：基础分 + 附加分 = 100 + 70分** ✅
