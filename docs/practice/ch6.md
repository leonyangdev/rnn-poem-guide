# 第 6 章　项目结构与数据准备

> **本章目标**：了解项目的模块化组织方式，读懂语料库的格式，并理解 `config.py` 中每个超参数的含义。

---

## 6.1 模块化的意义

原始代码将所有逻辑写在一个文件中，随着功能增加会变得难以维护。重构后，项目按**单一职责原则**拆分为 6 个模块：

```
poems/
├── data/
│   └── poems.txt        ← 语料库（313 首唐诗）
├── config.py            ← 所有超参数与路径（统一出口）
├── dataset.py           ← 数据预处理 + PoemDataset
├── model.py             ← PoemRNNLM 模型定义
├── trainer.py           ← 训练循环
├── generator.py         ← 诗歌生成（推理）
└── main.py              ← 主入口，串联所有模块
```

这种结构的好处：
- **修改超参数**：只改 `config.py`，不需要搜索散落在代码各处的魔法数字
- **替换模型**：只改 `model.py`，其余模块无需修改
- **复用生成逻辑**：`generator.py` 可以单独导入，在任意脚本中使用训练好的模型

---

## 6.2 语料库格式

`data/poems.txt` 的格式很简单：**每行一首诗，标点符号保留在行内**。

```
兰叶春葳蕤，桂华秋皎洁。欣欣此生意，自尔为佳节。
岱宗夫如何，齐鲁青未了。造化钟神秀，阴阳割昏晓。
花间一壶酒，独酌无相亲。举杯邀明月，对影成三人。
...（共 313 行）
```

数据统计：
- 诗的数量：313 首
- 不重复的字：**2439 个**
- 去标点后的训练样本数（`seq_len=24`）：**11,814 条**

---

## 6.3 config.py 逐行解读

```python
# 数据路径
DATA_PATH = './data/poems.txt'
SEQ_LEN   = 24          # 滑动窗口大小：每条训练样本包含 24 个字
```

`SEQ_LEN` 决定了模型在训练时能看到的最长上下文。值越大，模型能学到更长距离的依赖，但训练也越慢，内存占用越高。

```python
# 模型结构
EMBEDDING_SIZE = 256    # 词嵌入维度
HIDDEN_SIZE    = 512    # RNN 隐藏单元数
NUM_LAYERS     = 2      # RNN 堆叠层数
DROPOUT        = 0.2    # Dropout 概率
```

```python
# 训练超参数
BATCH_SIZE     = 32     # 每个 mini-batch 的样本数
LEARNING_RATE  = 1e-3   # Adam 学习率
EPOCH_NUM      = 20     # 训练总轮数
MAX_GRAD_NORM  = 5.0    # 梯度裁剪阈值
```

```python
# 生成参数
DEFAULT_LINE_NUM    = 4    # 生成行数（每行两句）
DEFAULT_LINE_LEN    = 7    # 每句字数（七言=7，五言=5）
DEFAULT_TEMPERATURE = 1.0  # 采样温度
```

---

## 6.4 设备检测：兼容 CUDA / MPS / CPU

本项目支持三种计算后端：

| 设备 | 适用场景 |
|------|----------|
| **CUDA** | NVIDIA GPU，训练最快 |
| **MPS** | Apple Silicon（M 系列芯片），MacBook 用户的 GPU 加速 |
| **CPU** | 无 GPU 时的回退，速度较慢但总能运行 |

```python
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

> MacBook Pro 用户通常会自动使用 MPS，训练速度比纯 CPU 快 3–5 倍。

---

::: tip 小结
- 项目按单一职责分为 6 个模块，config 统一管理所有超参数
- 语料库共 313 首唐诗，2439 个不重复字，生成 11,814 条训练样本
- 设备检测按优先级：CUDA → MPS → CPU
:::
