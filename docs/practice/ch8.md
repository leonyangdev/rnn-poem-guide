# 第 8 章　搭建模型

> **本章目标**：逐层拆解 `model.py` 中的 `PoemRNNLM`，彻底搞清楚每一层的输入输出形状和作用。

---

## 8.1 维度字典：先把符号搞清楚

本项目中出现的所有维度符号统一如下，遇到任何形状标注，先对号入座：

| 符号 | 全称 | 本项目的值 | 含义 |
|------|------|-----------|------|
| **N** | batch size | 32 | 一次送入模型的样本（诗句片段）数量 |
| **L** | sequence length | 24 | 每条样本包含的字数（序列长度） |
| **E** | embedding size | 256 | 每个字被表示为多少维的向量 |
| **H** | hidden size | 512 | RNN 隐藏状态向量的维度 |
| **V** | vocab size | 2439 | 词表中字的总数（模型最终要在这么多字里选一个） |

---

## 8.2 整体数据流（带形状注释）

```
输入 x:              (N=32, L=24)           ← 32 条样本，每条 24 个字 ID（整数）
    │
    ▼  nn.Embedding
嵌入向量:            (N=32, L=24, E=256)    ← 每个整数 ID 换成 256 维浮点向量
    │
    ▼  nn.RNN (2层)
RNN output:          (N=32, L=24, H=512)   ← 每个时间步，每条样本，512 维隐藏向量
RNN hidden:          (2,   N=32, H=512)    ← 序列末尾的隐藏状态，第 0 轴按层叠放
    │
    ▼  nn.Dropout(0.2)
    │
    ▼  nn.Linear
logits:              (N=32, L=24, V=2439)  ← 每个位置对词表所有字的未归一化打分
```

---

## 8.3 各层详解与维度分析

### Embedding 层

```python
self.embedding = nn.Embedding(num_embeddings=2439, embedding_dim=256)
```

**形状变化**：`(N, L)` 整数 → `(N, L, E)` 浮点

整个操作就是"按 ID 取行"。想象 Embedding 是一张 2439 行 × 256 列的表，输入的每个整数指定取哪一行，所以第三个轴 E 就是从表里取出的那一行向量。

```
x[0] = [156, 891, 73, ...]   # 第 0 条样本的 24 个字 ID
         ↓    ↓    ↓
embedded[0] = [[第156行], [第891行], [第73行], ...]
              shape: (L=24, E=256)
```

- **参数量**：$2439 \times 256 = 624,384$（约 62 万）
- 梯度只流向本 batch 中实际出现过的字，其余行本轮不更新

### RNN 层

```python
self.rnn = nn.RNN(input_size=256, hidden_size=512, num_layers=2,
                  batch_first=True, dropout=0.2)
```

**形状变化**：`(N, L, E)` → output `(N, L, H)` + hidden `(num_layers, N, H)`

这里有两个返回值，是最容易混淆的地方：

| 返回值 | 形状 | 包含什么 |
|--------|------|---------|
| `output` | `(N, L, H)` | **L 个时间步**的最后一层隐藏状态，用于接 Linear 做预测 |
| `hidden` | `(num_layers, N, H)` | **最后一个时间步**所有层的隐藏状态，用于传递记忆 |

以 N=2、L=4、num_layers=2 为例，`hidden` 的结构是：

```
hidden[0] = 第 1 层 RNN 在 t=4 时刻的隐藏状态  # shape (N=2, H=512)
hidden[1] = 第 2 层 RNN 在 t=4 时刻的隐藏状态  # shape (N=2, H=512)
```

`batch_first=True` 的作用：告诉 PyTorch "N 在前，L 在中"。如果不设置，默认是 `(L, N, H)`，很多人会在这里搞混批次维和序列维。

### Dropout 层

```python
self.dropout = nn.Dropout(0.2)
```

形状不变，只是随机将 20% 的位置置零。`model.eval()` 后自动关闭。

### Linear 层（输出头）

```python
self.linear = nn.Linear(in_features=512, out_features=2439)
```

**形状变化**：`(N, L, H)` → `(N, L, V)`

Linear 对最后一个轴做矩阵乘法：每个位置的 512 维向量 × 权重矩阵 $(512 \times 2439)$ = 2439 维 logits。前两个轴 N 和 L 不参与计算，只是"搭便车"。

- **参数量**：$512 \times 2439 + 2439 \approx 125$ 万
- 输出 logits 不经过 softmax，让 CrossEntropyLoss 自己处理

---

## 8.4 forward 的两种调用场景

`hx` 参数区分了训练和推理两种用法：

```python
# 训练时：每个 batch 独立，hx=None，从全零隐藏状态开始
logits, _ = model(x)                 # 丢弃 hidden，batch 之间无状态

# 推理时：逐字连续生成，hx 持续传递
logits, hidden = model(x, hidden)    # 保留跨步的上下文记忆
```

---

## 8.5 模型参数量汇总

| 层 | 参数量 |
|----|--------|
| Embedding | ~62 万 |
| RNN（2 层） | ~270 万 |
| Linear | ~125 万 |
| **合计** | **~279 万** |

这是一个相当小的模型，在 CPU 上也能在几分钟内完成训练。

---

::: tip 小结
- N=样本数，L=序列长度，E=嵌入维，H=隐藏维，V=词表大小
- Embedding：`(N,L)` 整数 → `(N,L,E)` 向量（查表）
- RNN 返回两个值：output `(N,L,H)` 含所有时间步；hidden `(layers,N,H)` 只含最后时间步
- Linear 只对最后一轴做乘法，N 和 L 不变
- `batch_first=True` 把 N 放在第 0 轴，与直觉一致
:::
