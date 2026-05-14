# 第 8 章　搭建模型

> **本章目标**：逐层拆解 `model.py` 中的 `PoemRNNLM`，理解每一层的输入输出形状和作用。

---

## 8.1 整体数据流

```
输入 x: (N, L)
    │
    ▼  nn.Embedding
嵌入向量: (N, L, E)          E = embedding_size = 256
    │
    ▼  nn.RNN (2层)
RNN 输出: (N, L, H)          H = hidden_size = 512
隐藏状态: (num_layers, N, H)
    │
    ▼  nn.Dropout(0.2)
    │
    ▼  nn.Linear
logits: (N, L, V)            V = vocab_size = 2439
```

---

## 8.2 完整代码

```python
class PoemRNNLM(nn.Module):
    def __init__(self, vocab_size, embedding_size=128, hidden_size=256,
                 num_layers=1, dropout=0.0):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_size
        )

        self.rnn = nn.RNN(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.dropout = nn.Dropout(dropout)

        self.linear = nn.Linear(
            in_features=hidden_size,
            out_features=vocab_size
        )

    def forward(self, x, hx=None):
        embedded = self.embedding(x)          # (N,L) → (N,L,E)
        output, hidden = self.rnn(embedded, hx)  # → (N,L,H), (layers,N,H)
        output = self.dropout(output)
        logits = self.linear(output)          # (N,L,H) → (N,L,V)
        return logits, hidden
```

---

## 8.3 各层详解

### Embedding 层

```python
self.embedding = nn.Embedding(num_embeddings=2439, embedding_dim=256)
```

- **参数量**：$2439 \times 256 = 624,384$（约 62 万）
- 这是模型中唯一一个"查表"操作，没有权重矩阵乘法，极高效
- 梯度只会回流到本 batch 中实际出现过的字对应的行

### RNN 层

```python
self.rnn = nn.RNN(input_size=256, hidden_size=512, num_layers=2,
                  batch_first=True, dropout=0.2)
```

- `batch_first=True`：张量形状为 `(N, L, H)` 而非默认的 `(L, N, H)`，更直观
- `dropout=0.2`：仅在层与层之间生效（num_layers=1 时无效），已在代码中做了判断
- **参数量**（单层）：$4 \times (E \times H + H \times H + 2H)$

  > RNN 有 4 组参数（实际上是 $W_x, W_h, b_x, b_h$），两层合计约 **270 万**参数

### Dropout 层

```python
self.dropout = nn.Dropout(0.2)
```

- 调用 `model.train()` 后生效，`model.eval()` 后自动跳过
- 放在 Linear 之前，对 RNN 输出进行正则化

### Linear 层（输出头）

```python
self.linear = nn.Linear(in_features=512, out_features=2439)
```

- **参数量**：$512 \times 2439 + 2439 = 1,252,968$（约 125 万）
- 输出 `logits`，形状 `(N, L, 2439)`，表示序列每个位置上对词表所有字的未归一化打分

---

## 8.4 forward 的两种调用场景

`hx` 参数区分了训练和推理两种用法：

```python
# 训练时：每个 batch 独立，hx=None，从全零隐藏状态开始
logits, _ = model(x)          # 丢弃 hidden，batch 之间无状态

# 推理时：连续生成，hx 持续传递
logits, hidden = model(x, hidden)   # 保留上下文记忆
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
- 数据流：`(N,L)` → Embedding → `(N,L,E)` → RNN → `(N,L,H)` → Dropout → Linear → `(N,L,V)`
- RNN 的 `batch_first=True` 让形状更直观
- `forward(x, hx)` 中 `hx=None` 用于训练，传入 `hidden` 用于推理
- 模型总参数约 279 万，轻量级，CPU 可训
:::
