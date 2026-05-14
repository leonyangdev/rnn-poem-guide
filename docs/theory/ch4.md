# 第 4 章　训练原理

> **本章目标**：从损失函数到参数更新，完整描述语言模型的训练过程，重点理解梯度裁剪和 Dropout 的作用。

---

## 4.1 交叉熵损失

语言模型的训练目标是：**让模型给真实的下一个字分配尽可能高的概率**。

数学上等价于最小化**交叉熵损失（Cross-Entropy Loss）**：

$$
\mathcal{L} = -\frac{1}{N \cdot L} \sum_{n=1}^{N} \sum_{t=1}^{L} \log P(y_{n,t} \mid x_{n,1}, \ldots, x_{n,t})
$$

其中 $N$ 是 batch 大小，$L$ 是序列长度，$y_{n,t}$ 是第 $n$ 条样本第 $t$ 位的真实字 ID。

**直觉**：如果真实字是"晓"，但模型给"晓"的概率只有 0.01，那么 $-\log(0.01) \approx 4.6$，损失很大。反之，若概率为 0.9，损失只有 $-\log(0.9) \approx 0.1$。

在 PyTorch 中，`nn.CrossEntropyLoss` 已内置 softmax，不需要手动归一化：

```python
criterion = nn.CrossEntropyLoss()

# logits: (N, vocab_size, L)，target: (N, L)
loss = criterion(logits.transpose(1, 2), y)
```

> `transpose(1, 2)` 是因为 CrossEntropyLoss 要求类别维在第 1 轴，即形状 `(N, C, L)`。

---

## 4.2 随时间反向传播（BPTT）

RNN 展开后是一个普通的计算图，梯度通过**随时间反向传播（Backpropagation Through Time, BPTT）** 计算。

```
前向传播（计算损失）：
x₁ → h₁ → x₂ → h₂ → x₃ → h₃ → L

反向传播（计算梯度）：
∂L/∂h₃ → ∂L/∂h₂ → ∂L/∂h₁
         ↘          ↘          ↘
        ∂L/∂W_h   ∂L/∂W_h   ∂L/∂W_h  （梯度累加）
```

由于权重 $W_h$ 在所有时间步共享，它的梯度是所有时间步梯度贡献的**累加**。序列越长，梯度累加路径越多，越容易出现数值问题。

---

## 4.3 梯度裁剪

应对梯度爆炸最简单有效的方法是**梯度裁剪（Gradient Clipping）**：

$$
\text{if } \|\mathbf{g}\|_2 > \theta \text{, then } \mathbf{g} \leftarrow \frac{\theta}{\|\mathbf{g}\|_2} \cdot \mathbf{g}
$$

当所有参数梯度的 L2 范数超过阈值 $\theta$ 时，等比例缩小所有梯度，方向不变，只是"踩刹车"。

```python
# 在 loss.backward() 之后，optimizer.step() 之前调用
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

本项目阈值设为 `5.0`，这是 RNN 语言模型训练中的常见经验值。

---

## 4.4 Dropout 正则化

当训练集较小（本项目只有 313 首诗），模型容易**过拟合**——在训练集上 loss 极低，但在没见过的字组合上生成效果很差。

**Dropout** 的做法：训练时以概率 $p$ 随机将部分神经元输出置零，迫使网络不能依赖任何单一神经元，从而学习更鲁棒的表示。

```
训练时（model.train()）：
 h = [0.8, 0.0, 0.5, 0.0, 0.3]   ← 随机置零 2 个（p=0.4）

推理时（model.eval()）：
 h = [0.8, 0.6, 0.5, 0.4, 0.3]   ← 保持完整，不随机丢弃
```

本项目在两处使用 Dropout（`p=0.2`）：
1. **RNN 层间**：堆叠 RNN 的层与层之间
2. **输出层前**：RNN 输出经过 Dropout 再进入 Linear

---

## 4.5 Adam 优化器

优化器决定了如何用梯度更新参数。本项目使用 **Adam**，它在 SGD 基础上增加了两个关键机制：

| 机制 | 作用 |
|------|------|
| 一阶动量（Momentum） | 积累过去梯度方向，减少震荡 |
| 二阶动量（自适应学习率） | 对频繁更新的参数降低学习率，对稀少更新的参数提高学习率 |

对于 NLP 任务，Embedding 中大多数字的嵌入向量在每个 batch 中只有少数字被更新，Adam 的自适应学习率使这些稀疏更新更有效，因此 Adam 比 SGD 更适合语言模型训练。

```python
optimizer = optim.Adam(model.parameters(), lr=1e-3)
```

---

## 4.6 完整的单步训练流程

```python
# 1. 前向传播
logits, _ = model(x)                              # (N,L,V)

# 2. 计算损失
loss = criterion(logits.transpose(1, 2), y)       # (N,V,L) vs (N,L)

# 3. 反向传播
loss.backward()

# 4. 梯度裁剪（防止爆炸）
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

# 5. 参数更新
optimizer.step()

# 6. 梯度清零（为下一个 batch 准备）
optimizer.zero_grad()
```

> **为什么梯度清零放在最后？** PyTorch 默认梯度累加，若不清零，本次 batch 的梯度会叠加到下一次，导致错误更新。每个 batch 训练完成后必须清零。

---

::: tip 小结
- 交叉熵损失 = 最大化真实字的预测概率
- BPTT 将 RNN 的梯度沿时间步反向传播
- 梯度裁剪：超过阈值时等比例缩小梯度，防止爆炸
- Dropout：训练时随机置零神经元，缓解过拟合
- Adam：自适应学习率，适合 NLP 稀疏梯度场景
:::
