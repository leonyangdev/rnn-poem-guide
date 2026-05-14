# 第 9 章　训练流程

> **本章目标**：逐步拆解 `trainer.py`，理解一个完整训练 epoch 的每一行代码在做什么。

---

## 9.1 训练模式与推理模式

PyTorch 模型有两种状态，必须在合适的时机切换：

```python
model.train()   # 开启训练模式：Dropout 随机置零，BatchNorm 更新统计量
model.eval()    # 开启推理模式：Dropout 关闭，BatchNorm 使用固定统计量
```

训练循环开始前调用 `model.train()`，生成诗歌前调用 `model.eval()`，漏掉任何一个都会导致结果异常。

---

## 9.2 单个 batch 的完整训练步骤

```python
for i, (x, y) in enumerate(dataloader):
    # 步骤 1：数据移至目标设备
    x, y = x.to(device), y.to(device)      # (N, L)

    # 步骤 2：前向传播
    logits, _ = model(x)                    # (N, L, V)

    # 步骤 3：计算损失
    # CrossEntropyLoss 要求预测形状 (N, C, L)，目标形状 (N, L)
    loss = criterion(logits.transpose(1, 2), y)

    # 步骤 4：反向传播
    loss.backward()

    # 步骤 5：梯度裁剪（必须在 step 之前）
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

    # 步骤 6：参数更新
    optimizer.step()

    # 步骤 7：梯度清零（为下一个 batch 准备）
    optimizer.zero_grad()
```

---

## 9.3 为什么需要 transpose(1, 2)

这是实践中最容易出错的形状操作，值得单独讲清楚。

### CrossEntropyLoss 对形状的规定

`nn.CrossEntropyLoss` 处理序列分类时，要求：
- **预测**：形状 `(N, C, L)`，**C（类别数）必须在第 1 轴**
- **目标**：形状 `(N, L)`

### 我们的 logits 形状是什么

模型输出的 `logits` 形状是 `(N, L, V)`：

```
logits.shape = (32, 24, 2439)
               N   L   V
               │   │   └── 第 2 轴：对词表的打分（类别）
               │   └────── 第 1 轴：序列位置
               └────────── 第 0 轴：样本
```

问题在于：**V（类别维）在第 2 轴，但 CrossEntropyLoss 要求类别在第 1 轴。**

### transpose(1, 2) 做了什么

`.transpose(1, 2)` 把**第 1 轴和第 2 轴互换**：

```
logits         (N=32, L=24, V=2439)
    ↓ .transpose(1, 2)
               (N=32, V=2439, L=24)   ← V 换到了第 1 轴 ✓

y.shape      = (N=32, L=24)           ← 目标不需要变
```

```python
# 可以用 .shape 验证
print(logits.shape)               # torch.Size([32, 24, 2439])
print(logits.transpose(1,2).shape)# torch.Size([32, 2439, 24])
print(y.shape)                    # torch.Size([32, 24])

loss = criterion(logits.transpose(1, 2), y)   # ✓ 形状匹配
```

::: warning 常见错误
如果忘记 transpose 直接传 `logits`，PyTorch 会抛出：
```
RuntimeError: Expected target size (32, 2439), got torch.Size([32, 24])
```
这个报错信息很迷惑——它把 L=24 误当成类别维来检查，导致错误信息与真实原因对不上。遇到这个报错，先检查 transpose 是否漏写。
:::

---

## 9.4 损失统计：为什么不直接平均 batch loss

```python
batch_size = x.shape[0]
total_loss += loss.item() * batch_size   # 累加"总损失"
total_samples += batch_size

avg_loss = total_loss / total_samples    # epoch 结束后求平均
```

直接对所有 batch 的 loss 求平均有一个陷阱：最后一个 batch 可能样本数不足 32（例如只有 14 个），但它会和其他 batch 的 loss 等权重。用"总损失 / 总样本数"才是真正的样本级平均。

---

## 9.5 观察训练曲线

正常的训练曲线应该是**单调下降，逐渐平缓**的：

```
epoch  1: loss ≈ 5.0  （随机预测阶段，-log(1/2439) ≈ 7.8 为上界）
epoch  5: loss ≈ 3.5
epoch 10: loss ≈ 2.8
epoch 20: loss ≈ 2.2
```

**异常情况诊断**：

| 现象 | 可能原因 |
|------|----------|
| loss 不下降，维持在 7+ | 学习率过大，梯度更新在发散 |
| loss 下降很慢 | 学习率过小；或 batch_size 太大 |
| loss 震荡剧烈 | 学习率过大；或未做梯度裁剪 |
| train loss 极低但生成质量差 | 过拟合（训练集太小） |

---

::: tip 小结
- `model.train()` / `model.eval()` 必须在正确时机调用
- 训练七步：移设备 → 前向 → 损失 → 反向 → 裁剪 → 更新 → 清零
- `transpose(1, 2)` 是为了满足 CrossEntropyLoss 对输入形状的要求
- 损失统计要用"总损失/总样本数"，而非直接平均各 batch loss
:::
