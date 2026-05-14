# 第 7 章　构建词表与数据集

> **本章目标**：动手读懂 `dataset.py`，理解从原始文本到训练样本 `(x, y)` 的完整流水线。

---

## 7.1 整体流水线

```
poems.txt
    │
    ▼
build_vocab()   ─→  poems（字列表）, id2word, word2id
    │
    ▼
encode_poems()  ─→  poems_id（ID 列表）
    │
    ▼
PoemDataset()   ─→  (x, y) 样本对
    │
    ▼
DataLoader      ─→  (N, L) 的批次张量
```

---

## 7.2 build_vocab：构建词表

```python
def build_vocab(file_path):
    poems = []
    char_set = set()

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = re.sub(r"[，。、？！：]", "", line).strip()
            if not line:
                continue
            char_set.update(list(line))   # 收集不重复的字
            poems.append(list(line))

    id2word = sorted(list(char_set)) + ["<UNK>"]
    word2id = {word: idx for idx, word in enumerate(id2word)}
    return poems, id2word, word2id
```

**关键细节**：

1. **去标点**：正则 `[，。、？！：]` 去掉所有中文标点，因为标点不属于诗的语义内容，生成时由程序控制插入位置
2. **排序**：`sorted(list(char_set))` 确保每次运行词表顺序一致，模型可复现
3. **`<UNK>` 标记**：追加在末尾，处理推理时出现的词表外字符

运行后得到：
```python
len(id2word)  # → 2439
word2id["春"]  # → 某个整数，如 156
id2word[156]  # → "春"
```

---

## 7.3 encode_poems：语料 ID 化

```python
def encode_poems(poems, word2id):
    unk_id = word2id["<UNK>"]
    return [
        [word2id.get(char, unk_id) for char in poem]
        for poem in poems
    ]
```

把每首诗的字符列表转换为整数 ID 列表：

```
['春', '眠', '不', '觉', '晓'] → [156, 891, 73, 441, 1203]
```

---

## 7.4 PoemDataset：滑动窗口切样本

语言模型的训练样本是 **(x, y)** 对，其中 **y 是 x 右移一位的版本**：

```
poem_id = [156, 891, 73, 441, 1203, ...]

seq_len = 4 时：
  i=0: x=[156,891,73,441]   y=[891,73,441,1203]
  i=1: x=[891,73,441,1203]  y=[73,441,1203,...]
  ...
```

这样，模型在每个位置 $t$ 的任务就是：给定 $x_t$（以及之前的隐藏状态），预测 $y_t = x_{t+1}$。

```python
class PoemDataset(Dataset):
    def __init__(self, poems_id, seq_len):
        self.samples = []
        for poem_id in poems_id:
            for i in range(len(poem_id) - seq_len):
                x = poem_id[i:     i + seq_len]
                y = poem_id[i + 1: i + 1 + seq_len]
                self.samples.append((x, y))

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.LongTensor(x), torch.LongTensor(y)
```

---

## 7.5 DataLoader 的作用

```python
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

DataLoader 做两件事：

1. **批量打包**：将多个 `(x, y)` 样本叠加成 `(N, L)` 的张量，利用 GPU 并行计算
2. **shuffle 打乱**：每轮训练前随机打乱样本顺序，避免模型记住 batch 的排列规律，有助于泛化

本项目所有样本长度相同（`seq_len=24`），无需 padding。

---

::: tip 小结
- `build_vocab`：去标点 → 收集字符集 → 排序构建双向映射
- `encode_poems`：字符 → 整数 ID
- `PoemDataset`：滑动窗口生成 `(x, y)` 对，y 是 x 右移一位
- `DataLoader`：批量打包 + shuffle，输出 `(N, L)` 张量
:::
