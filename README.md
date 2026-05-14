# 用 RNN 写古诗

> 一本面向深度学习初学者的小册子，带你用 PyTorch 从零实现一个能写唐诗的循环神经网络。

**在线阅读**：[https://leonyangdev.github.io/rnn-poem-guide/](https://leonyangdev.github.io/rnn-poem-guide/)

## 内容简介

本小册子分为**理论篇**与**实践篇**两个部分：

| 章节 | 内容 |
|------|------|
| 第 1 章 | 语言模型是什么：链式法则、N-gram vs 神经 LM |
| 第 2 章 | 词嵌入：把文字变成向量，N/L/E 维度详解 |
| 第 3 章 | 循环神经网络：隐藏状态、output vs hidden 维度 |
| 第 4 章 | 训练原理：交叉熵、BPTT、梯度裁剪、Dropout |
| 第 5 章 | 文本生成：自回归、温度采样、hidden state 传递 |
| 第 6 章 | 项目结构与数据准备 |
| 第 7 章 | 构建词表与数据集 |
| 第 8 章 | 搭建模型（含维度字典） |
| 第 9 章 | 训练流程（含 transpose 详解） |
| 第 10 章 | 生成古诗 |
| 第 11 章 | 动手实验：调参与观察 |

## 配套代码

代码位于 [`poems/`](../.) 目录：

```
poems/
├── config.py       # 超参数配置
├── dataset.py      # 数据预处理 + PoemDataset
├── model.py        # PoemRNNLM 模型
├── trainer.py      # 训练循环
├── generator.py    # 诗歌生成
├── main.py         # 主入口
└── data/
    └── poems.txt   # 313 首唐诗语料
```

运行方式：

```bash
cd poems/
python main.py
```

支持设备：CUDA（NVIDIA GPU）/ MPS（Apple Silicon）/ CPU，自动检测。

## 本地预览文档

```bash
cd rnn-poem-guide/
npm install
npm run docs:dev
```

## License

[MIT](./LICENSE)
