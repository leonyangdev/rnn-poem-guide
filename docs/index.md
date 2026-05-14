---
layout: home

hero:
  name: "用 RNN 写古诗"
  text: "从原理到实践"
  tagline: 一本面向深度学习初学者的小册子，带你用 PyTorch 从零实现一个能写唐诗的循环神经网络。
  actions:
    - theme: brand
      text: 开始阅读 →
      link: /theory/ch1
    - theme: alt
      text: 查看代码
      link: https://github.com/leonyangdev/rnn-poem-guide

features:
  - icon: 🧮
    title: 理论篇（第 1–5 章）
    details: 从语言模型的概率基础出发，逐步讲清词嵌入、RNN 时序建模、BPTT 训练原理，以及温度采样等推理策略。
  - icon: 🛠️
    title: 实践篇（第 6–11 章）
    details: 与代码模块一一对应，逐行拆解 dataset / model / trainer / generator，最后给出一套完整的调参实验方案。
  - icon: 📜
    title: 配套语料
    details: 使用 313 首唐诗（2439 个不重复汉字）作为训练集，模型训练完成后即可逐字自回归地生成新诗。
---
