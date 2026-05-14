import { defineConfig } from 'vitepress'

export default defineConfig({
  title: '用 RNN 写古诗',
  description: '从语言模型原理到 PyTorch 实践的深度学习小册子',
  base: '/rnn-poem-guide/',
  lang: 'zh-CN',

  head: [
    ['meta', { name: 'theme-color', content: '#646cff' }],
  ],

  themeConfig: {
    logo: '📜',
    siteTitle: '用 RNN 写古诗',

    // ── 顶部导航 ──────────────────────────────────────────────────────────
    nav: [
      { text: '首页', link: '/' },
      { text: '理论篇', link: '/theory/ch1', activeMatch: '/theory/' },
      { text: '实践篇', link: '/practice/ch6', activeMatch: '/practice/' },
    ],

    // ── 侧边栏 ────────────────────────────────────────────────────────────
    sidebar: {
      '/theory/': [
        {
          text: '理论篇',
          items: [
            { text: '第 1 章　语言模型是什么', link: '/theory/ch1' },
            { text: '第 2 章　词嵌入：把文字变成向量', link: '/theory/ch2' },
            { text: '第 3 章　循环神经网络（RNN）', link: '/theory/ch3' },
            { text: '第 4 章　训练原理', link: '/theory/ch4' },
            { text: '第 5 章　文本生成与推理策略', link: '/theory/ch5' },
          ],
        },
      ],
      '/practice/': [
        {
          text: '实践篇',
          items: [
            { text: '第 6 章　项目结构与数据准备', link: '/practice/ch6' },
            { text: '第 7 章　构建词表与数据集', link: '/practice/ch7' },
            { text: '第 8 章　搭建模型', link: '/practice/ch8' },
            { text: '第 9 章　训练流程', link: '/practice/ch9' },
            { text: '第 10 章　生成古诗', link: '/practice/ch10' },
            { text: '第 11 章　动手实验：调参与观察', link: '/practice/ch11' },
          ],
        },
      ],
    },

    // ── 页脚与社交链接 ─────────────────────────────────────────────────────
    socialLinks: [
      { icon: 'github', link: 'https://github.com/leonyangdev/rnn-poem-guide' },
    ],

    footer: {
      message: '基于 MIT 协议发布',
      copyright: 'Copyright © 2025 Leon Yang',
    },

    // ── 中文本地化 ────────────────────────────────────────────────────────
    docFooter: {
      prev: '上一章',
      next: '下一章',
    },
    outline: {
      label: '本章目录',
      level: [2, 3],
    },
    lastUpdated: {
      text: '最后更新于',
    },
    returnToTopLabel: '回到顶部',
    sidebarMenuLabel: '目录',
    darkModeSwitchLabel: '外观',
  },

  markdown: {
    math: true,
  },
})
