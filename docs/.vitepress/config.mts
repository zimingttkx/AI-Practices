import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'AI-Practices',
  description: 'A Systematic Approach to AI Research & Engineering',

  // GitHub Pages 子路径配置
  base: '/AI-Practices/',

  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/AI-Practices/logo.svg' }],
    ['meta', { name: 'theme-color', content: '#007AFF' }],
    ['meta', { name: 'author', content: 'zimingttkx' }],
    ['meta', { name: 'keywords', content: 'AI, Machine Learning, Deep Learning, Neural Networks, Computer Vision, NLP, PyTorch, TensorFlow' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:title', content: 'AI-Practices' }],
    ['meta', { property: 'og:description', content: 'A Systematic Approach to AI Research & Engineering' }],
    ['meta', { property: 'og:url', content: 'https://zimingttkx.github.io/AI-Practices/' }],
    ['meta', { name: 'twitter:card', content: 'summary_large_image' }],
    ['meta', { name: 'twitter:title', content: 'AI-Practices' }],
    ['meta', { name: 'twitter:description', content: 'A Systematic Approach to AI Research & Engineering' }],
  ],

  // Markdown 配置
  markdown: {
    theme: {
      light: 'github-light',
      dark: 'github-dark'
    },
    lineNumbers: true,
    math: true
  },

  sitemap: {
    hostname: 'https://zimingttkx.github.io/AI-Practices/'
  },

  lastUpdated: true,
  cleanUrls: true,

  // 多语言配置
  locales: {
    root: {
      label: '简体中文',
      lang: 'zh-CN',
      themeConfig: {
        nav: [
          { text: '首页', link: '/' },
          {
            text: '入门指南',
            items: [
              { text: '快速开始', link: '/zh/guide/getting-started' },
              { text: '安装配置', link: '/zh/guide/installation' },
              { text: '项目架构', link: '/zh/guide/architecture' }
            ]
          },
          {
            text: '学习模块',
            items: [
              { text: '模块概览', link: '/zh/modules/' },
              { text: '01 - 机器学习基础', link: '/zh/modules/01-foundations' },
              { text: '02 - 神经网络', link: '/zh/modules/02-neural-networks' },
              { text: '03 - 计算机视觉', link: '/zh/modules/03-computer-vision' },
              { text: '04 - 序列模型', link: '/zh/modules/04-sequence-models' },
              { text: '05 - 高级专题', link: '/zh/modules/05-advanced' },
              { text: '06 - 生成模型', link: '/zh/modules/06-generative' },
              { text: '07 - 强化学习', link: '/zh/modules/07-reinforcement-learning' },
              { text: '08 - 理论笔记', link: '/zh/modules/08-theory' },
              { text: '09 - 实战项目', link: '/zh/modules/09-projects' },
              { text: '10 - 大语言模型', link: '/zh/modules/10-llm' },
              { text: '11 - 多模态学习', link: '/zh/modules/11-multimodal' },
              { text: '12 - 部署优化', link: '/zh/modules/12-deployment' },
              { text: '13 - 分布式训练', link: '/zh/modules/13-distributed' },
              { text: '14 - 智能体推理', link: '/zh/modules/14-agents' }
            ]
          },
          { text: 'GitHub', link: 'https://github.com/zimingttkx/AI-Practices' }
        ],
        sidebar: {
          '/zh/guide/': [
            {
              text: '入门指南',
              collapsed: false,
              items: [
                { text: '快速开始', link: '/zh/guide/getting-started' },
                { text: '安装配置', link: '/zh/guide/installation' },
                { text: '项目架构', link: '/zh/guide/architecture' }
              ]
            }
          ],
          '/zh/modules/': [
            {
              text: '基础模块',
              collapsed: false,
              items: [
                { text: '模块概览', link: '/zh/modules/' },
                { text: '01 - 机器学习基础', link: '/zh/modules/01-foundations' },
                { text: '02 - 神经网络', link: '/zh/modules/02-neural-networks' }
              ]
            },
            {
              text: '核心模块',
              collapsed: false,
              items: [
                { text: '03 - 计算机视觉', link: '/zh/modules/03-computer-vision' },
                { text: '04 - 序列模型', link: '/zh/modules/04-sequence-models' }
              ]
            },
            {
              text: '进阶模块',
              collapsed: false,
              items: [
                { text: '05 - 高级专题', link: '/zh/modules/05-advanced' },
                { text: '06 - 生成模型', link: '/zh/modules/06-generative' },
                { text: '07 - 强化学习', link: '/zh/modules/07-reinforcement-learning' }
              ]
            },
            {
              text: '参考与实战',
              collapsed: false,
              items: [
                { text: '08 - 理论笔记', link: '/zh/modules/08-theory' },
                { text: '09 - 实战项目', link: '/zh/modules/09-projects' }
              ]
            },
            {
              text: '前沿技术',
              collapsed: false,
              items: [
                { text: '10 - 大语言模型', link: '/zh/modules/10-llm' },
                { text: '11 - 多模态学习', link: '/zh/modules/11-multimodal' },
                { text: '12 - 部署优化', link: '/zh/modules/12-deployment' },
                { text: '13 - 分布式训练', link: '/zh/modules/13-distributed' },
                { text: '14 - 智能体推理', link: '/zh/modules/14-agents' }
              ]
            }
          ]
        },
        outline: {
          level: [2, 3],
          label: '页面导航'
        },
        docFooter: {
          prev: '上一页',
          next: '下一页'
        },
        lastUpdated: {
          text: '最后更新于'
        },
        editLink: {
          pattern: 'https://github.com/zimingttkx/AI-Practices/edit/main/docs/:path',
          text: '在 GitHub 上编辑此页'
        },
        returnToTopLabel: '返回顶部',
        sidebarMenuLabel: '菜单',
        darkModeSwitchLabel: '主题',
        lightModeSwitchTitle: '切换到浅色模式',
        darkModeSwitchTitle: '切换到深色模式'
      }
    },
    en: {
      label: 'English',
      lang: 'en-US',
      link: '/en/',
      themeConfig: {
        nav: [
          { text: 'Home', link: '/en/' },
          {
            text: 'Guide',
            items: [
              { text: 'Getting Started', link: '/en/guide/getting-started' },
              { text: 'Installation', link: '/en/guide/installation' },
              { text: 'Architecture', link: '/en/guide/architecture' }
            ]
          },
          {
            text: 'Modules',
            items: [
              { text: 'Overview', link: '/en/modules/' },
              { text: '01 - ML Foundations', link: '/en/modules/01-foundations' },
              { text: '02 - Neural Networks', link: '/en/modules/02-neural-networks' },
              { text: '03 - Computer Vision', link: '/en/modules/03-computer-vision' },
              { text: '04 - Sequence Models', link: '/en/modules/04-sequence-models' },
              { text: '05 - Advanced Topics', link: '/en/modules/05-advanced' },
              { text: '06 - Generative Models', link: '/en/modules/06-generative' },
              { text: '07 - Reinforcement Learning', link: '/en/modules/07-reinforcement-learning' },
              { text: '08 - Theory Notes', link: '/en/modules/08-theory' },
              { text: '09 - Projects', link: '/en/modules/09-projects' },
              { text: '10 - Large Language Models', link: '/en/modules/10-llm' },
              { text: '11 - Multimodal Learning', link: '/en/modules/11-multimodal' },
              { text: '12 - Deployment', link: '/en/modules/12-deployment' },
              { text: '13 - Distributed Training', link: '/en/modules/13-distributed' },
              { text: '14 - Agents & Reasoning', link: '/en/modules/14-agents' }
            ]
          },
          { text: 'GitHub', link: 'https://github.com/zimingttkx/AI-Practices' }
        ],
        sidebar: {
          '/en/guide/': [
            {
              text: 'Guide',
              collapsed: false,
              items: [
                { text: 'Getting Started', link: '/en/guide/getting-started' },
                { text: 'Installation', link: '/en/guide/installation' },
                { text: 'Architecture', link: '/en/guide/architecture' }
              ]
            }
          ],
          '/en/modules/': [
            {
              text: 'Foundation',
              collapsed: false,
              items: [
                { text: 'Overview', link: '/en/modules/' },
                { text: '01 - ML Foundations', link: '/en/modules/01-foundations' },
                { text: '02 - Neural Networks', link: '/en/modules/02-neural-networks' }
              ]
            },
            {
              text: 'Core Modules',
              collapsed: false,
              items: [
                { text: '03 - Computer Vision', link: '/en/modules/03-computer-vision' },
                { text: '04 - Sequence Models', link: '/en/modules/04-sequence-models' }
              ]
            },
            {
              text: 'Advanced',
              collapsed: false,
              items: [
                { text: '05 - Advanced Topics', link: '/en/modules/05-advanced' },
                { text: '06 - Generative Models', link: '/en/modules/06-generative' },
                { text: '07 - Reinforcement Learning', link: '/en/modules/07-reinforcement-learning' }
              ]
            },
            {
              text: 'Reference & Practice',
              collapsed: false,
              items: [
                { text: '08 - Theory Notes', link: '/en/modules/08-theory' },
                { text: '09 - Projects', link: '/en/modules/09-projects' }
              ]
            },
            {
              text: 'Frontier Tech',
              collapsed: false,
              items: [
                { text: '10 - Large Language Models', link: '/en/modules/10-llm' },
                { text: '11 - Multimodal Learning', link: '/en/modules/11-multimodal' },
                { text: '12 - Deployment', link: '/en/modules/12-deployment' },
                { text: '13 - Distributed Training', link: '/en/modules/13-distributed' },
                { text: '14 - Agents & Reasoning', link: '/en/modules/14-agents' }
              ]
            }
          ]
        },
        outline: {
          level: [2, 3],
          label: 'On this page'
        },
        editLink: {
          pattern: 'https://github.com/zimingttkx/AI-Practices/edit/main/docs/:path',
          text: 'Edit this page on GitHub'
        },
        returnToTopLabel: 'Return to top',
        sidebarMenuLabel: 'Menu',
        darkModeSwitchLabel: 'Appearance',
        lightModeSwitchTitle: 'Switch to light theme',
        darkModeSwitchTitle: 'Switch to dark theme'
      }
    }
  },

  themeConfig: {
    logo: '/logo.svg',
    siteTitle: 'AI-Practices',

    socialLinks: [
      { icon: 'github', link: 'https://github.com/zimingttkx/AI-Practices' }
    ],

    search: {
      provider: 'local',
      options: {
        locales: {
          root: {
            translations: {
              button: {
                buttonText: '搜索文档',
                buttonAriaLabel: '搜索文档'
              },
              modal: {
                noResultsText: '无法找到相关结果',
                resetButtonTitle: '清除查询条件',
                footer: {
                  selectText: '选择',
                  navigateText: '切换',
                  closeText: '关闭'
                }
              }
            }
          },
          en: {
            translations: {
              button: {
                buttonText: 'Search',
                buttonAriaLabel: 'Search'
              },
              modal: {
                noResultsText: 'No results found',
                resetButtonTitle: 'Clear query',
                footer: {
                  selectText: 'Select',
                  navigateText: 'Navigate',
                  closeText: 'Close'
                }
              }
            }
          }
        }
      }
    },

    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2024-present zimingttkx'
    }
  }
})
