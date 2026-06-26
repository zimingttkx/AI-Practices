# 强化学习 (Reinforcement Learning)

从基础理论到前沿算法的完整强化学习学习路径。

## 前置知识

- ✅ 必需：Python、NumPy、概率论基础
- 📖 推荐：马尔可夫链、动态规划
- 🔗 前序模块：[02-neural-networks](../02-neural-networks)

## 目录结构

```
07-reinforcement-learning/
├── 01-mdp-basics/           # MDP基础与动态规划
├── 02-temporal-difference/  # 时序差分学习
├── 03-q-learning/           # Q学习算法
├── 04-deep-q-learning/      # 深度Q网络(DQN)
├── 05-policy-gradient/      # 策略梯度方法
├── 06-actor-critic/         # Actor-Critic方法
├── 07-advanced-algorithms/  # 高级算法(DDPG/TD3/SAC)
├── 08-reward-optimization/  # 奖励优化与探索
├── tools/                   # 工具库(Gymnasium/TF-Agents)
└── KNOWLEDGE_SYSTEM.md      # 知识体系总览
```

## 学习路径

```
MDP基础 → 时序差分 → Q学习 → DQN → 策略梯度 → Actor-Critic → 高级算法
   ↓                                                              ↓
动态规划                                                    奖励优化/探索
```

## 模块概览

| 模块 | 核心内容 | 关键算法 |
|------|----------|----------|
| 01-mdp-basics | 状态、动作、奖励、策略 | 值迭代、策略迭代 |
| 02-temporal-difference | 在线学习、自举 | TD(0)、TD(λ)、SARSA |
| 03-q-learning | 离策略学习 | Q-Learning、Double Q |
| 04-deep-q-learning | 函数逼近 | DQN、Double DQN、Dueling、PER |
| 05-policy-gradient | 直接策略优化 | REINFORCE、基线方法 |
| 06-actor-critic | 混合方法 | A2C、A3C、PPO、GAE |
| 07-advanced-algorithms | 连续控制 | DDPG、TD3、SAC |
| 08-reward-optimization | 奖励设计 | 奖励塑形、好奇心驱动、HER |

## 快速开始

```bash
# 安装依赖
pip install torch numpy gymnasium matplotlib

# 运行DQN示例
cd 04-deep-q-learning
python train.py --env CartPole-v1 --double --dueling
```

## 参考资源

- [Sutton & Barto - RL: An Introduction](http://incompleteideas.net/book/the-book.html)
- [David Silver RL Course](https://www.davidsilver.uk/teaching/)
- [OpenAI Spinning Up](https://spinningup.openai.com/)
