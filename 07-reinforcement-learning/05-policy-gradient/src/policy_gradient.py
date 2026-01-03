#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略梯度方法 (Policy Gradient Methods)

本模块实现了强化学习中的策略梯度系列算法，包括：
    - REINFORCE (Williams, 1992): 蒙特卡洛策略梯度
    - REINFORCE with Baseline: 引入基线减少方差
    - A2C (Advantage Actor-Critic): 优势函数演员-评论家
    - GAE (Generalized Advantage Estimation): 广义优势估计
    - 连续动作空间的高斯策略

理论背景
--------
策略梯度方法直接参数化策略 π_θ(a|s)，通过梯度上升最大化期望回报：

    J(θ) = E_τ~π_θ[R(τ)] = E_τ~π_θ[Σ_t γ^t r_t]

策略梯度定理 (Sutton et al., 1999):

    ∇_θ J(θ) = E_π_θ[∇_θ log π_θ(a|s) · Q^π(s,a)]

其中 Q^π(s,a) 可用不同方式估计：
    - Monte Carlo: G_t = Σ_{k=t}^T γ^{k-t} r_k (REINFORCE)
    - TD(0): r_t + γV(s_{t+1}) (Actor-Critic)
    - n-step: Σ_{k=0}^{n-1} γ^k r_{t+k} + γ^n V(s_{t+n})
    - GAE: 指数加权的多步TD误差

参考文献
--------
[1] Williams, R.J. (1992). Simple statistical gradient-following algorithms
    for connectionist reinforcement learning. Machine Learning.
[2] Sutton, R.S. et al. (1999). Policy gradient methods for reinforcement
    learning with function approximation. NeurIPS.
[3] Schulman, J. et al. (2016). High-dimensional continuous control using
    generalized advantage estimation. ICLR.
[4] Mnih, V. et al. (2016). Asynchronous methods for deep reinforcement
    learning. ICML.

运行环境
--------
    pip install torch>=1.9.0 gymnasium>=0.26.0 numpy matplotlib

Author: zimingttkx
Date: 2024
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    List, Tuple, Optional, Dict, Any, Union, Callable, NamedTuple
)
from collections import deque
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal

# 尝试导入gymnasium，失败则提供友好提示
try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False
    gym = None
    spaces = None

# 尝试导入matplotlib用于可视化
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


# ============================================================================
#                           数据结构定义
# ============================================================================

class Transition(NamedTuple):
    """单步转移数据"""
    state: np.ndarray
    action: Union[int, np.ndarray]
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: Optional[torch.Tensor] = None
    value: Optional[torch.Tensor] = None


@dataclass
class EpisodeBuffer:
    """
    回合数据缓冲区

    存储一个完整回合的轨迹数据，用于策略更新。

    Attributes
    ----------
    states : List[np.ndarray]
        状态序列
    actions : List[Union[int, np.ndarray]]
        动作序列
    rewards : List[float]
        奖励序列
    log_probs : List[torch.Tensor]
        动作对数概率序列
    values : List[torch.Tensor]
        状态价值估计序列
    dones : List[bool]
        终止标志序列
    """
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[Union[int, np.ndarray]] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    log_probs: List[torch.Tensor] = field(default_factory=list)
    values: List[torch.Tensor] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    entropies: List[torch.Tensor] = field(default_factory=list)

    def store(
        self,
        state: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        log_prob: torch.Tensor,
        value: Optional[torch.Tensor] = None,
        done: bool = False,
        entropy: Optional[torch.Tensor] = None
    ) -> None:
        """存储单步数据"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        if value is not None:
            self.values.append(value)
        if entropy is not None:
            self.entropies.append(entropy)

    def clear(self) -> None:
        """清空缓冲区"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
        self.entropies.clear()

    def __len__(self) -> int:
        return len(self.rewards)

    @property
    def total_reward(self) -> float:
        """计算回合总奖励"""
        return sum(self.rewards)


@dataclass
class TrainingConfig:
    """
    训练配置参数

    Attributes
    ----------
    gamma : float
        折扣因子，控制对未来奖励的重视程度。
        典型值: 0.99 (长期任务), 0.95 (短期任务)
    lr_actor : float
        策略网络学习率。建议比critic小，策略变化应平稳。
    lr_critic : float
        价值网络学习率。可以比actor大，价值估计收敛更快。
    entropy_coef : float
        熵正则化系数，鼓励探索。太大导致随机策略，太小导致过早收敛。
    value_coef : float
        价值损失系数，平衡actor和critic的训练。
    max_grad_norm : float
        梯度裁剪阈值，防止梯度爆炸。
    gae_lambda : float
        GAE的λ参数，控制偏差-方差权衡。
        λ=0: TD(0)，低方差高偏差
        λ=1: Monte Carlo，高方差低偏差
    n_steps : int
        n-step return的步数。
    normalize_advantage : bool
        是否标准化优势函数，减少方差。
    """
    gamma: float = 0.99
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    gae_lambda: float = 0.95
    n_steps: int = 5
    normalize_advantage: bool = True
    device: str = "cpu"


# ============================================================================
#                           神经网络模块
# ============================================================================

def init_weights(module: nn.Module, gain: float = 1.0) -> None:
    """
    正交初始化网络权重

    正交初始化 (Saxe et al., 2014) 有助于保持梯度的稳定传播，
    对于深度网络和RNN尤其有效。

    Parameters
    ----------
    module : nn.Module
        待初始化的模块
    gain : float
        增益因子，用于调整初始化的尺度
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class MLP(nn.Module):
    """
    多层感知机基础模块

    构建具有指定层数和激活函数的前馈神经网络。
    支持自定义最后一层的激活函数。

    Parameters
    ----------
    input_dim : int
        输入维度
    output_dim : int
        输出维度
    hidden_dims : List[int]
        隐藏层维度列表
    activation : nn.Module
        隐藏层激活函数
    output_activation : Optional[nn.Module]
        输出层激活函数，None表示无激活

    Example
    -------
    >>> mlp = MLP(4, 2, [128, 128], nn.ReLU(), nn.Softmax(dim=-1))
    >>> x = torch.randn(32, 4)
    >>> out = mlp(x)  # shape: (32, 2)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [128, 128],
        activation: nn.Module = nn.ReLU(),
        output_activation: Optional[nn.Module] = None
    ):
        super().__init__()

        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            # 最后一层使用输出激活函数
            if i < len(dims) - 2:
                layers.append(activation)
            elif output_activation is not None:
                layers.append(output_activation)

        self.net = nn.Sequential(*layers)

        # 应用正交初始化
        self.apply(lambda m: init_weights(m, gain=np.sqrt(2)))
        # 输出层使用较小的初始化
        init_weights(self.net[-1] if output_activation is None else self.net[-2], gain=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiscretePolicy(nn.Module):
    """
    离散动作空间的策略网络

    使用Softmax输出动作概率分布。对于离散动作空间，
    策略定义为类别分布：π(a|s) = softmax(f_θ(s))_a

    Parameters
    ----------
    state_dim : int
        状态空间维度
    action_dim : int
        动作空间大小（离散动作数量）
    hidden_dims : List[int]
        隐藏层维度

    Attributes
    ----------
    net : MLP
        策略网络主体

    Notes
    -----
    为了数值稳定性，网络输出logits而非概率，
    在采样时使用Categorical分布处理。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [128, 128]
    ):
        super().__init__()
        self.net = MLP(state_dim, action_dim, hidden_dims)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        计算动作logits

        Parameters
        ----------
        state : torch.Tensor
            状态张量，shape: (batch_size, state_dim)

        Returns
        -------
        torch.Tensor
            动作logits，shape: (batch_size, action_dim)
        """
        return self.net(state)

    def get_distribution(self, state: torch.Tensor) -> Categorical:
        """获取动作分布"""
        logits = self.forward(state)
        return Categorical(logits=logits)

    def sample(
        self,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        从策略中采样动作

        Parameters
        ----------
        state : torch.Tensor
            状态张量

        Returns
        -------
        action : torch.Tensor
            采样的动作
        log_prob : torch.Tensor
            动作的对数概率
        entropy : torch.Tensor
            分布的熵
        """
        dist = self.get_distribution(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy

    def evaluate(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        评估给定状态-动作对的对数概率

        用于重要性采样和PPO等需要重新计算旧轨迹概率的算法。
        """
        dist = self.get_distribution(state)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy


class ContinuousPolicy(nn.Module):
    """
    连续动作空间的高斯策略网络

    输出高斯分布的均值和标准差：π(a|s) = N(μ_θ(s), σ_θ(s)²)

    对于有界动作空间，使用Tanh压缩，并相应调整对数概率：
        a = tanh(u), u ~ N(μ, σ²)
        log π(a|s) = log N(u|μ,σ²) - Σ log(1 - tanh²(u))

    Parameters
    ----------
    state_dim : int
        状态空间维度
    action_dim : int
        动作空间维度
    hidden_dims : List[int]
        隐藏层维度
    log_std_min : float
        对数标准差下界，防止策略坍塌
    log_std_max : float
        对数标准差上界，防止过度探索

    Notes
    -----
    使用独立的网络输出均值，标准差作为可学习参数或状态依赖。
    对数空间参数化标准差保证正值。
    """

    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        state_dependent_std: bool = False
    ):
        super().__init__()

        self.action_dim = action_dim
        self.state_dependent_std = state_dependent_std

        # 共享特征提取层
        self.feature_net = MLP(
            state_dim,
            hidden_dims[-1],
            hidden_dims[:-1],
            activation=nn.ReLU()
        )

        # 均值输出层
        self.mean_layer = nn.Linear(hidden_dims[-1], action_dim)
        init_weights(self.mean_layer, gain=0.01)

        # 标准差参数
        if state_dependent_std:
            self.log_std_layer = nn.Linear(hidden_dims[-1], action_dim)
            init_weights(self.log_std_layer, gain=0.01)
        else:
            # 状态无关的可学习对数标准差
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(
        self,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算高斯分布参数

        Returns
        -------
        mean : torch.Tensor
            动作均值
        std : torch.Tensor
            动作标准差
        """
        features = self.feature_net(state)
        mean = self.mean_layer(features)

        if self.state_dependent_std:
            log_std = self.log_std_layer(features)
        else:
            log_std = self.log_std.expand_as(mean)

        # 裁剪对数标准差
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()

        return mean, std

    def get_distribution(self, state: torch.Tensor) -> Normal:
        """获取高斯动作分布"""
        mean, std = self.forward(state)
        return Normal(mean, std)

    def sample(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        从高斯策略采样动作

        Parameters
        ----------
        state : torch.Tensor
            状态张量
        deterministic : bool
            是否使用确定性策略（直接输出均值）

        Returns
        -------
        action : torch.Tensor
            采样/确定性动作，已通过tanh压缩到[-1,1]
        log_prob : torch.Tensor
            动作对数概率（已做变量变换校正）
        entropy : torch.Tensor
            高斯分布的熵
        """
        mean, std = self.forward(state)

        if deterministic:
            # 确定性动作：直接使用均值
            action = torch.tanh(mean)
            # 确定性情况下log_prob无意义，返回0
            log_prob = torch.zeros(state.shape[0], device=state.device)
        else:
            # 随机采样
            dist = Normal(mean, std)
            # 使用重参数化技巧采样
            u = dist.rsample()
            action = torch.tanh(u)

            # 计算对数概率，考虑tanh变换的Jacobian
            # log π(a) = log N(u) - Σ log(1 - tanh²(u))
            log_prob = dist.log_prob(u).sum(dim=-1)
            # Jacobian校正项：log(1 - tanh²(u)) = log(1 - a²)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)

        # 高斯分布的熵
        entropy = dist.entropy().sum(dim=-1) if not deterministic else torch.zeros_like(log_prob)

        return action, log_prob, entropy

    def evaluate(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """评估给定动作的对数概率"""
        mean, std = self.forward(state)
        dist = Normal(mean, std)

        # 反向计算u = arctanh(action)
        # 数值稳定性：裁剪到(-1+ε, 1-ε)
        action_clipped = torch.clamp(action, -1 + 1e-6, 1 - 1e-6)
        u = torch.atanh(action_clipped)

        log_prob = dist.log_prob(u).sum(dim=-1)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy


class ValueNetwork(nn.Module):
    """
    状态价值函数网络 V(s)

    估计给定状态的期望累积回报。用作策略梯度的基线
    或Actor-Critic中的Critic。

    Parameters
    ----------
    state_dim : int
        状态空间维度
    hidden_dims : List[int]
        隐藏层维度
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int] = [128, 128]
    ):
        super().__init__()
        self.net = MLP(state_dim, 1, hidden_dims)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        估计状态价值

        Returns
        -------
        torch.Tensor
            状态价值估计，shape: (batch_size, 1)
        """
        return self.net(state)


class ActorCriticNetwork(nn.Module):
    """
    共享特征的Actor-Critic网络

    Actor和Critic共享底层特征提取网络，然后分别输出
    策略分布和价值估计。共享特征可以：
    1. 减少参数量
    2. 通过多任务学习提升特征质量
    3. 加速训练收敛

    Parameters
    ----------
    state_dim : int
        状态空间维度
    action_dim : int
        动作空间大小
    hidden_dim : int
        隐藏层维度
    continuous : bool
        是否为连续动作空间

    Architecture
    ------------
    state -> [shared_net] -> features
                              |-> [actor_head] -> policy
                              |-> [critic_head] -> value
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        continuous: bool = False
    ):
        super().__init__()

        self.continuous = continuous
        self.action_dim = action_dim

        # 共享特征提取层
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor头
        if continuous:
            self.actor_mean = nn.Linear(hidden_dim, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.actor_head = nn.Linear(hidden_dim, action_dim)

        # Critic头
        self.critic_head = nn.Linear(hidden_dim, 1)

        # 初始化
        self.apply(lambda m: init_weights(m, gain=np.sqrt(2)))
        if continuous:
            init_weights(self.actor_mean, gain=0.01)
        else:
            init_weights(self.actor_head, gain=0.01)
        init_weights(self.critic_head, gain=1.0)

    def forward(
        self,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Returns
        -------
        policy_output : torch.Tensor
            离散：动作logits
            连续：(mean, log_std)
        value : torch.Tensor
            状态价值估计
        """
        features = self.shared_net(state)
        value = self.critic_head(features)

        if self.continuous:
            mean = self.actor_mean(features)
            return (mean, self.actor_log_std), value
        else:
            logits = self.actor_head(features)
            return logits, value

    def get_action_and_value(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取动作、对数概率、熵和价值

        如果提供了action，则计算该动作的对数概率（用于PPO）；
        否则从策略中采样新动作。
        """
        features = self.shared_net(state)
        value = self.critic_head(features)

        if self.continuous:
            mean = self.actor_mean(features)
            std = self.actor_log_std.exp()
            dist = Normal(mean, std)

            if action is None:
                u = dist.rsample()
                action = torch.tanh(u)
            else:
                action_clipped = torch.clamp(action, -1 + 1e-6, 1 - 1e-6)
                u = torch.atanh(action_clipped)

            log_prob = dist.log_prob(u).sum(dim=-1)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        else:
            logits = self.actor_head(features)
            dist = Categorical(logits=logits)

            if action is None:
                action = dist.sample()

            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

        return action, log_prob, entropy, value.squeeze(-1)


# ============================================================================
#                           回报计算工具
# ============================================================================

def compute_returns(
    rewards: List[float],
    gamma: float,
    normalize: bool = True
) -> torch.Tensor:
    """
    计算蒙特卡洛回报 (Monte Carlo Returns)

    G_t = r_t + γr_{t+1} + γ²r_{t+2} + ... = Σ_{k=0}^{T-t} γ^k r_{t+k}

    从后向前累积计算，时间复杂度O(T)。

    Parameters
    ----------
    rewards : List[float]
        奖励序列 [r_0, r_1, ..., r_{T-1}]
    gamma : float
        折扣因子
    normalize : bool
        是否标准化回报（减均值除标准差）

    Returns
    -------
    torch.Tensor
        回报序列 [G_0, G_1, ..., G_{T-1}]

    Example
    -------
    >>> rewards = [1.0, 1.0, 1.0]
    >>> gamma = 0.99
    >>> returns = compute_returns(rewards, gamma)
    >>> # G_2 = 1.0
    >>> # G_1 = 1.0 + 0.99 * 1.0 = 1.99
    >>> # G_0 = 1.0 + 0.99 * 1.99 = 2.9701
    """
    returns = []
    G = 0.0

    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32)

    if normalize and len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    return returns


def compute_gae(
    rewards: List[float],
    values: List[torch.Tensor],
    next_value: float,
    dones: List[bool],
    gamma: float,
    gae_lambda: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算广义优势估计 (Generalized Advantage Estimation)

    GAE通过指数加权的多步TD误差来估计优势函数，
    在偏差和方差之间取得平衡：

        δ_t = r_t + γV(s_{t+1}) - V(s_t)  (TD误差)
        A_t^GAE = Σ_{k=0}^{∞} (γλ)^k δ_{t+k}

    λ=0: A_t = δ_t = r_t + γV(s_{t+1}) - V(s_t) (TD(0), 高偏差低方差)
    λ=1: A_t = G_t - V(s_t) (Monte Carlo, 低偏差高方差)

    Parameters
    ----------
    rewards : List[float]
        奖励序列
    values : List[torch.Tensor]
        价值估计序列
    next_value : float
        最后状态的下一状态价值（用于bootstrap）
    dones : List[bool]
        回合终止标志
    gamma : float
        折扣因子
    gae_lambda : float
        GAE的λ参数

    Returns
    -------
    advantages : torch.Tensor
        GAE优势估计
    returns : torch.Tensor
        目标回报（用于训练critic）
    """
    advantages = []
    gae = 0.0

    # 提取values的数值
    values_list = [v.item() if isinstance(v, torch.Tensor) else v for v in values]
    values_list.append(next_value)

    # 从后向前计算
    for t in reversed(range(len(rewards))):
        # 如果回合结束，下一状态价值为0
        next_val = 0.0 if dones[t] else values_list[t + 1]

        # TD误差
        delta = rewards[t] + gamma * next_val - values_list[t]

        # GAE累积
        # 如果回合结束，重置gae
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    advantages = torch.tensor(advantages, dtype=torch.float32)

    # 计算目标回报：A + V = G（近似）
    returns = advantages + torch.tensor(values_list[:-1], dtype=torch.float32)

    return advantages, returns


def compute_n_step_returns(
    rewards: List[float],
    values: List[torch.Tensor],
    next_value: float,
    dones: List[bool],
    gamma: float,
    n_steps: int
) -> torch.Tensor:
    """
    计算n-step回报

    G_t^{(n)} = r_t + γr_{t+1} + ... + γ^{n-1}r_{t+n-1} + γ^n V(s_{t+n})

    n-step方法在TD(0)（n=1）和Monte Carlo（n=∞）之间取得平衡：
    - 小n: 低方差，但可能有偏差（依赖于V的准确性）
    - 大n: 高方差，但偏差更小

    Parameters
    ----------
    n_steps : int
        向前看的步数

    Returns
    -------
    torch.Tensor
        n-step回报序列
    """
    T = len(rewards)
    returns = []

    # 准备values数组
    values_list = [v.item() if isinstance(v, torch.Tensor) else v for v in values]
    values_list.append(next_value)

    for t in range(T):
        G = 0.0

        # 累积n步奖励
        for k in range(n_steps):
            if t + k >= T:
                break
            if dones[t + k] and k > 0:
                break
            G += (gamma ** k) * rewards[t + k]

            if dones[t + k]:
                break
        else:
            # 如果没有遇到done，加上bootstrap值
            if t + n_steps < T:
                if not dones[t + n_steps - 1]:
                    G += (gamma ** n_steps) * values_list[t + n_steps]
            else:
                G += (gamma ** (T - t)) * next_value

        returns.append(G)

    return torch.tensor(returns, dtype=torch.float32)


# ============================================================================
#                           策略梯度算法
# ============================================================================

class BasePolicyGradient(ABC):
    """
    策略梯度算法基类

    定义了策略梯度算法的通用接口和工具方法。
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.training_info = {
            "episode_rewards": [],
            "episode_lengths": [],
            "losses": []
        }

    @abstractmethod
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[Any, Dict[str, torch.Tensor]]:
        """选择动作"""
        pass

    @abstractmethod
    def update(self, buffer: EpisodeBuffer) -> Dict[str, float]:
        """更新策略"""
        pass

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        """将numpy数组转换为tensor"""
        return torch.tensor(x, dtype=torch.float32, device=self.device)


class REINFORCE(BasePolicyGradient):
    """
    REINFORCE算法 (Williams, 1992)

    最基础的策略梯度算法，使用蒙特卡洛回报作为Q值的无偏估计：

        ∇_θ J(θ) ≈ (1/N) Σ_i Σ_t ∇_θ log π_θ(a_t|s_t) G_t

    其中 G_t = Σ_{k=t}^T γ^{k-t} r_k 是从时刻t开始的累积折扣回报。

    特点：
    - 无偏估计
    - 高方差（需要大量样本）
    - 需要完整回合才能更新（on-policy, episodic）

    Parameters
    ----------
    state_dim : int
        状态空间维度
    action_dim : int
        动作空间大小
    config : TrainingConfig
        训练配置
    continuous : bool
        是否为连续动作空间

    Example
    -------
    >>> agent = REINFORCE(4, 2, TrainingConfig(gamma=0.99, lr_actor=1e-3))
    >>> # 收集一个回合
    >>> state = env.reset()
    >>> while not done:
    ...     action, info = agent.select_action(state)
    ...     next_state, reward, done, _ = env.step(action)
    ...     buffer.store(state, action, reward, info['log_prob'])
    ...     state = next_state
    >>> # 回合结束，更新
    >>> loss_info = agent.update(buffer)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: TrainingConfig = TrainingConfig(),
        continuous: bool = False
    ):
        super().__init__(config)

        self.continuous = continuous

        # 构建策略网络
        if continuous:
            self.policy = ContinuousPolicy(state_dim, action_dim).to(self.device)
        else:
            self.policy = DiscretePolicy(state_dim, action_dim).to(self.device)

        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=config.lr_actor
        )

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[Union[int, np.ndarray], Dict[str, torch.Tensor]]:
        """
        根据当前策略选择动作

        Parameters
        ----------
        state : np.ndarray
            当前状态
        deterministic : bool
            是否使用确定性策略（测试时使用）

        Returns
        -------
        action : int or np.ndarray
            选择的动作
        info : dict
            包含log_prob和entropy的字典
        """
        state_t = self._to_tensor(state).unsqueeze(0)

        with torch.no_grad() if deterministic else torch.enable_grad():
            if self.continuous:
                action, log_prob, entropy = self.policy.sample(state_t, deterministic)
                action = action.squeeze(0).cpu().numpy()
            else:
                if deterministic:
                    # 确定性策略：选择概率最大的动作
                    logits = self.policy(state_t)
                    action = logits.argmax(dim=-1).item()
                    log_prob = torch.zeros(1)
                    entropy = torch.zeros(1)
                else:
                    action, log_prob, entropy = self.policy.sample(state_t)
                    action = action.item()

        return action, {"log_prob": log_prob, "entropy": entropy}

    def update(self, buffer: EpisodeBuffer) -> Dict[str, float]:
        """
        回合结束后更新策略

        使用REINFORCE梯度估计：
            ∇_θ J ≈ Σ_t ∇_θ log π_θ(a_t|s_t) G_t

        Returns
        -------
        dict
            包含policy_loss和entropy的训练信息
        """
        # 计算蒙特卡洛回报
        returns = compute_returns(
            buffer.rewards,
            self.config.gamma,
            normalize=self.config.normalize_advantage
        )

        # 堆叠log_probs
        log_probs = torch.stack(buffer.log_probs)

        # 策略损失：-E[log π(a|s) * G]
        # 负号是因为我们用梯度下降（minimize），但目标是maximize
        policy_loss = -(log_probs * returns).mean()

        # 熵正则化（可选）
        if buffer.entropies and self.config.entropy_coef > 0:
            entropies = torch.stack(buffer.entropies)
            entropy_loss = -entropies.mean()
            total_loss = policy_loss + self.config.entropy_coef * entropy_loss
        else:
            total_loss = policy_loss
            entropy_loss = torch.tensor(0.0)

        # 反向传播和优化
        self.optimizer.zero_grad()
        total_loss.backward()

        # 梯度裁剪
        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.config.max_grad_norm
            )

        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "entropy": -entropy_loss.item() if isinstance(entropy_loss, torch.Tensor) else 0.0,
            "mean_return": returns.mean().item()
        }


class REINFORCEBaseline(BasePolicyGradient):
    """
    带基线的REINFORCE算法

    引入状态价值函数V(s)作为基线(baseline)来减少方差：

        ∇_θ J(θ) ≈ Σ_t ∇_θ log π_θ(a_t|s_t) (G_t - V(s_t))

    其中 A_t = G_t - V(s_t) 称为优势函数(advantage function)，
    表示动作相对于平均水平的好坏程度。

    理论保证：任何只依赖于状态的基线b(s)都不改变梯度期望
        E[∇_θ log π(a|s) b(s)] = 0

    但V(s)能有效减少方差，因为它移除了与动作无关的回报波动。

    Parameters
    ----------
    state_dim : int
        状态空间维度
    action_dim : int
        动作空间大小
    config : TrainingConfig
        训练配置（包含lr_actor和lr_critic）
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: TrainingConfig = TrainingConfig(),
        continuous: bool = False
    ):
        super().__init__(config)

        self.continuous = continuous

        # 策略网络（Actor）
        if continuous:
            self.policy = ContinuousPolicy(state_dim, action_dim).to(self.device)
        else:
            self.policy = DiscretePolicy(state_dim, action_dim).to(self.device)

        # 价值网络（Baseline/Critic）
        self.value_net = ValueNetwork(state_dim).to(self.device)

        # 分开的优化器（可以使用不同学习率）
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(),
            lr=config.lr_actor
        )
        self.value_optimizer = optim.Adam(
            self.value_net.parameters(),
            lr=config.lr_critic
        )

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[Union[int, np.ndarray], Dict[str, torch.Tensor]]:
        """选择动作并返回价值估计"""
        state_t = self._to_tensor(state).unsqueeze(0)

        # 获取价值估计
        with torch.no_grad():
            value = self.value_net(state_t)

        # 获取动作
        with torch.no_grad() if deterministic else torch.enable_grad():
            if self.continuous:
                action, log_prob, entropy = self.policy.sample(state_t, deterministic)
                action = action.squeeze(0).cpu().numpy()
            else:
                if deterministic:
                    logits = self.policy(state_t)
                    action = logits.argmax(dim=-1).item()
                    log_prob = torch.zeros(1)
                    entropy = torch.zeros(1)
                else:
                    action, log_prob, entropy = self.policy.sample(state_t)
                    action = action.item()

        return action, {
            "log_prob": log_prob,
            "value": value,
            "entropy": entropy
        }

    def update(self, buffer: EpisodeBuffer) -> Dict[str, float]:
        """
        更新策略和价值网络

        策略更新使用优势函数：A_t = G_t - V(s_t)
        价值网络使用MSE损失拟合蒙特卡洛回报
        """
        # 计算蒙特卡洛回报
        returns = compute_returns(
            buffer.rewards,
            self.config.gamma,
            normalize=False  # 标准化在优势计算后进行
        )

        # 获取存储的价值估计
        values = torch.cat(buffer.values).squeeze()
        log_probs = torch.stack(buffer.log_probs)

        # 计算优势：A = G - V
        # 注意：用detach()切断梯度，基线不应影响策略梯度
        advantages = returns - values.detach()

        # 标准化优势（重要的方差减少技巧）
        if self.config.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 策略损失
        policy_loss = -(log_probs * advantages).mean()

        # 价值损失（MSE）
        value_loss = F.mse_loss(values, returns)

        # 熵损失
        if buffer.entropies:
            entropies = torch.stack(buffer.entropies)
            entropy_bonus = entropies.mean()
        else:
            entropy_bonus = torch.tensor(0.0)

        # 更新策略网络
        self.policy_optimizer.zero_grad()
        (policy_loss - self.config.entropy_coef * entropy_bonus).backward()
        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.config.max_grad_norm
            )
        self.policy_optimizer.step()

        # 更新价值网络
        self.value_optimizer.zero_grad()
        value_loss.backward()
        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                self.value_net.parameters(),
                self.config.max_grad_norm
            )
        self.value_optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy_bonus.item(),
            "mean_advantage": advantages.mean().item(),
            "mean_return": returns.mean().item()
        }


class A2C(BasePolicyGradient):
    """
    优势演员-评论家算法 (Advantage Actor-Critic, A2C)

    A2C是一种在线策略梯度算法，结合了：
    - Actor: 学习策略 π_θ(a|s)
    - Critic: 学习价值函数 V_φ(s)

    与REINFORCE不同，A2C使用TD误差进行在线更新：

        δ_t = r_t + γV(s_{t+1}) - V(s_t)  (TD误差/优势估计)

        Actor: ∇_θ J ≈ Σ_t ∇_θ log π_θ(a_t|s_t) δ_t
        Critic: 最小化 (V(s_t) - (r_t + γV(s_{t+1})))²

    A2C还可以使用GAE来获得更好的优势估计。

    优点：
    - 可以单步更新（不需要完整回合）
    - 方差更低（使用价值函数bootstrap）
    - 支持并行环境（A3C的同步版本）

    Parameters
    ----------
    state_dim : int
        状态空间维度
    action_dim : int
        动作空间大小
    config : TrainingConfig
        训练配置
    use_gae : bool
        是否使用GAE计算优势
    shared_network : bool
        Actor和Critic是否共享网络
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: TrainingConfig = TrainingConfig(),
        continuous: bool = False,
        use_gae: bool = True,
        shared_network: bool = True
    ):
        super().__init__(config)

        self.continuous = continuous
        self.use_gae = use_gae
        self.shared_network = shared_network

        if shared_network:
            # 使用共享网络
            self.model = ActorCriticNetwork(
                state_dim, action_dim,
                hidden_dim=256,
                continuous=continuous
            ).to(self.device)

            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config.lr_actor
            )
        else:
            # 分离的网络
            if continuous:
                self.policy = ContinuousPolicy(state_dim, action_dim).to(self.device)
            else:
                self.policy = DiscretePolicy(state_dim, action_dim).to(self.device)

            self.value_net = ValueNetwork(state_dim).to(self.device)

            self.policy_optimizer = optim.Adam(
                self.policy.parameters(),
                lr=config.lr_actor
            )
            self.value_optimizer = optim.Adam(
                self.value_net.parameters(),
                lr=config.lr_critic
            )

        # 用于n-step更新的缓冲区
        self.buffer = EpisodeBuffer()

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[Union[int, np.ndarray], Dict[str, torch.Tensor]]:
        """选择动作"""
        state_t = self._to_tensor(state).unsqueeze(0)

        if self.shared_network:
            with torch.no_grad() if deterministic else torch.enable_grad():
                action, log_prob, entropy, value = self.model.get_action_and_value(
                    state_t
                )

                if self.continuous:
                    action_out = action.squeeze(0).cpu().numpy()
                else:
                    action_out = action.item()
        else:
            with torch.no_grad():
                value = self.value_net(state_t)

            with torch.no_grad() if deterministic else torch.enable_grad():
                if self.continuous:
                    action, log_prob, entropy = self.policy.sample(state_t, deterministic)
                    action_out = action.squeeze(0).cpu().numpy()
                else:
                    if deterministic:
                        logits = self.policy(state_t)
                        action_out = logits.argmax(dim=-1).item()
                        log_prob = torch.zeros(1)
                        entropy = torch.zeros(1)
                    else:
                        action, log_prob, entropy = self.policy.sample(state_t)
                        action_out = action.item()

        return action_out, {
            "log_prob": log_prob,
            "value": value if self.shared_network else value.squeeze(),
            "entropy": entropy
        }

    def update(
        self,
        buffer: EpisodeBuffer,
        next_state: Optional[np.ndarray] = None,
        done: bool = True
    ) -> Dict[str, float]:
        """
        更新Actor和Critic

        Parameters
        ----------
        buffer : EpisodeBuffer
            收集的经验数据
        next_state : np.ndarray, optional
            最后一个状态的下一状态（用于bootstrap）
        done : bool
            回合是否结束
        """
        # 获取下一状态的价值（用于bootstrap）
        if next_state is not None and not done:
            next_state_t = self._to_tensor(next_state).unsqueeze(0)
            with torch.no_grad():
                if self.shared_network:
                    _, next_value = self.model(next_state_t)
                    next_value = next_value.item()
                else:
                    next_value = self.value_net(next_state_t).item()
        else:
            next_value = 0.0

        # 计算优势和回报
        if self.use_gae:
            advantages, returns = compute_gae(
                buffer.rewards,
                buffer.values,
                next_value,
                buffer.dones,
                self.config.gamma,
                self.config.gae_lambda
            )
        else:
            returns = compute_n_step_returns(
                buffer.rewards,
                buffer.values,
                next_value,
                buffer.dones,
                self.config.gamma,
                self.config.n_steps
            )
            values_t = torch.tensor(
                [v.item() if isinstance(v, torch.Tensor) else v for v in buffer.values],
                dtype=torch.float32
            )
            advantages = returns - values_t

        # 标准化优势
        if self.config.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 获取log_probs和values
        log_probs = torch.stack(buffer.log_probs)

        if self.shared_network:
            values = torch.stack([v for v in buffer.values]).squeeze()
        else:
            values = torch.cat(buffer.values).squeeze()

        # 计算损失
        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(values, returns)

        # 熵正则化
        if buffer.entropies:
            entropies = torch.stack(buffer.entropies)
            entropy_bonus = entropies.mean()
        else:
            entropy_bonus = torch.tensor(0.0)

        # 总损失
        total_loss = (
            policy_loss
            + self.config.value_coef * value_loss
            - self.config.entropy_coef * entropy_bonus
        )

        # 优化
        if self.shared_network:
            self.optimizer.zero_grad()
            total_loss.backward()
            if self.config.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
            self.optimizer.step()
        else:
            # 分别更新
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            total_loss.backward()
            if self.config.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.config.max_grad_norm
                )
                nn.utils.clip_grad_norm_(
                    self.value_net.parameters(),
                    self.config.max_grad_norm
                )
            self.policy_optimizer.step()
            self.value_optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy_bonus.item(),
            "total_loss": total_loss.item(),
            "mean_advantage": advantages.mean().item()
        }


# ============================================================================
#                           训练工具
# ============================================================================

class TrainingLogger:
    """训练日志记录器"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.episode_rewards = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)
        self.losses = {}

    def log_episode(self, reward: float, length: int) -> None:
        """记录回合信息"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)

    def log_loss(self, loss_dict: Dict[str, float]) -> None:
        """记录损失信息"""
        for key, value in loss_dict.items():
            if key not in self.losses:
                self.losses[key] = deque(maxlen=self.window_size)
            self.losses[key].append(value)

    @property
    def mean_reward(self) -> float:
        """最近回合的平均奖励"""
        return np.mean(self.episode_rewards) if self.episode_rewards else 0.0

    @property
    def mean_length(self) -> float:
        """最近回合的平均长度"""
        return np.mean(self.episode_lengths) if self.episode_lengths else 0.0

    def get_summary(self) -> str:
        """获取训练摘要"""
        summary = f"Reward: {self.mean_reward:.2f} | Length: {self.mean_length:.1f}"
        for key, values in self.losses.items():
            if values:
                summary += f" | {key}: {np.mean(values):.4f}"
        return summary


def train_policy_gradient(
    agent: BasePolicyGradient,
    env_name: str = "CartPole-v1",
    num_episodes: int = 500,
    max_steps: int = 500,
    log_interval: int = 50,
    render: bool = False,
    seed: Optional[int] = None
) -> Tuple[BasePolicyGradient, List[float]]:
    """
    训练策略梯度智能体

    Parameters
    ----------
    agent : BasePolicyGradient
        策略梯度智能体
    env_name : str
        Gymnasium环境名称
    num_episodes : int
        训练回合数
    max_steps : int
        每回合最大步数
    log_interval : int
        日志打印间隔
    render : bool
        是否渲染环境
    seed : int, optional
        随机种子

    Returns
    -------
    agent : BasePolicyGradient
        训练后的智能体
    rewards_history : List[float]
        每回合的总奖励
    """
    if not HAS_GYMNASIUM:
        raise ImportError("需要安装gymnasium: pip install gymnasium")

    # 创建环境
    env = gym.make(env_name, render_mode="human" if render else None)

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    logger = TrainingLogger()
    rewards_history = []

    print(f"\n{'=' * 60}")
    print(f"训练 {agent.__class__.__name__} on {env_name}")
    print(f"{'=' * 60}\n")

    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed + episode if seed else None)
        buffer = EpisodeBuffer()
        total_reward = 0.0

        for step in range(max_steps):
            # 选择动作
            action, info = agent.select_action(state)

            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 存储经验
            buffer.store(
                state=state,
                action=action,
                reward=reward,
                log_prob=info["log_prob"],
                value=info.get("value"),
                done=done,
                entropy=info.get("entropy")
            )

            total_reward += reward
            state = next_state

            if done:
                break

        # 回合结束，更新策略
        if isinstance(agent, A2C):
            loss_info = agent.update(buffer, next_state, done)
        else:
            loss_info = agent.update(buffer)

        # 记录
        rewards_history.append(total_reward)
        logger.log_episode(total_reward, step + 1)
        logger.log_loss(loss_info)

        # 打印日志
        if (episode + 1) % log_interval == 0:
            print(f"Episode {episode + 1:4d} | {logger.get_summary()}")

    env.close()

    print(f"\n{'=' * 60}")
    print(f"训练完成！最终平均奖励: {logger.mean_reward:.2f}")
    print(f"{'=' * 60}\n")

    return agent, rewards_history


def compare_algorithms(
    env_name: str = "CartPole-v1",
    num_episodes: int = 300,
    seed: int = 42
) -> Dict[str, List[float]]:
    """
    比较不同策略梯度算法的性能

    Parameters
    ----------
    env_name : str
        环境名称
    num_episodes : int
        训练回合数
    seed : int
        随机种子

    Returns
    -------
    Dict[str, List[float]]
        各算法的奖励历史
    """
    if not HAS_GYMNASIUM:
        raise ImportError("需要安装gymnasium: pip install gymnasium")

    # 获取环境信息
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()

    results = {}

    # 配置
    config = TrainingConfig(
        gamma=0.99,
        lr_actor=1e-3,
        lr_critic=1e-3,
        entropy_coef=0.01
    )

    # REINFORCE
    print("\n[1/3] 训练 REINFORCE...")
    agent_rf = REINFORCE(state_dim, action_dim, config)
    _, rewards_rf = train_policy_gradient(
        agent_rf, env_name, num_episodes, log_interval=100, seed=seed
    )
    results["REINFORCE"] = rewards_rf

    # REINFORCE + Baseline
    print("\n[2/3] 训练 REINFORCE + Baseline...")
    agent_rfb = REINFORCEBaseline(state_dim, action_dim, config)
    _, rewards_rfb = train_policy_gradient(
        agent_rfb, env_name, num_episodes, log_interval=100, seed=seed
    )
    results["REINFORCE+Baseline"] = rewards_rfb

    # A2C
    print("\n[3/3] 训练 A2C...")
    agent_a2c = A2C(state_dim, action_dim, config, use_gae=True)
    _, rewards_a2c = train_policy_gradient(
        agent_a2c, env_name, num_episodes, log_interval=100, seed=seed
    )
    results["A2C (GAE)"] = rewards_a2c

    return results


def plot_training_curves(
    results: Dict[str, List[float]],
    window_size: int = 50,
    save_path: Optional[str] = None
) -> None:
    """
    绘制训练曲线

    Parameters
    ----------
    results : Dict[str, List[float]]
        各算法的奖励历史
    window_size : int
        滑动平均窗口大小
    save_path : str, optional
        图片保存路径
    """
    if not HAS_MATPLOTLIB:
        print("需要安装matplotlib: pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, (name, rewards) in enumerate(results.items()):
        color = colors[idx % len(colors)]

        # 原始曲线（透明）
        ax.plot(rewards, alpha=0.2, color=color)

        # 滑动平均
        if len(rewards) > window_size:
            smoothed = np.convolve(
                rewards,
                np.ones(window_size) / window_size,
                mode="valid"
            )
            ax.plot(
                range(window_size - 1, len(rewards)),
                smoothed,
                label=name,
                color=color,
                linewidth=2
            )
        else:
            ax.plot(rewards, label=name, color=color, linewidth=2)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Total Reward", fontsize=12)
    ax.set_title("Policy Gradient Methods Comparison", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"图表已保存至: {save_path}")

    plt.show()


# ============================================================================
#                           单元测试
# ============================================================================

def run_tests() -> bool:
    """
    运行单元测试

    测试所有核心组件的功能正确性。

    Returns
    -------
    bool
        所有测试是否通过
    """
    print("\n" + "=" * 60)
    print("运行单元测试")
    print("=" * 60 + "\n")

    all_passed = True

    # 测试1: MLP网络
    print("[测试1] MLP网络构建...")
    try:
        mlp = MLP(4, 2, [128, 128])
        x = torch.randn(32, 4)
        out = mlp(x)
        assert out.shape == (32, 2), f"输出形状错误: {out.shape}"
        print("  [通过] MLP前向传播正常")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试2: 离散策略网络
    print("[测试2] 离散策略网络...")
    try:
        policy = DiscretePolicy(4, 2)
        state = torch.randn(1, 4)
        action, log_prob, entropy = policy.sample(state)
        assert action.shape == (1,), f"动作形状错误: {action.shape}"
        assert log_prob.shape == (1,), f"log_prob形状错误: {log_prob.shape}"
        assert entropy.shape == (1,), f"熵形状错误: {entropy.shape}"

        # 检查概率和为1
        dist = policy.get_distribution(state)
        probs = dist.probs
        assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
        print("  [通过] 离散策略采样正常")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试3: 连续策略网络
    print("[测试3] 连续策略网络...")
    try:
        policy = ContinuousPolicy(4, 2)
        state = torch.randn(1, 4)
        action, log_prob, entropy = policy.sample(state)
        assert action.shape == (1, 2), f"动作形状错误: {action.shape}"
        assert torch.all(action >= -1) and torch.all(action <= 1), "动作未在[-1,1]范围内"
        print("  [通过] 连续策略采样正常，动作范围正确")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试4: 价值网络
    print("[测试4] 价值网络...")
    try:
        value_net = ValueNetwork(4)
        state = torch.randn(32, 4)
        value = value_net(state)
        assert value.shape == (32, 1), f"价值形状错误: {value.shape}"
        print("  [通过] 价值网络前向传播正常")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试5: 回报计算
    print("[测试5] 蒙特卡洛回报计算...")
    try:
        rewards = [1.0, 1.0, 1.0]
        gamma = 0.99
        returns = compute_returns(rewards, gamma, normalize=False)

        # 验证：G_2=1, G_1=1+0.99*1=1.99, G_0=1+0.99*1.99=2.9701
        expected = torch.tensor([2.9701, 1.99, 1.0])
        assert torch.allclose(returns, expected, atol=1e-4), f"回报计算错误: {returns}"
        print("  [通过] 蒙特卡洛回报计算正确")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试6: GAE计算
    print("[测试6] GAE优势估计...")
    try:
        rewards = [1.0, 1.0, 1.0]
        values = [torch.tensor(0.5), torch.tensor(0.5), torch.tensor(0.5)]
        dones = [False, False, True]
        next_value = 0.0
        gamma = 0.99
        gae_lambda = 0.95

        advantages, returns = compute_gae(
            rewards, values, next_value, dones, gamma, gae_lambda
        )

        assert len(advantages) == 3, f"优势长度错误: {len(advantages)}"
        assert len(returns) == 3, f"回报长度错误: {len(returns)}"
        print("  [通过] GAE计算正常")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试7: Actor-Critic网络
    print("[测试7] Actor-Critic网络...")
    try:
        ac_net = ActorCriticNetwork(4, 2, hidden_dim=128, continuous=False)
        state = torch.randn(1, 4)
        action, log_prob, entropy, value = ac_net.get_action_and_value(state)

        assert action.shape == (1,), f"动作形状错误: {action.shape}"
        assert value.shape == (1,), f"价值形状错误: {value.shape}"
        print("  [通过] Actor-Critic网络正常")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试8: REINFORCE算法
    print("[测试8] REINFORCE算法...")
    try:
        config = TrainingConfig(gamma=0.99, lr_actor=1e-3)
        agent = REINFORCE(4, 2, config)

        # 模拟一个回合
        buffer = EpisodeBuffer()
        for _ in range(10):
            state = np.random.randn(4)
            action, info = agent.select_action(state)
            buffer.store(
                state=state,
                action=action,
                reward=1.0,
                log_prob=info["log_prob"],
                entropy=info["entropy"]
            )

        # 更新
        loss_info = agent.update(buffer)
        assert "policy_loss" in loss_info, "缺少policy_loss"
        print("  [通过] REINFORCE更新正常")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试9: A2C算法
    print("[测试9] A2C算法...")
    try:
        config = TrainingConfig(gamma=0.99, lr_actor=1e-3)
        agent = A2C(4, 2, config, use_gae=True, shared_network=True)

        buffer = EpisodeBuffer()
        state = np.random.randn(4)
        for i in range(10):
            action, info = agent.select_action(state)
            next_state = np.random.randn(4)
            buffer.store(
                state=state,
                action=action,
                reward=1.0,
                log_prob=info["log_prob"],
                value=info["value"],
                done=(i == 9),
                entropy=info["entropy"]
            )
            state = next_state

        loss_info = agent.update(buffer, next_state, done=True)
        assert "policy_loss" in loss_info, "缺少policy_loss"
        assert "value_loss" in loss_info, "缺少value_loss"
        print("  [通过] A2C更新正常")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试10: EpisodeBuffer
    print("[测试10] EpisodeBuffer...")
    try:
        buffer = EpisodeBuffer()
        for i in range(5):
            buffer.store(
                state=np.zeros(4),
                action=0,
                reward=1.0,
                log_prob=torch.tensor(0.0)
            )

        assert len(buffer) == 5, f"缓冲区长度错误: {len(buffer)}"
        assert buffer.total_reward == 5.0, f"总奖励错误: {buffer.total_reward}"

        buffer.clear()
        assert len(buffer) == 0, "清空失败"
        print("  [通过] EpisodeBuffer功能正常")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 总结
    print("\n" + "=" * 60)
    if all_passed:
        print("所有测试通过！")
    else:
        print("部分测试失败，请检查。")
    print("=" * 60 + "\n")

    return all_passed


# ============================================================================
#                           主程序
# ============================================================================

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="策略梯度方法实现与训练"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="test",
        choices=["test", "train", "compare"],
        help="运行模式: test(测试), train(训练), compare(比较算法)"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="CartPole-v1",
        help="Gymnasium环境名称"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=300,
        help="训练回合数"
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="a2c",
        choices=["reinforce", "reinforce_baseline", "a2c"],
        help="算法选择"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )

    args = parser.parse_args()

    if args.mode == "test":
        # 运行单元测试
        run_tests()

    elif args.mode == "train":
        # 单独训练
        if not HAS_GYMNASIUM:
            print("需要安装gymnasium: pip install gymnasium")
            return

        env = gym.make(args.env)
        state_dim = env.observation_space.shape[0]

        if isinstance(env.action_space, spaces.Discrete):
            action_dim = env.action_space.n
            continuous = False
        else:
            action_dim = env.action_space.shape[0]
            continuous = True
        env.close()

        config = TrainingConfig(
            gamma=0.99,
            lr_actor=1e-3,
            lr_critic=1e-3,
            entropy_coef=0.01
        )

        if args.algo == "reinforce":
            agent = REINFORCE(state_dim, action_dim, config, continuous)
        elif args.algo == "reinforce_baseline":
            agent = REINFORCEBaseline(state_dim, action_dim, config, continuous)
        else:
            agent = A2C(state_dim, action_dim, config, continuous, use_gae=True)

        agent, rewards = train_policy_gradient(
            agent,
            args.env,
            args.episodes,
            log_interval=50,
            seed=args.seed
        )

        # 绘制学习曲线
        if HAS_MATPLOTLIB:
            plot_training_curves(
                {args.algo.upper(): rewards},
                save_path=f"{args.algo}_learning_curve.png"
            )

    elif args.mode == "compare":
        # 比较不同算法
        results = compare_algorithms(
            args.env,
            args.episodes,
            args.seed
        )

        # 绘制比较图
        if HAS_MATPLOTLIB:
            plot_training_curves(
                results,
                save_path="policy_gradient_comparison.png"
            )


if __name__ == "__main__":
    main()
