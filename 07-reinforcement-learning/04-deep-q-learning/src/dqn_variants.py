#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Q-Network Variants: Production-Grade Implementation
=========================================================

A comprehensive, production-ready implementation of Deep Q-Network variants
for deep reinforcement learning research and deployment.

============================================================
Core Idea (核心思想)
============================================================
Deep Q-Network (DQN) 通过神经网络近似Q函数实现了端到端的强化学习。然而，
原始DQN存在过估计偏差、样本效率低和探索能力弱等问题。本模块实现了一系列
算法改进，每个变体针对特定的失效模式进行优化：

- **Double DQN**: 解耦动作选择与评估，消除过估计偏差
- **Dueling DQN**: 分离状态价值与动作优势，提升泛化能力
- **Noisy Networks**: 参数化噪声实现状态依赖的探索策略
- **Categorical DQN (C51)**: 建模完整回报分布，捕获不确定性
- **Prioritized Replay**: 优先采样高TD误差样本，提升样本效率
- **Rainbow**: 组合所有改进，达到最优性能

============================================================
Mathematical Foundation (数学基础)
============================================================

1. Double DQN (van Hasselt et al., 2016)
   ------------------------------------
   **Problem**: Standard DQN uses max operator for both action selection
   and evaluation, causing systematic overestimation:

   Standard TD Target:

   .. math::
       y = r + \\gamma \\max_{a'} Q(s', a'; \\theta^-)

   Overestimation occurs because:

   .. math::
       \\mathbb{E}[\\max_{a'} Q(s', a')] \\geq \\max_{a'} \\mathbb{E}[Q(s', a')]

   **Solution**: Decouple selection (online network θ) from evaluation (target θ⁻):

   .. math::
       y^{\\text{Double}} = r + \\gamma Q\\left(s', \\underbrace{\\arg\\max_{a'} Q(s', a'; \\theta)}_{\\text{online selects}}; \\theta^-\\right)

   The online network selects the best action, but the target network evaluates it.
   This prevents the same estimation noise from inflating both selection and evaluation.

2. Dueling DQN (Wang et al., 2016)
   --------------------------------
   **Key Insight**: Decompose Q-function into state value V(s) and advantage A(s,a):

   .. math::
       Q(s, a) = V(s) + A(s, a) - \\frac{1}{|\\mathcal{A}|} \\sum_{a'} A(s, a')

   where:

   - V(s) ∈ ℝ: State value - "how good is being in this state?"
   - A(s, a) ∈ ℝ^|A|: Advantage - "how much better is action a than average?"

   **Identifiability**: Mean subtraction ensures unique decomposition:

   .. math::
       \\sum_{a} A(s, a) = 0 \\quad \\Rightarrow \\quad V(s) = \\max_a Q(s, a)

   **Benefit**: Enables learning about state values without needing to evaluate
   every action, improving generalization in states where action choice matters less.

3. Noisy Networks (Fortunato et al., 2017)
   ----------------------------------------
   **Motivation**: Replace static ε-greedy with learned, state-dependent exploration.

   Noisy Linear Layer:

   .. math::
       y = (\\underbrace{\\mu^w}_{\\text{mean}} + \\underbrace{\\sigma^w \\odot \\varepsilon^w}_{\\text{noise}}) x + (\\mu^b + \\sigma^b \\odot \\varepsilon^b)

   **Factorized Gaussian Noise** (efficient parameterization):

   .. math::
       \\varepsilon_{ij} = f(\\varepsilon_i) \\cdot f(\\varepsilon_j), \\quad f(x) = \\text{sign}(x)\\sqrt{|x|}

   Reduces noise parameters from O(pq) to O(p+q) for (p×q) weight matrix.

   **Key Properties**:
   - σ parameters are learned end-to-end via backpropagation
   - Network automatically reduces exploration as it becomes more certain
   - Exploration is inherently state-dependent

4. Categorical DQN / C51 (Bellemare et al., 2017)
   -----------------------------------------------
   **Paradigm Shift**: Model full return distribution, not just expected value.

   Distribution Representation:

   .. math::
       Z(s, a) \\sim \\text{Categorical}(z_1, ..., z_N; p_1(s,a), ..., p_N(s,a))

   **Support Atoms**: Fixed discretization of value range:

   .. math::
       z_i = V_{\\min} + i \\cdot \\Delta z, \\quad \\Delta z = \\frac{V_{\\max} - V_{\\min}}{N - 1}

   **Distributional Bellman Equation**:

   .. math::
       \\mathcal{T} Z(s, a) \\stackrel{D}{=} R + \\gamma Z(S', A')

   **Categorical Projection**: Project shifted distribution back onto support:

   .. math::
       (\\Phi \\mathcal{T} Z)_i = \\sum_j \\left[1 - \\frac{|[\\mathcal{T}z_j]_{V_{\\min}}^{V_{\\max}} - z_i|}{\\Delta z}\\right]_0^1 p_j

   **KL Divergence Loss**:

   .. math::
       L = D_{KL}(\\Phi \\mathcal{T} Z(s, a) \\| Z(s, a; \\theta))

5. Prioritized Experience Replay (Schaul et al., 2016)
   ----------------------------------------------------
   **Core Idea**: Sample important transitions more frequently.

   **Priority Definition** (proportional to TD error):

   .. math::
       p_i = |\\delta_i| + \\epsilon, \\quad \\delta_i = r + \\gamma \\max_{a'} Q(s', a') - Q(s, a)

   **Sampling Probability**:

   .. math::
       P(i) = \\frac{p_i^\\alpha}{\\sum_k p_k^\\alpha}

   - α = 0: Uniform sampling
   - α = 1: Full prioritization

   **Importance Sampling Correction** (unbias gradients):

   .. math::
       w_i = \\left( \\frac{1}{N \\cdot P(i)} \\right)^\\beta / \\max_j w_j

   β anneals from β₀ to 1 over training to fully correct bias at convergence.

6. Multi-step Learning (Sutton, 1988)
   -----------------------------------
   **N-step Returns**: Bootstrap after n steps instead of 1:

   .. math::
       G_t^{(n)} = \\sum_{k=0}^{n-1} \\gamma^k R_{t+k+1} + \\gamma^n \\max_{a'} Q(S_{t+n}, a'; \\theta^-)

   **Bias-Variance Trade-off**:

   - n = 1 (TD): Low variance, high bias (bootstrap early)
   - n = ∞ (MC): High variance, zero bias (no bootstrapping)
   - n = 3-5: Sweet spot for many tasks

7. Rainbow (Hessel et al., 2018)
   -----------------------------
   **Integration**: Combines all improvements synergistically:

   Double + Dueling + Noisy + Categorical + PER + Multi-step

   Each component addresses orthogonal issues, and their combination
   achieves state-of-the-art on Atari (441% human-normalized median score).

============================================================
Problem Statement (问题定义)
============================================================

Vanilla DQN suffers from several fundamental limitations:

1. **Overestimation Bias** (过估计偏差):
   - Max operator in TD target systematically overestimates Q-values
   - E[max Q] ≥ max E[Q] by Jensen's inequality
   - Errors accumulate through bootstrapping, causing divergence
   - Leads to suboptimal policies that favor overestimated actions

2. **Sample Inefficiency** (样本效率低):
   - Uniform replay treats all transitions equally
   - Rare but informative experiences (e.g., rewards, failures) undersampled
   - Redundant sampling of frequent transitions
   - Slow learning in sparse reward environments

3. **Exploration Challenges** (探索困难):
   - ε-greedy is state-independent and inefficient
   - Same exploration probability regardless of uncertainty
   - No intrinsic motivation to visit uncertain states
   - Struggles with deep exploration problems

4. **Scalar Value Limitation** (标量值局限):
   - Expected value ignores return distribution shape
   - Risk-neutral decision making (ignores variance)
   - Loses information about outcome uncertainty
   - Cannot distinguish high-variance from low-variance returns

============================================================
Algorithm Comparison (算法对比)
============================================================

+------------------+---------------+---------------+----------------+----------------+
| Variant          | Overestimation| Sample Eff.   | Exploration    | Computation    |
+==================+===============+===============+================+================+
| Vanilla DQN      | High          | Low           | ε-greedy       | Base           |
+------------------+---------------+---------------+----------------+----------------+
| Double DQN       | Reduced       | Low           | ε-greedy       | +~0%           |
+------------------+---------------+---------------+----------------+----------------+
| Dueling DQN      | Medium        | Medium        | ε-greedy       | +~20%          |
+------------------+---------------+---------------+----------------+----------------+
| Noisy DQN        | High          | Low           | Learned        | +~50%          |
+------------------+---------------+---------------+----------------+----------------+
| Categorical DQN  | Low           | High          | ε-greedy       | +~100%         |
+------------------+---------------+---------------+----------------+----------------+
| Rainbow          | Lowest        | Highest       | Learned        | +~200%         |
+------------------+---------------+---------------+----------------+----------------+

**Performance on Atari** (median human-normalized score):

- DQN: 79%
- Double DQN: 117%
- Dueling DQN: 151%
- Prioritized DQN: 141%
- Categorical DQN: 235%
- Rainbow: 441%

============================================================
Complexity Analysis (复杂度分析)
============================================================

Let: d = state_dim, h = hidden_dim, A = action_dim, N = atoms (C51)

**Space Complexity** (参数量):

+------------------+-------------------------------------+-------------+
| Variant          | Parameters                          | Multiplier  |
+==================+=====================================+=============+
| Vanilla DQN      | O(d·h + h² + h·A)                   | 1x          |
+------------------+-------------------------------------+-------------+
| Double DQN       | O(d·h + h² + h·A)                   | 1x          |
+------------------+-------------------------------------+-------------+
| Dueling DQN      | O(d·h + 2h² + h·A + h)              | ~1.5x       |
+------------------+-------------------------------------+-------------+
| Noisy DQN        | O(2(d·h + h² + h·A))                | 2x (μ + σ)  |
+------------------+-------------------------------------+-------------+
| Categorical DQN  | O(d·h + h² + h·A·N)                 | N atoms     |
+------------------+-------------------------------------+-------------+
| Rainbow          | O(2(d·h + 2h² + h·A·N))             | combines    |
+------------------+-------------------------------------+-------------+

**Time Complexity** (per update step):

+----------------------+------------------+---------------------------+
| Operation            | Complexity       | Notes                     |
+======================+==================+===========================+
| Forward pass         | O(B · |θ|)       | B = batch size            |
+----------------------+------------------+---------------------------+
| Backward pass        | O(B · |θ|)       | Same as forward           |
+----------------------+------------------+---------------------------+
| Categorical project  | O(B · A · N)     | Distribution projection   |
+----------------------+------------------+---------------------------+
| PER sampling         | O(B log M)       | M = buffer size           |
+----------------------+------------------+---------------------------+
| Uniform sampling     | O(B)             | Random access             |
+----------------------+------------------+---------------------------+

============================================================
Summary (总结)
============================================================

Each DQN variant addresses specific limitations:

+------------------+--------------------------------------------------+
| Variant          | Key Contribution                                 |
+==================+==================================================+
| Double DQN       | Fixes overestimation via decoupled evaluation    |
+------------------+--------------------------------------------------+
| Dueling DQN      | Better generalization via V/A decomposition      |
+------------------+--------------------------------------------------+
| Noisy DQN        | Learned, state-dependent exploration             |
+------------------+--------------------------------------------------+
| Categorical DQN  | Full return distribution modeling                |
+------------------+--------------------------------------------------+
| PER              | Efficient sampling of informative transitions    |
+------------------+--------------------------------------------------+
| Multi-step       | Faster credit assignment                         |
+------------------+--------------------------------------------------+

**Rainbow** combines all innovations synergistically, achieving SOTA results.

============================================================
Implementation Features (实现特点)
============================================================

This implementation provides:

1. **Modularity**: Each component (buffer, network, agent) is independent
2. **Type Safety**: Full type annotations for static analysis
3. **Validation**: Input validation with descriptive error messages
4. **Efficiency**: Vectorized operations, __slots__ for memory
5. **Reproducibility**: Deterministic seeding support
6. **Persistence**: Save/load model checkpoints
7. **Testing**: Comprehensive unit tests with mock data

============================================================
References
============================================================

[1] Mnih, V. et al. (2015). Human-level control through deep reinforcement
    learning. Nature, 518(7540):529-533.

[2] van Hasselt, H. et al. (2016). Deep Reinforcement Learning with Double
    Q-learning. AAAI.

[3] Wang, Z. et al. (2016). Dueling Network Architectures for Deep
    Reinforcement Learning. ICML.

[4] Fortunato, M. et al. (2017). Noisy Networks for Exploration. ICLR.

[5] Bellemare, M. et al. (2017). A Distributional Perspective on
    Reinforcement Learning. ICML.

[6] Schaul, T. et al. (2016). Prioritized Experience Replay. ICLR.

[7] Hessel, M. et al. (2018). Rainbow: Combining Improvements in Deep
    Reinforcement Learning. AAAI.

============================================================
Dependencies
============================================================

.. code-block:: bash

    pip install torch numpy gymnasium matplotlib

============================================================
Usage Examples
============================================================

Basic training:

.. code-block:: python

    config = DQNVariantConfig(state_dim=4, action_dim=2)
    agent = DQNVariantAgent(config, DQNVariant.RAINBOW)
    rewards, _ = train_agent(agent, env_name="CartPole-v1")

Compare variants:

.. code-block:: python

    results = compare_variants(env_name="CartPole-v1", num_episodes=300)
    plot_comparison(results)

Author: zimingttkx
License: MIT
Version: 2.0.0
"""

from __future__ import annotations

import math
import os
import random
import warnings
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Deque,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from numpy.typing import NDArray
from torch import Tensor

# ============================================================
# Optional Dependencies
# ============================================================

try:
    import gymnasium as gym
    from gymnasium import Env
    HAS_GYM = True
except ImportError:
    gym = None  # type: ignore[assignment]
    Env = None  # type: ignore[assignment, misc]
    HAS_GYM = False
    warnings.warn(
        "gymnasium not installed. Environment interaction unavailable. "
        "Install with: pip install gymnasium",
        ImportWarning,
        stacklevel=2,
    )

try:
    import matplotlib.pyplot as plt
    from matplotlib import cm
    HAS_MATPLOTLIB = True
except ImportError:
    plt = None  # type: ignore[assignment]
    cm = None  # type: ignore[assignment]
    HAS_MATPLOTLIB = False


# ============================================================
# Type Definitions (类型定义)
# ============================================================

# NumPy array type aliases for clarity and static type checking
FloatArray = NDArray[np.floating[Any]]
"""Float-valued NumPy array for states, rewards, etc."""

IntArray = NDArray[np.int64]
"""Integer-valued NumPy array for actions, indices, etc."""


class Transition(NamedTuple):
    """
    Single-step transition tuple for experience replay.

    Core Idea (核心思想)
    --------------------
    表示MDP中单步交互的基本数据单元：τ = (s_t, a_t, r_t, s_{t+1}, done_t)。
    使用NamedTuple实现内存高效且不可变的数据存储。

    Mathematical Definition (数学定义)
    ----------------------------------
    A transition represents one step of the MDP:

    .. math::
        \\tau_t = (s_t, a_t, r_t, s_{t+1}, d_t)

    where:

    - s_t ∈ S: Current state
    - a_t ∈ A: Action taken
    - r_t ∈ ℝ: Immediate reward r(s_t, a_t)
    - s_{t+1} ∈ S: Next state from transition P(·|s_t, a_t)
    - d_t ∈ {0, 1}: Terminal flag

    Attributes
    ----------
    state : FloatArray
        Current observation s_t ∈ ℝ^d, shape (state_dim,)
    action : int
        Discrete action index a_t ∈ {0, ..., |A|-1}
    reward : float
        Scalar immediate reward r_t
    next_state : FloatArray
        Next observation s_{t+1} ∈ ℝ^d, shape (state_dim,)
    done : bool
        Episode termination flag (True if terminal state)

    Notes
    -----
    - NamedTuple provides immutability and memory efficiency
    - Fields are stored contiguously for cache-friendly access
    - Used as the atomic unit for all replay buffer implementations
    """
    state: FloatArray
    action: int
    reward: float
    next_state: FloatArray
    done: bool


class NStepTransition(NamedTuple):
    """
    N-step transition for multi-step learning.

    Core Idea (核心思想)
    --------------------
    存储n步累积回报用于多步学习。相比单步TD，n步方法在偏差与方差之间
    提供了更灵活的权衡，通常可以加速学习。

    Mathematical Definition (数学定义)
    ----------------------------------
    N-step return from time t:

    .. math::
        G_t^{(n)} = \\sum_{k=0}^{n-1} \\gamma^k R_{t+k+1} + \\gamma^n V(S_{t+n})

    The transition stores:

    - Initial state s_t and action a_t
    - Cumulative discounted reward: Σ_{k=0}^{n-1} γ^k r_{t+k+1}
    - Bootstrap state s_{t+n} for value estimation

    Attributes
    ----------
    state : FloatArray
        Initial observation s_t ∈ ℝ^d at start of n-step sequence
    action : int
        Action a_t taken at initial state
    n_step_return : float
        Cumulative n-step discounted return: Σ_{k=0}^{n-1} γ^k r_{t+k+1}
    next_state : FloatArray
        State s_{t+n} for bootstrapping (n steps ahead)
    done : bool
        True if episode terminated within n steps
    n_steps : int
        Actual number of steps (may be < n if episode ended early)

    Notes
    -----
    - n_steps can be less than configured n when episode terminates early
    - When done=True, no bootstrapping is needed (terminal state)
    - Provides faster credit assignment compared to 1-step TD

    See Also
    --------
    NStepReplayBuffer : Buffer implementation for n-step transitions
    """
    state: FloatArray
    action: int
    n_step_return: float
    next_state: FloatArray
    done: bool
    n_steps: int


# ============================================================
# Replay Buffers (经验回放缓冲区)
# ============================================================


class ReplayBuffer:
    """
    Uniform experience replay buffer with O(1) operations.

    Core Idea (核心思想)
    --------------------
    经验回放是DQN的核心创新之一，解决了两个关键问题：

    1. **打破时序相关性**: 连续采样的样本高度相关，违反SGD的i.i.d.假设。
       随机采样打破时序关联，提供更稳定的梯度估计。

    2. **提高数据效率**: 每个经验可被多次使用，而非"用后即弃"。
       这对于数据收集成本高的RL场景尤为重要。

    Mathematical Foundation (数学基础)
    ----------------------------------
    Uniform sampling probability:

    .. math::
        P(i) = \\frac{1}{|\\mathcal{D}|}, \\quad \\forall i \\in \\mathcal{D}

    Expected gradient with replay buffer:

    .. math::
        \\nabla_\\theta L = \\mathbb{E}_{(s,a,r,s') \\sim \\mathcal{U}(\\mathcal{D})}
        \\left[ (Q(s,a;\\theta) - y)^2 \\right]

    Implementation Details (实现细节)
    ---------------------------------
    - Uses ``collections.deque`` with ``maxlen`` for automatic FIFO eviction
    - ``__slots__`` reduces memory footprint per instance
    - Vectorized NumPy operations for batch sampling

    Complexity Analysis (复杂度分析)
    --------------------------------
    +------------+------------+----------------------------------+
    | Operation  | Complexity | Notes                            |
    +============+============+==================================+
    | push()     | O(1)*      | Amortized; may trigger eviction  |
    +------------+------------+----------------------------------+
    | sample()   | O(B)       | B = batch_size                   |
    +------------+------------+----------------------------------+
    | __len__()  | O(1)       | Cached by deque                  |
    +------------+------------+----------------------------------+

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store. Must be positive.
        When full, oldest transitions are evicted (FIFO policy).

    Attributes
    ----------
    capacity : int
        Maximum buffer size (read-only property)

    Raises
    ------
    ValueError
        If ``capacity <= 0``

    Examples
    --------
    >>> buffer = ReplayBuffer(capacity=10000)
    >>> buffer.push(state, action, reward, next_state, done)
    >>> if buffer.is_ready(min_size=1000):
    ...     states, actions, rewards, next_states, dones = buffer.sample(64)

    See Also
    --------
    PrioritizedReplayBuffer : Priority-based sampling for improved efficiency
    NStepReplayBuffer : Multi-step returns for faster credit assignment
    """

    __slots__ = ("_capacity", "_buffer")

    def __init__(self, capacity: int) -> None:
        """
        Initialize uniform replay buffer.

        Parameters
        ----------
        capacity : int
            Maximum number of transitions to store. Must be positive.

        Raises
        ------
        ValueError
            If capacity is not a positive integer.
        """
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError(
                f"capacity must be a positive integer, got {capacity!r} "
                f"(type: {type(capacity).__name__})"
            )
        self._capacity = capacity
        self._buffer: Deque[Transition] = deque(maxlen=capacity)

    @property
    def capacity(self) -> int:
        """Maximum buffer capacity (read-only)."""
        return self._capacity

    def __len__(self) -> int:
        """Return current number of stored transitions."""
        return len(self._buffer)

    def push(
        self,
        state: FloatArray,
        action: int,
        reward: float,
        next_state: FloatArray,
        done: bool,
    ) -> None:
        """
        Store a single transition in the buffer.

        Parameters
        ----------
        state : FloatArray
            Current state observation, shape (state_dim,)
        action : int
            Discrete action index
        reward : float
            Immediate reward received
        next_state : FloatArray
            Next state observation, shape (state_dim,)
        done : bool
            Whether episode terminated after this transition

        Notes
        -----
        - O(1) amortized time complexity
        - Automatically evicts oldest transition when at capacity (FIFO)
        """
        self._buffer.append(Transition(state, action, reward, next_state, done))

    def sample(
        self,
        batch_size: int,
    ) -> Tuple[FloatArray, IntArray, FloatArray, FloatArray, FloatArray]:
        """
        Sample a uniform random mini-batch of transitions.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample

        Returns
        -------
        states : FloatArray
            Batch of states, shape (batch_size, state_dim)
        actions : IntArray
            Batch of actions, shape (batch_size,)
        rewards : FloatArray
            Batch of rewards, shape (batch_size,)
        next_states : FloatArray
            Batch of next states, shape (batch_size, state_dim)
        dones : FloatArray
            Batch of done flags (0.0 or 1.0), shape (batch_size,)

        Raises
        ------
        ValueError
            If batch_size exceeds current buffer size

        Notes
        -----
        - Samples without replacement within each call
        - Returns NumPy arrays for efficient tensor conversion
        """
        if batch_size > len(self._buffer):
            raise ValueError(
                f"batch_size ({batch_size}) exceeds buffer size ({len(self._buffer)}). "
                f"Ensure buffer has at least {batch_size} transitions before sampling."
            )

        batch = random.sample(list(self._buffer), batch_size)

        states = np.array([t.state for t in batch], dtype=np.float32)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        batch_rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.array([t.next_state for t in batch], dtype=np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)

        return states, actions, batch_rewards, next_states, dones

    def is_ready(self, min_size: int) -> bool:
        """
        Check if buffer has sufficient samples for training.

        Parameters
        ----------
        min_size : int
            Minimum required number of transitions

        Returns
        -------
        bool
            True if len(buffer) >= min_size
        """
        return len(self._buffer) >= min_size


class SumTree:
    """
    Binary sum tree for O(log N) prioritized sampling.

    Core Idea (核心思想)
    --------------------
    SumTree是一种特殊的二叉树数据结构，用于高效实现优先经验回放(PER)。
    其核心思想是：

    1. **叶节点**: 存储每个样本的优先级值
    2. **内部节点**: 存储子节点优先级之和
    3. **根节点**: 存储所有优先级的总和

    Mathematical Foundation (数学基础)
    ----------------------------------
    Tree structure for N leaves:

    - Total nodes: 2N - 1
    - Leaf indices: [N-1, 2N-2]
    - Parent of node i: (i - 1) // 2
    - Children of node i: 2i + 1 (left), 2i + 2 (right)

    Proportional sampling uses cumulative sum:

    .. math::
        \\text{sample}(u) \\to i \\text{ such that } \\sum_{j<i} p_j < u \\leq \\sum_{j \\leq i} p_j

    where u ~ Uniform(0, total_priority).

    Complexity Analysis (复杂度分析)
    --------------------------------
    +------------------+------------+----------------------------------+
    | Operation        | Complexity | Notes                            |
    +==================+============+==================================+
    | add()            | O(log N)   | Insert + update ancestors        |
    +------------------+------------+----------------------------------+
    | update_priority()| O(log N)   | Update leaf + propagate          |
    +------------------+------------+----------------------------------+
    | get()            | O(log N)   | Binary search by cumsum          |
    +------------------+------------+----------------------------------+
    | total_priority   | O(1)       | Root node value                  |
    +------------------+------------+----------------------------------+

    Parameters
    ----------
    capacity : int
        Maximum number of elements (leaf nodes)

    Attributes
    ----------
    total_priority : float
        Sum of all priorities (root node value)
    capacity : int
        Maximum number of elements

    Notes
    -----
    - Array-based implementation avoids pointer overhead
    - Priority updates automatically propagate to root
    - Enables efficient stratified sampling for PER
    """

    __slots__ = ("_capacity", "_tree", "_data", "_write_idx", "_size")

    def __init__(self, capacity: int) -> None:
        """
        Initialize sum tree with given capacity.

        Parameters
        ----------
        capacity : int
            Maximum number of leaf nodes (data elements)
        """
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError(f"capacity must be a positive integer, got {capacity!r}")

        self._capacity = capacity
        # Tree array: internal nodes [0, capacity-2], leaves [capacity-1, 2*capacity-2]
        self._tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self._data: List[Optional[Any]] = [None] * capacity
        self._write_idx = 0
        self._size = 0

    @property
    def total_priority(self) -> float:
        """Total priority sum (root node value). O(1)."""
        return float(self._tree[0])

    @property
    def capacity(self) -> int:
        """Maximum number of elements."""
        return self._capacity

    def __len__(self) -> int:
        """Current number of stored elements."""
        return self._size

    def add(self, priority: float, data: Any) -> None:
        """
        Add element with specified priority.

        Parameters
        ----------
        priority : float
            Priority value (must be non-negative)
        data : Any
            Data to store (typically a Transition)

        Notes
        -----
        - O(log N) due to priority propagation
        - Overwrites oldest element when at capacity (FIFO)
        """
        if priority < 0:
            raise ValueError(f"priority must be non-negative, got {priority}")

        tree_idx = self._write_idx + self._capacity - 1
        self._data[self._write_idx] = data
        self._update(tree_idx, priority)
        self._write_idx = (self._write_idx + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def _update(self, tree_idx: int, priority: float) -> None:
        """Update priority at tree_idx and propagate delta to root."""
        delta = priority - self._tree[tree_idx]
        self._tree[tree_idx] = priority
        # Propagate change up to root
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self._tree[tree_idx] += delta

    def update_priority(self, tree_idx: int, priority: float) -> None:
        """
        Update priority of existing element.

        Parameters
        ----------
        tree_idx : int
            Tree index of leaf node (returned by get())
        priority : float
            New priority value
        """
        if priority < 0:
            raise ValueError(f"priority must be non-negative, got {priority}")
        self._update(tree_idx, priority)

    def get(self, cumsum: float) -> Tuple[int, float, Any]:
        """
        Sample element by cumulative sum (proportional sampling).

        Parameters
        ----------
        cumsum : float
            Target cumulative sum in [0, total_priority)

        Returns
        -------
        tree_idx : int
            Index in tree array (for priority updates)
        priority : float
            Priority of sampled element
        data : Any
            Stored data element

        Notes
        -----
        O(log N) binary search from root to leaf.
        """
        parent = 0
        while True:
            left = 2 * parent + 1
            right = left + 1
            if left >= len(self._tree):
                # Reached leaf
                break
            if cumsum <= self._tree[left]:
                parent = left
            else:
                cumsum -= self._tree[left]
                parent = right

        data_idx = parent - self._capacity + 1
        return parent, float(self._tree[parent]), self._data[data_idx]

    def min_priority(self) -> float:
        """
        Get minimum non-zero priority among stored elements.

        Returns
        -------
        float
            Minimum priority, or 0.0 if empty
        """
        if self._size == 0:
            return 0.0
        start = self._capacity - 1
        priorities = self._tree[start:start + self._size]
        positive_priorities = priorities[priorities > 0]
        if len(positive_priorities) == 0:
            return 0.0
        return float(np.min(positive_priorities))


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER) buffer.

    Core Idea (核心思想)
    --------------------
    优先经验回放通过TD误差大小对样本进行优先采样，使得学习更加高效：

    1. **重要性采样**: 高TD误差的样本包含更多学习信号
    2. **偏差校正**: 使用重要性采样权重保持梯度无偏

    Mathematical Foundation (数学基础)
    ----------------------------------
    **Priority Definition**:

    .. math::
        p_i = |\\delta_i| + \\epsilon

    where δ_i is TD error and ε prevents zero priority.

    **Sampling Probability**:

    .. math::
        P(i) = \\frac{p_i^\\alpha}{\\sum_k p_k^\\alpha}

    - α = 0: Uniform sampling (ignores priorities)
    - α = 1: Full prioritization

    **Importance Sampling Weights** (unbias gradients):

    .. math::
        w_i = \\left( \\frac{1}{N \\cdot P(i)} \\right)^\\beta / \\max_j w_j

    β anneals from β₀ to 1 over training to fully correct bias.

    Implementation Details (实现细节)
    ---------------------------------
    - Uses SumTree for O(log N) proportional sampling
    - Stratified sampling: divides priority range into equal segments
    - New transitions get max priority for guaranteed initial sampling

    Complexity Analysis (复杂度分析)
    --------------------------------
    +------------------+------------+----------------------------------+
    | Operation        | Complexity | Notes                            |
    +==================+============+==================================+
    | push()           | O(log N)   | Add to sum tree                  |
    +------------------+------------+----------------------------------+
    | sample()         | O(B log N) | B stratified samples             |
    +------------------+------------+----------------------------------+
    | update_priorities| O(B log N) | Update B priorities              |
    +------------------+------------+----------------------------------+

    Parameters
    ----------
    capacity : int
        Maximum buffer size
    alpha : float, default=0.6
        Prioritization exponent. 0 = uniform, 1 = full priority.
    beta_start : float, default=0.4
        Initial importance sampling exponent
    beta_frames : int, default=100000
        Frames over which to anneal β from beta_start to 1.0
    epsilon : float, default=1e-6
        Small constant preventing zero priority

    Attributes
    ----------
    capacity : int
        Maximum buffer size
    beta : float
        Current importance sampling exponent

    Raises
    ------
    ValueError
        If alpha or beta_start not in [0, 1]

    References
    ----------
    Schaul, T. et al. (2016). Prioritized Experience Replay. ICLR.

    See Also
    --------
    ReplayBuffer : Uniform sampling baseline
    SumTree : Underlying data structure
    """

    __slots__ = (
        "_capacity", "_alpha", "_beta", "_beta_start", "_beta_frames",
        "_epsilon", "_sum_tree", "_max_priority", "_frame",
    )

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        epsilon: float = 1e-6,
    ) -> None:
        """
        Initialize prioritized replay buffer.

        Parameters
        ----------
        capacity : int
            Maximum number of transitions to store
        alpha : float, default=0.6
            Prioritization exponent in [0, 1].
            0 = uniform sampling, 1 = full prioritization.
        beta_start : float, default=0.4
            Initial importance sampling exponent in [0, 1].
            Anneals to 1.0 over training.
        beta_frames : int, default=100000
            Number of frames to anneal β from beta_start to 1.0
        epsilon : float, default=1e-6
            Small constant added to priorities to prevent zero priority

        Raises
        ------
        ValueError
            If alpha or beta_start not in [0, 1], or capacity not positive
        """
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError(f"capacity must be positive integer, got {capacity!r}")
        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if not 0 <= beta_start <= 1:
            raise ValueError(f"beta_start must be in [0, 1], got {beta_start}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        self._capacity = capacity
        self._alpha = alpha
        self._beta_start = beta_start
        self._beta = beta_start
        self._beta_frames = max(1, beta_frames)
        self._epsilon = epsilon

        self._sum_tree = SumTree(capacity)
        self._max_priority = 1.0
        self._frame = 0

    @property
    def capacity(self) -> int:
        """Maximum buffer size (read-only)."""
        return self._capacity

    @property
    def beta(self) -> float:
        """Current importance sampling exponent."""
        return self._beta

    def __len__(self) -> int:
        """Current number of stored transitions."""
        return len(self._sum_tree)

    def push(
        self,
        state: FloatArray,
        action: int,
        reward: float,
        next_state: FloatArray,
        done: bool,
    ) -> None:
        """
        Store transition with maximum priority.

        New transitions receive max priority to guarantee they are
        sampled at least once before priority updates.

        Parameters
        ----------
        state : FloatArray
            Current state observation
        action : int
            Discrete action index
        reward : float
            Immediate reward
        next_state : FloatArray
            Next state observation
        done : bool
            Episode termination flag
        """
        transition = Transition(state, action, reward, next_state, done)
        priority = self._max_priority ** self._alpha
        self._sum_tree.add(priority, transition)

    def sample(
        self,
        batch_size: int,
    ) -> Tuple[FloatArray, IntArray, FloatArray, FloatArray, FloatArray, IntArray, FloatArray]:
        """
        Sample batch with prioritized probabilities.

        Uses stratified sampling: divides total priority into B equal
        segments and samples one transition from each segment.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample

        Returns
        -------
        states : FloatArray
            Batch of states, shape (batch_size, state_dim)
        actions : IntArray
            Batch of actions, shape (batch_size,)
        rewards : FloatArray
            Batch of rewards, shape (batch_size,)
        next_states : FloatArray
            Batch of next states, shape (batch_size, state_dim)
        dones : FloatArray
            Batch of done flags, shape (batch_size,)
        indices : IntArray
            Tree indices for priority updates, shape (batch_size,)
        weights : FloatArray
            Importance sampling weights, shape (batch_size,)

        Raises
        ------
        ValueError
            If batch_size exceeds buffer size
        """
        buffer_len = len(self._sum_tree)
        if batch_size > buffer_len:
            raise ValueError(
                f"batch_size ({batch_size}) exceeds buffer size ({buffer_len})"
            )

        self._anneal_beta()

        indices = np.empty(batch_size, dtype=np.int64)
        weights = np.empty(batch_size, dtype=np.float32)
        batch: List[Transition] = []

        total = self._sum_tree.total_priority
        segment = total / batch_size

        min_prob = self._sum_tree.min_priority() ** self._alpha / total
        max_weight = (buffer_len * min_prob) ** (-self._beta) if min_prob > 0 else 1.0

        for i in range(batch_size):
            low, high = segment * i, segment * (i + 1)
            cumsum = random.uniform(low, high)
            tree_idx, priority, data = self._sum_tree.get(cumsum)

            prob = priority / total
            weight = (buffer_len * prob) ** (-self._beta) / max_weight

            indices[i] = tree_idx
            weights[i] = weight
            batch.append(data)

        states = np.array([t.state for t in batch], dtype=np.float32)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        batch_rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.array([t.next_state for t in batch], dtype=np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)

        return states, actions, batch_rewards, next_states, dones, indices, weights

    def _anneal_beta(self) -> None:
        """Linearly anneal β from beta_start to 1.0 over beta_frames."""
        self._frame += 1
        fraction = min(1.0, self._frame / self._beta_frames)
        self._beta = self._beta_start + fraction * (1.0 - self._beta_start)

    def update_priorities(
        self,
        indices: IntArray,
        td_errors: FloatArray,
    ) -> None:
        """
        Update priorities based on TD errors.

        Priority formula: p_i = (|δ_i| + ε)^α

        Parameters
        ----------
        indices : IntArray
            Tree indices from sample(), shape (batch_size,)
        td_errors : FloatArray
            TD errors for each transition, shape (batch_size,)
        """
        priorities = (np.abs(td_errors) + self._epsilon) ** self._alpha
        for idx, priority in zip(indices, priorities):
            self._sum_tree.update_priority(int(idx), float(priority))
        self._max_priority = max(
            self._max_priority,
            float(np.max(priorities)) ** (1.0 / self._alpha)
        )

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has sufficient samples."""
        return len(self._sum_tree) >= min_size


class NStepReplayBuffer:
    """
    N-step experience replay buffer for multi-step learning.

    Core Idea (核心思想)
    --------------------
    N步回报通过在多步之后再进行bootstrap，实现了TD(0)和Monte Carlo之间的折中：

    - **更快的信用分配**: 奖励信号传播更远
    - **降低偏差**: 使用更多真实回报，减少bootstrap偏差
    - **增加方差**: 引入更多随机性

    Mathematical Foundation (数学基础)
    ----------------------------------
    **N-step Return**:

    .. math::
        G_t^{(n)} = \\sum_{k=0}^{n-1} \\gamma^k R_{t+k+1} + \\gamma^n V(S_{t+n})

    **N-step TD Target**:

    .. math::
        y_t^{(n)} = G_t^{(n)} = \\sum_{k=0}^{n-1} \\gamma^k r_{t+k+1} + \\gamma^n \\max_{a'} Q(s_{t+n}, a')

    **Bias-Variance Trade-off**:

    +-------+------------------+------------------+
    | n     | Bias             | Variance         |
    +=======+==================+==================+
    | 1     | High (bootstrap) | Low              |
    +-------+------------------+------------------+
    | 3-5   | Medium           | Medium           |
    +-------+------------------+------------------+
    | ∞     | Zero (MC)        | High             |
    +-------+------------------+------------------+

    Parameters
    ----------
    capacity : int
        Maximum number of n-step transitions to store
    n_steps : int, default=3
        Number of steps for returns calculation
    gamma : float, default=0.99
        Discount factor

    Attributes
    ----------
    capacity : int
        Maximum buffer size
    n_steps : int
        Number of steps for multi-step returns

    References
    ----------
    Sutton, R. S. (1988). Learning to predict by the methods of temporal
    differences. Machine learning, 3(1):9-44.
    """

    __slots__ = (
        "_capacity", "_n_steps", "_gamma", "_buffer", "_n_step_buffer",
    )

    def __init__(
        self,
        capacity: int,
        n_steps: int = 3,
        gamma: float = 0.99,
    ) -> None:
        """
        Initialize n-step replay buffer.

        Parameters
        ----------
        capacity : int
            Maximum number of n-step transitions to store
        n_steps : int, default=3
            Number of steps for return calculation. Must be >= 1.
        gamma : float, default=0.99
            Discount factor for return calculation. Must be in [0, 1].

        Raises
        ------
        ValueError
            If n_steps < 1 or gamma not in [0, 1]
        """
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError(f"capacity must be positive integer, got {capacity!r}")
        if not isinstance(n_steps, int) or n_steps < 1:
            raise ValueError(f"n_steps must be integer >= 1, got {n_steps}")
        if not 0 <= gamma <= 1:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")

        self._capacity = capacity
        self._n_steps = n_steps
        self._gamma = gamma

        self._buffer: Deque[NStepTransition] = deque(maxlen=capacity)
        self._n_step_buffer: Deque[Transition] = deque(maxlen=n_steps)

    @property
    def capacity(self) -> int:
        """Maximum buffer size (read-only)."""
        return self._capacity

    @property
    def n_steps(self) -> int:
        """Number of steps for multi-step returns."""
        return self._n_steps

    def __len__(self) -> int:
        """Current number of stored n-step transitions."""
        return len(self._buffer)

    def push(
        self,
        state: FloatArray,
        action: int,
        reward: float,
        next_state: FloatArray,
        done: bool,
    ) -> Optional[NStepTransition]:
        """
        Add transition and compute n-step return when ready.

        Accumulates single-step transitions until n steps collected,
        then computes discounted n-step return and stores the result.

        Parameters
        ----------
        state : FloatArray
            Current state observation
        action : int
            Discrete action index
        reward : float
            Immediate reward
        next_state : FloatArray
            Next state observation
        done : bool
            Episode termination flag

        Returns
        -------
        Optional[NStepTransition]
            N-step transition if n steps accumulated, else None.
            Also returns early if episode terminates.
        """
        self._n_step_buffer.append(
            Transition(state, action, reward, next_state, done)
        )

        if len(self._n_step_buffer) < self._n_steps:
            if done:
                return self._make_n_step_transition()
            return None

        return self._make_n_step_transition()

    def _make_n_step_transition(self) -> NStepTransition:
        """Compute n-step return and create transition."""
        n_step_return = 0.0
        gamma_power = 1.0

        for t in self._n_step_buffer:
            n_step_return += gamma_power * t.reward
            gamma_power *= self._gamma
            if t.done:
                break

        first = self._n_step_buffer[0]
        last = self._n_step_buffer[-1]

        n_step = NStepTransition(
            state=first.state,
            action=first.action,
            n_step_return=n_step_return,
            next_state=last.next_state,
            done=last.done,
            n_steps=len(self._n_step_buffer),
        )

        self._buffer.append(n_step)

        if last.done:
            self._n_step_buffer.clear()
        else:
            self._n_step_buffer.popleft()

        return n_step

    def sample(
        self,
        batch_size: int,
    ) -> Tuple[FloatArray, IntArray, FloatArray, FloatArray, FloatArray, IntArray]:
        """
        Sample n-step transitions.

        Returns:
            states, actions, n_step_returns, next_states, dones, n_steps
        """
        if batch_size > len(self._buffer):
            raise ValueError(
                f"batch_size {batch_size} exceeds buffer size {len(self._buffer)}"
            )

        batch = random.sample(self._buffer, batch_size)

        states = np.array([t.state for t in batch], dtype=np.float32)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        returns = np.array([t.n_step_return for t in batch], dtype=np.float32)
        next_states = np.array([t.next_state for t in batch], dtype=np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)
        n_steps = np.array([t.n_steps for t in batch], dtype=np.int64)

        return states, actions, returns, next_states, dones, n_steps

    def is_ready(self, min_size: int) -> bool:
        return len(self._buffer) >= min_size

    def reset_episode(self) -> None:
        """Clear n-step buffer at episode end."""
        self._n_step_buffer.clear()


# ============================================================
# Neural Network Layers
# ============================================================


class NoisyLinear(nn.Module):
    """
    Noisy linear layer for exploration (Fortunato et al., 2017).

    Replaces ε-greedy with parametric noise for state-dependent exploration:
        y = (μ^w + σ^w ⊙ ε^w) x + (μ^b + σ^b ⊙ ε^b)

    Uses factorized Gaussian noise for efficiency:
        ε_{ij} = f(ε_i) · f(ε_j), where f(x) = sign(x)√|x|

    Noise scales σ are learned, allowing the network to modulate
    exploration based on state uncertainty.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma_init: float = 0.5,
    ) -> None:
        """
        Initialize noisy linear layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            sigma_init: Initial noise scale
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        # Learnable parameters: μ and σ for weights and biases
        self.weight_mu = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        self.weight_sigma = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Noise buffers (not parameters)
        self.register_buffer(
            "weight_epsilon",
            torch.empty(out_features, in_features),
        )
        self.register_buffer(
            "bias_epsilon",
            torch.empty(out_features),
        )

        self._reset_parameters()
        self.reset_noise()

    def _reset_parameters(self) -> None:
        """Initialize μ and σ parameters."""
        # μ initialized uniformly in [-1/√fan_in, 1/√fan_in]
        bound = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)

        # σ initialized to sigma_init / √fan_in
        sigma = self.sigma_init / math.sqrt(self.in_features)
        self.weight_sigma.data.fill_(sigma)
        self.bias_sigma.data.fill_(sigma)

    def _factorized_noise(self, size: int) -> Tensor:
        """Generate factorized Gaussian noise: f(x) = sign(x)√|x|."""
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        """Sample new noise for exploration."""
        epsilon_in = self._factorized_noise(self.in_features)
        epsilon_out = self._factorized_noise(self.out_features)

        # Outer product for factorized noise: ε_{ij} = ε_i · ε_j
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with noisy weights.

        w = μ^w + σ^w ⊙ ε^w
        b = μ^b + σ^b ⊙ ε^b
        y = wx + b
        """
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # Evaluation: use mean parameters only
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"sigma_init={self.sigma_init}"
        )


# ============================================================
# Network Architectures
# ============================================================


def _init_weights(module: nn.Module, gain: float = math.sqrt(2)) -> None:
    """Orthogonal initialization for stable deep RL training."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class DQNNetwork(nn.Module):
    """
    Standard DQN network architecture.

    Maps states to Q-values: f_θ: ℝ^d → ℝ^{|A|}
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        layers: List[nn.Module] = []
        prev_dim = state_dim

        for _ in range(num_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, action_dim))

        self.network = nn.Sequential(*layers)
        self.apply(_init_weights)

    def forward(self, state: Tensor) -> Tensor:
        """Forward pass: state → Q-values."""
        return self.network(state)


class DuelingNetwork(nn.Module):
    """
    Dueling DQN architecture (Wang et al., 2016).

    Decomposes Q-function into state value and action advantage:
        Q(s, a) = V(s) + A(s, a) - mean_a A(s, a)

    Enables better generalization by separating "how good is the state"
    from "how good is the action relative to others."
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared feature extraction
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Value stream: V(s) ∈ ℝ
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        # Advantage stream: A(s, ·) ∈ ℝ^{|A|}
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )

        self.apply(_init_weights)

    def forward(self, state: Tensor) -> Tensor:
        """
        Forward pass with value-advantage aggregation.

        Q(s, a) = V(s) + [A(s, a) - mean_a' A(s, a')]
        """
        features = self.feature(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Mean subtraction for identifiability
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values


class NoisyNetwork(nn.Module):
    """
    Noisy DQN network for learned exploration (Fortunato et al., 2017).

    Replaces ε-greedy with parametric noise layers that learn
    state-dependent exploration through gradient descent.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        sigma_init: float = 0.5,
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Standard layers for feature extraction
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Noisy layers for exploration
        self.noisy1 = NoisyLinear(hidden_dim, hidden_dim, sigma_init)
        self.noisy2 = NoisyLinear(hidden_dim, action_dim, sigma_init)

    def forward(self, state: Tensor) -> Tensor:
        features = self.feature(state)
        x = F.relu(self.noisy1(features))
        return self.noisy2(x)

    def reset_noise(self) -> None:
        """Sample new noise for all noisy layers."""
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()


class NoisyDuelingNetwork(nn.Module):
    """
    Combines Dueling architecture with Noisy Networks.

    Features:
    - Value-advantage decomposition for stable learning
    - Parametric noise for learned exploration
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        sigma_init: float = 0.5,
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared feature extraction
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Noisy value stream
        self.value_noisy1 = NoisyLinear(hidden_dim, hidden_dim, sigma_init)
        self.value_noisy2 = NoisyLinear(hidden_dim, 1, sigma_init)

        # Noisy advantage stream
        self.adv_noisy1 = NoisyLinear(hidden_dim, hidden_dim, sigma_init)
        self.adv_noisy2 = NoisyLinear(hidden_dim, action_dim, sigma_init)

    def forward(self, state: Tensor) -> Tensor:
        features = self.feature(state)

        # Value stream
        value = F.relu(self.value_noisy1(features))
        value = self.value_noisy2(value)

        # Advantage stream
        advantage = F.relu(self.adv_noisy1(features))
        advantage = self.adv_noisy2(advantage)

        # Aggregate with mean subtraction
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values

    def reset_noise(self) -> None:
        """Reset noise for all noisy layers."""
        self.value_noisy1.reset_noise()
        self.value_noisy2.reset_noise()
        self.adv_noisy1.reset_noise()
        self.adv_noisy2.reset_noise()


class CategoricalNetwork(nn.Module):
    """
    Categorical DQN / C51 network (Bellemare et al., 2017).

    Models full return distribution instead of expected value:
        Z(s, a) ~ Categorical(z_1, ..., z_N; p_1, ..., p_N)

    Support atoms: z_i = V_min + i · Δz
    Output: Log probabilities for each atom

    Distributional perspective provides richer learning signal
    and leads to more stable optimization.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        # Support atoms: z_i = V_min + i · Δz
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        self.register_buffer(
            "support",
            torch.linspace(v_min, v_max, num_atoms),
        )

        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Output: |A| × N_atoms logits
        self.output = nn.Linear(hidden_dim, action_dim * num_atoms)

        self.apply(_init_weights)

    def forward(self, state: Tensor) -> Tensor:
        """
        Forward pass returning log probabilities.

        Returns:
            Log probabilities (batch, action_dim, num_atoms)
        """
        batch_size = state.shape[0]
        features = self.feature(state)
        logits = self.output(features)

        # Reshape to (batch, actions, atoms) and apply log-softmax
        logits = logits.view(batch_size, self.action_dim, self.num_atoms)
        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs

    def get_q_values(self, state: Tensor) -> Tensor:
        """Compute expected Q-values from distribution: Q(s,a) = Σ_i z_i · p_i."""
        log_probs = self.forward(state)
        probs = log_probs.exp()
        q_values = (probs * self.support).sum(dim=-1)
        return q_values


class CategoricalDuelingNetwork(nn.Module):
    """
    Categorical Dueling network combining distributional RL with value-advantage decomposition.

    Extends C51 with Dueling architecture:
    - Separate value and advantage distributions
    - Combines benefits of both approaches
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        self.register_buffer(
            "support",
            torch.linspace(v_min, v_max, num_atoms),
        )

        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Value stream: outputs atom probabilities
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_atoms),
        )

        # Advantage stream: outputs atom probabilities for each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim * num_atoms),
        )

        self.apply(_init_weights)

    def forward(self, state: Tensor) -> Tensor:
        """Forward pass with distributional dueling aggregation."""
        batch_size = state.shape[0]
        features = self.feature(state)

        value = self.value_stream(features).view(batch_size, 1, self.num_atoms)
        advantage = self.advantage_stream(features).view(
            batch_size, self.action_dim, self.num_atoms
        )

        # Aggregate: Q = V + (A - mean(A)) in logit space
        logits = value + (advantage - advantage.mean(dim=1, keepdim=True))
        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs

    def get_q_values(self, state: Tensor) -> Tensor:
        """Compute expected Q-values from distribution."""
        log_probs = self.forward(state)
        probs = log_probs.exp()
        q_values = (probs * self.support).sum(dim=-1)
        return q_values


# ============================================================
# Rainbow Network (Full Combination)
# ============================================================


class RainbowNetwork(nn.Module):
    """
    Rainbow network combining all DQN improvements (Hessel et al., 2018).

    Integrates:
    - Dueling architecture (value-advantage decomposition)
    - Noisy networks (learned exploration)
    - Categorical distribution (distributional RL)

    Achieves state-of-the-art performance on Atari benchmarks
    by combining complementary algorithmic improvements.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        sigma_init: float = 0.5,
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        self.register_buffer(
            "support",
            torch.linspace(v_min, v_max, num_atoms),
        )

        # Shared feature extraction
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Noisy value stream
        self.value_noisy1 = NoisyLinear(hidden_dim, hidden_dim, sigma_init)
        self.value_noisy2 = NoisyLinear(hidden_dim, num_atoms, sigma_init)

        # Noisy advantage stream
        self.adv_noisy1 = NoisyLinear(hidden_dim, hidden_dim, sigma_init)
        self.adv_noisy2 = NoisyLinear(
            hidden_dim, action_dim * num_atoms, sigma_init
        )

    def forward(self, state: Tensor) -> Tensor:
        """Forward pass with all Rainbow components."""
        batch_size = state.shape[0]
        features = self.feature(state)

        # Value stream
        value = F.relu(self.value_noisy1(features))
        value = self.value_noisy2(value).view(batch_size, 1, self.num_atoms)

        # Advantage stream
        advantage = F.relu(self.adv_noisy1(features))
        advantage = self.adv_noisy2(advantage).view(
            batch_size, self.action_dim, self.num_atoms
        )

        # Dueling aggregation in logit space
        logits = value + (advantage - advantage.mean(dim=1, keepdim=True))
        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs

    def get_q_values(self, state: Tensor) -> Tensor:
        """Compute expected Q-values: Q(s,a) = Σ_i z_i · p_i."""
        log_probs = self.forward(state)
        probs = log_probs.exp()
        q_values = (probs * self.support).sum(dim=-1)
        return q_values

    def reset_noise(self) -> None:
        """Reset noise for exploration."""
        self.value_noisy1.reset_noise()
        self.value_noisy2.reset_noise()
        self.adv_noisy1.reset_noise()
        self.adv_noisy2.reset_noise()


# ============================================================
# Configuration
# ============================================================


class DQNVariant(Enum):
    """Available DQN algorithm variants."""
    VANILLA = "vanilla"
    DOUBLE = "double"
    DUELING = "dueling"
    NOISY = "noisy"
    CATEGORICAL = "categorical"
    DOUBLE_DUELING = "double_dueling"
    RAINBOW = "rainbow"


@dataclass
class DQNVariantConfig:
    """
    Configuration for DQN variant agents.

    Centralizes all hyperparameters with validation for reproducibility
    and systematic experimentation.
    """

    # Environment
    state_dim: int
    action_dim: int

    # Architecture
    hidden_dim: int = 128
    num_layers: int = 2

    # Distributional RL
    num_atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0

    # Noisy Networks
    sigma_init: float = 0.5

    # Learning
    learning_rate: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 64

    # Exploration (for non-noisy variants)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 10000

    # Replay buffer
    buffer_size: int = 100000
    min_buffer_size: int = 1000

    # Prioritized replay
    use_per: bool = False
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 100000

    # Multi-step learning
    n_steps: int = 1

    # Target network
    target_update_freq: int = 100
    soft_update_tau: Optional[float] = None

    # Training
    grad_clip: float = 10.0
    device: str = "auto"
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {self.state_dim}")
        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {self.action_dim}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if not 0 < self.learning_rate <= 1:
            raise ValueError(f"learning_rate must be in (0, 1], got {self.learning_rate}")
        if not 0 <= self.gamma <= 1:
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {self.n_steps}")
        if self.num_atoms < 2:
            raise ValueError(f"num_atoms must be >= 2, got {self.num_atoms}")
        if self.v_min >= self.v_max:
            raise ValueError(f"v_min ({self.v_min}) must be < v_max ({self.v_max})")

    def get_device(self) -> torch.device:
        """Get compute device."""
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.device)


# ============================================================
# DQN Variant Agent
# ============================================================


class DQNVariantAgent:
    """
    Unified DQN variant agent supporting all algorithmic improvements.

    Supports:
    - Double DQN: Decoupled action selection and evaluation
    - Dueling DQN: Value-advantage decomposition
    - Noisy DQN: Learned exploration via noisy layers
    - Categorical DQN: Distributional value estimation
    - Rainbow: All improvements combined

    Provides consistent API for training, evaluation, and model persistence.
    """

    def __init__(
        self,
        config: DQNVariantConfig,
        variant: DQNVariant = DQNVariant.DOUBLE_DUELING,
    ) -> None:
        """
        Initialize agent with specified variant.

        Args:
            config: Hyperparameter configuration
            variant: Algorithm variant to use
        """
        self.config = config
        self.variant = variant
        self.device = config.get_device()

        if config.seed is not None:
            self._set_seed(config.seed)

        self._init_networks()
        self._init_buffer()
        self._init_training_state()

    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _init_networks(self) -> None:
        """Initialize online and target networks based on variant."""
        cfg = self.config

        # Select network architecture
        if self.variant == DQNVariant.VANILLA:
            self.q_network = DQNNetwork(
                cfg.state_dim, cfg.action_dim, cfg.hidden_dim, cfg.num_layers
            )
        elif self.variant == DQNVariant.DOUBLE:
            self.q_network = DQNNetwork(
                cfg.state_dim, cfg.action_dim, cfg.hidden_dim, cfg.num_layers
            )
        elif self.variant == DQNVariant.DUELING:
            self.q_network = DuelingNetwork(
                cfg.state_dim, cfg.action_dim, cfg.hidden_dim
            )
        elif self.variant == DQNVariant.NOISY:
            self.q_network = NoisyNetwork(
                cfg.state_dim, cfg.action_dim, cfg.hidden_dim, cfg.sigma_init
            )
        elif self.variant == DQNVariant.CATEGORICAL:
            self.q_network = CategoricalNetwork(
                cfg.state_dim, cfg.action_dim, cfg.hidden_dim,
                cfg.num_atoms, cfg.v_min, cfg.v_max
            )
        elif self.variant == DQNVariant.DOUBLE_DUELING:
            self.q_network = DuelingNetwork(
                cfg.state_dim, cfg.action_dim, cfg.hidden_dim
            )
        elif self.variant == DQNVariant.RAINBOW:
            self.q_network = RainbowNetwork(
                cfg.state_dim, cfg.action_dim, cfg.hidden_dim,
                cfg.num_atoms, cfg.v_min, cfg.v_max, cfg.sigma_init
            )
        else:
            raise ValueError(f"Unknown variant: {self.variant}")

        self.q_network = self.q_network.to(self.device)

        # Create target network (same architecture)
        self.target_network = type(self.q_network)(
            *self._get_network_args()
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        for param in self.target_network.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=cfg.learning_rate,
        )

    def _get_network_args(self) -> tuple:
        """Get constructor arguments for network based on variant."""
        cfg = self.config

        if self.variant in (DQNVariant.VANILLA, DQNVariant.DOUBLE):
            return (cfg.state_dim, cfg.action_dim, cfg.hidden_dim, cfg.num_layers)
        elif self.variant in (DQNVariant.DUELING, DQNVariant.DOUBLE_DUELING):
            return (cfg.state_dim, cfg.action_dim, cfg.hidden_dim)
        elif self.variant == DQNVariant.NOISY:
            return (cfg.state_dim, cfg.action_dim, cfg.hidden_dim, cfg.sigma_init)
        elif self.variant == DQNVariant.CATEGORICAL:
            return (
                cfg.state_dim, cfg.action_dim, cfg.hidden_dim,
                cfg.num_atoms, cfg.v_min, cfg.v_max
            )
        elif self.variant == DQNVariant.RAINBOW:
            return (
                cfg.state_dim, cfg.action_dim, cfg.hidden_dim,
                cfg.num_atoms, cfg.v_min, cfg.v_max, cfg.sigma_init
            )
        return ()

    def _init_buffer(self) -> None:
        """Initialize replay buffer based on configuration."""
        cfg = self.config

        if cfg.n_steps > 1:
            self.buffer = NStepReplayBuffer(
                cfg.buffer_size, cfg.n_steps, cfg.gamma
            )
        elif cfg.use_per:
            self.buffer = PrioritizedReplayBuffer(
                cfg.buffer_size, cfg.per_alpha, cfg.per_beta_start,
                cfg.per_beta_frames
            )
        else:
            self.buffer = ReplayBuffer(cfg.buffer_size)

    def _init_training_state(self) -> None:
        """Initialize training state variables."""
        self._epsilon = self.config.epsilon_start
        self._training_step = 0
        self._update_count = 0
        self._losses: List[float] = []
        self._q_values: List[float] = []

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def training_step(self) -> int:
        return self._training_step

    @property
    def losses(self) -> List[float]:
        return self._losses.copy()

    def _uses_noisy_exploration(self) -> bool:
        """Check if variant uses noisy networks for exploration."""
        return self.variant in (DQNVariant.NOISY, DQNVariant.RAINBOW)

    def _is_distributional(self) -> bool:
        """Check if variant uses distributional RL."""
        return self.variant in (DQNVariant.CATEGORICAL, DQNVariant.RAINBOW)

    def _uses_double(self) -> bool:
        """Check if variant uses Double DQN."""
        return self.variant in (
            DQNVariant.DOUBLE,
            DQNVariant.DOUBLE_DUELING,
            DQNVariant.RAINBOW,
        )

    def select_action(self, state: FloatArray, training: bool = True) -> int:
        """
        Select action using exploration policy.

        For noisy variants: Uses network noise for exploration
        For others: Uses ε-greedy
        """
        # Noisy networks: exploration is built-in
        if self._uses_noisy_exploration():
            if training and hasattr(self.q_network, "reset_noise"):
                self.q_network.reset_noise()

            state_tensor = torch.as_tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            with torch.no_grad():
                if self._is_distributional():
                    q_values = self.q_network.get_q_values(state_tensor)
                else:
                    q_values = self.q_network(state_tensor)

            return int(q_values.argmax(dim=1).item())

        # ε-greedy exploration
        if training and random.random() < self._epsilon:
            return random.randint(0, self.config.action_dim - 1)

        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            if self._is_distributional():
                q_values = self.q_network.get_q_values(state_tensor)
            else:
                q_values = self.q_network(state_tensor)

        return int(q_values.argmax(dim=1).item())

    def store_transition(
        self,
        state: FloatArray,
        action: int,
        reward: float,
        next_state: FloatArray,
        done: bool,
    ) -> None:
        """Store transition in replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    def update(self) -> Optional[float]:
        """
        Perform one gradient update step.

        Returns loss value or None if buffer insufficient.
        """
        if not self.buffer.is_ready(self.config.min_buffer_size):
            return None

        if self._is_distributional():
            return self._update_distributional()
        return self._update_value()

    def _update_value(self) -> float:
        """Update for value-based variants (DQN, Double, Dueling, Noisy)."""
        cfg = self.config

        # Sample batch
        if isinstance(self.buffer, PrioritizedReplayBuffer):
            (
                states, actions, rewards, next_states, dones, indices, weights
            ) = self.buffer.sample(cfg.batch_size)
            weights_t = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        elif isinstance(self.buffer, NStepReplayBuffer):
            states, actions, rewards, next_states, dones, n_steps = (
                self.buffer.sample(cfg.batch_size)
            )
            weights_t = None
            # Adjust gamma for n-step returns
            gamma_n = cfg.gamma ** n_steps
        else:
            states, actions, rewards, next_states, dones = (
                self.buffer.sample(cfg.batch_size)
            )
            weights_t = None

        # Convert to tensors
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        # Reset noise for noisy variants
        if self._uses_noisy_exploration():
            self.q_network.reset_noise()
            self.target_network.reset_noise()

        # Current Q-values
        current_q_values = self.q_network(states_t)
        current_q = current_q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Compute target Q-values
        with torch.no_grad():
            if self._uses_double():
                # Double DQN: select action with online, evaluate with target
                next_actions = self.q_network(next_states_t).argmax(dim=1)
                next_q = self.target_network(next_states_t).gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze(1)
            else:
                # Standard DQN
                next_q = self.target_network(next_states_t).max(dim=1)[0]

            # TD target
            if isinstance(self.buffer, NStepReplayBuffer):
                target_q = rewards_t + (gamma_n * next_q * (1.0 - dones_t))
            else:
                target_q = rewards_t + (cfg.gamma * next_q * (1.0 - dones_t))

        # Compute loss
        td_errors = current_q - target_q

        if weights_t is not None:
            # Weighted loss for PER
            loss = (weights_t * (td_errors ** 2)).mean()
            # Update priorities
            self.buffer.update_priorities(
                indices, td_errors.detach().cpu().numpy()
            )
        else:
            loss = F.smooth_l1_loss(current_q, target_q)

        # Optimize
        self._optimize(loss)

        loss_value = loss.item()
        self._losses.append(loss_value)
        self._q_values.append(current_q.mean().item())

        return loss_value

    def _update_distributional(self) -> float:
        """Update for distributional variants (Categorical, Rainbow)."""
        cfg = self.config

        # Sample batch
        if isinstance(self.buffer, PrioritizedReplayBuffer):
            (
                states, actions, rewards, next_states, dones, indices, weights
            ) = self.buffer.sample(cfg.batch_size)
            weights_t = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        else:
            states, actions, rewards, next_states, dones = (
                self.buffer.sample(cfg.batch_size)
            )
            weights_t = None

        batch_size = len(states)

        # Convert to tensors
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        # Reset noise
        if self._uses_noisy_exploration():
            self.q_network.reset_noise()
            self.target_network.reset_noise()

        # Current distribution: log p(s, a)
        current_log_probs = self.q_network(states_t)
        current_log_probs = current_log_probs[
            torch.arange(batch_size, device=self.device), actions_t
        ]  # (batch, atoms)

        # Compute target distribution
        with torch.no_grad():
            # Select greedy action for next state
            next_q_values = self.target_network.get_q_values(next_states_t)

            if self._uses_double():
                # Double: use online network for action selection
                online_next_q = self.q_network.get_q_values(next_states_t)
                next_actions = online_next_q.argmax(dim=1)
            else:
                next_actions = next_q_values.argmax(dim=1)

            # Get next state distribution
            next_log_probs = self.target_network(next_states_t)
            next_probs = next_log_probs.exp()
            next_probs = next_probs[
                torch.arange(batch_size, device=self.device), next_actions
            ]  # (batch, atoms)

            # Project distribution onto support
            target_probs = self._project_distribution(
                next_probs, rewards_t, dones_t
            )

        # KL divergence loss
        loss = -(target_probs * current_log_probs).sum(dim=-1)

        if weights_t is not None:
            loss = (weights_t * loss).mean()
            # Update priorities using expected Q-value error
            with torch.no_grad():
                current_q = (current_log_probs.exp() * self.q_network.support).sum(dim=-1)
                target_q = (target_probs * self.q_network.support).sum(dim=-1)
                td_errors = (current_q - target_q).cpu().numpy()
            self.buffer.update_priorities(indices, td_errors)
        else:
            loss = loss.mean()

        # Optimize
        self._optimize(loss)

        loss_value = loss.item()
        self._losses.append(loss_value)

        return loss_value

    def _project_distribution(
        self,
        next_probs: Tensor,
        rewards: Tensor,
        dones: Tensor,
    ) -> Tensor:
        """
        Project target distribution onto fixed support.

        Implements categorical projection algorithm from C51 paper:
        1. Compute projected atoms: Tz_j = r + γz_j (clipped to [V_min, V_max])
        2. Distribute probability mass to neighboring support atoms
        """
        cfg = self.config
        batch_size = next_probs.shape[0]

        support = self.q_network.support  # (atoms,)
        delta_z = self.q_network.delta_z

        # Compute projected support: Tz = r + γz (for non-terminal)
        # Shape: (batch, 1) + γ * (atoms,) → (batch, atoms)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        # For terminal states, only reward matters (no bootstrapping)
        tz = rewards + (1 - dones) * cfg.gamma * support
        tz = tz.clamp(cfg.v_min, cfg.v_max)

        # Compute indices and interpolation weights
        b = (tz - cfg.v_min) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        # Handle edge cases
        l = l.clamp(0, cfg.num_atoms - 1)
        u = u.clamp(0, cfg.num_atoms - 1)

        # Initialize target distribution
        target_probs = torch.zeros_like(next_probs)

        # Distribute probability mass
        offset = (
            torch.arange(batch_size, device=self.device).unsqueeze(1)
            * cfg.num_atoms
        )

        target_probs.view(-1).index_add_(
            0,
            (l + offset).view(-1),
            (next_probs * (u.float() - b)).view(-1),
        )
        target_probs.view(-1).index_add_(
            0,
            (u + offset).view(-1),
            (next_probs * (b - l.float())).view(-1),
        )

        return target_probs

    def _optimize(self, loss: Tensor) -> None:
        """Perform optimization step with gradient clipping."""
        self.optimizer.zero_grad()
        loss.backward()

        if self.config.grad_clip is not None:
            nn.utils.clip_grad_norm_(
                self.q_network.parameters(),
                self.config.grad_clip,
            )

        self.optimizer.step()

        self._update_count += 1
        self._sync_target_network()

    def _sync_target_network(self) -> None:
        """Synchronize target network parameters."""
        cfg = self.config

        if cfg.soft_update_tau is not None:
            # Soft update: θ⁻ ← τθ + (1-τ)θ⁻
            tau = cfg.soft_update_tau
            for target_p, online_p in zip(
                self.target_network.parameters(),
                self.q_network.parameters(),
            ):
                target_p.data.copy_(
                    tau * online_p.data + (1 - tau) * target_p.data
                )
        elif self._update_count % cfg.target_update_freq == 0:
            # Hard update: θ⁻ ← θ
            self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self) -> None:
        """Decay exploration rate (for non-noisy variants)."""
        if self._uses_noisy_exploration():
            return

        self._training_step += 1
        decay_progress = min(1.0, self._training_step / self.config.epsilon_decay_steps)
        self._epsilon = (
            self.config.epsilon_start
            + (self.config.epsilon_end - self.config.epsilon_start) * decay_progress
        )

    def train_step(
        self,
        state: FloatArray,
        action: int,
        reward: float,
        next_state: FloatArray,
        done: bool,
    ) -> Optional[float]:
        """Complete training step: store → update → decay epsilon."""
        self.store_transition(state, action, reward, next_state, done)
        loss = self.update()
        self.decay_epsilon()
        return loss

    def save(self, path: Union[str, Path]) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "variant": self.variant.value,
            "config": self.config.__dict__,
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self._epsilon,
            "training_step": self._training_step,
            "update_count": self._update_count,
            "losses": self._losses[-1000:],
        }
        torch.save(checkpoint, path)

    def load(self, path: Union[str, Path]) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self._epsilon = checkpoint.get("epsilon", self.config.epsilon_end)
        self._training_step = checkpoint.get("training_step", 0)
        self._update_count = checkpoint.get("update_count", 0)
        self._losses = checkpoint.get("losses", [])

    def set_train_mode(self) -> None:
        """Set network to training mode."""
        self.q_network.train()

    def set_eval_mode(self) -> None:
        """Set network to evaluation mode."""
        self.q_network.eval()


# ============================================================
# Training Utilities
# ============================================================


def train_agent(
    agent: DQNVariantAgent,
    env_name: str = "CartPole-v1",
    num_episodes: int = 500,
    max_steps: int = 500,
    eval_interval: int = 50,
    verbose: bool = True,
) -> Tuple[List[float], List[float]]:
    """
    Train DQN variant agent on Gymnasium environment.

    Args:
        agent: Configured DQNVariantAgent
        env_name: Gymnasium environment ID
        num_episodes: Training episodes
        max_steps: Maximum steps per episode
        eval_interval: Episodes between evaluations
        verbose: Print progress

    Returns:
        (training_rewards, eval_rewards)
    """
    if not HAS_GYM:
        raise RuntimeError("gymnasium not installed")

    env = gym.make(env_name)
    training_rewards: List[float] = []
    eval_rewards: List[float] = []

    if verbose:
        print(f"\nTraining {agent.variant.value.upper()} DQN on {env_name}")
        print("=" * 60)

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0

        for step in range(max_steps):
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.train_step(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if done:
                break

        # Handle n-step buffer episode reset
        if isinstance(agent.buffer, NStepReplayBuffer):
            agent.buffer.reset_episode()

        training_rewards.append(episode_reward)

        # Progress reporting
        if verbose and (episode + 1) % eval_interval == 0:
            avg_reward = np.mean(training_rewards[-eval_interval:])
            eval_reward = evaluate_agent(agent, env_name, num_episodes=5)
            eval_rewards.append(eval_reward)

            print(
                f"Episode {episode + 1:4d} | "
                f"Train: {avg_reward:7.2f} | "
                f"Eval: {eval_reward:7.2f} | "
                f"ε: {agent.epsilon:.3f}"
            )

    env.close()
    return training_rewards, eval_rewards


def evaluate_agent(
    agent: DQNVariantAgent,
    env_name: str,
    num_episodes: int = 10,
) -> float:
    """
    Evaluate agent with greedy policy.

    Returns:
        Mean episode reward
    """
    if not HAS_GYM:
        return 0.0

    env = gym.make(env_name)
    rewards: List[float] = []

    agent.set_eval_mode()

    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward

        rewards.append(episode_reward)

    agent.set_train_mode()
    env.close()

    return float(np.mean(rewards))


def compare_variants(
    env_name: str = "CartPole-v1",
    num_episodes: int = 300,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """
    Compare all DQN variants on environment.

    Returns:
        Dictionary mapping variant names to reward histories
    """
    if not HAS_GYM:
        raise RuntimeError("gymnasium not installed")

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()

    results: Dict[str, List[float]] = {}

    variants = [
        DQNVariant.VANILLA,
        DQNVariant.DOUBLE,
        DQNVariant.DUELING,
        DQNVariant.DOUBLE_DUELING,
        DQNVariant.NOISY,
        DQNVariant.CATEGORICAL,
    ]

    for variant in variants:
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Training {variant.value.upper()}")
            print("=" * 60)

        config = DQNVariantConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=128,
            learning_rate=1e-3,
            gamma=0.99,
            batch_size=64,
            buffer_size=100000,
            target_update_freq=100,
            seed=seed,
        )

        agent = DQNVariantAgent(config, variant)
        rewards, _ = train_agent(
            agent,
            env_name=env_name,
            num_episodes=num_episodes,
            verbose=verbose,
        )
        results[variant.value] = rewards

    return results


def plot_comparison(
    results: Dict[str, List[float]],
    title: str = "DQN Variants Comparison",
    window_size: int = 20,
    save_path: Optional[str] = None,
) -> None:
    """Plot training curves for algorithm comparison."""
    if not HAS_MATPLOTLIB:
        warnings.warn("matplotlib not installed, skipping plot")
        return

    plt.figure(figsize=(12, 6))
    colors = plt.cm.tab10.colors

    for idx, (name, rewards) in enumerate(results.items()):
        if len(rewards) >= window_size:
            smoothed = np.convolve(
                rewards,
                np.ones(window_size) / window_size,
                mode="valid",
            )
            plt.plot(
                smoothed,
                label=name.upper(),
                color=colors[idx % len(colors)],
                linewidth=2,
                alpha=0.8,
            )

    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Total Reward", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved: {save_path}")

    plt.close()


# ============================================================
# Unit Tests
# ============================================================


def _run_tests() -> bool:
    """Run comprehensive unit tests."""
    print("\n" + "=" * 60)
    print("DQN Variants Unit Tests")
    print("=" * 60)

    passed = 0
    failed = 0

    # Test 1: Replay buffers
    print("\n[Test 1] Replay Buffers")
    try:
        # Standard buffer
        buffer = ReplayBuffer(100)
        state = np.random.randn(4).astype(np.float32)
        for i in range(50):
            buffer.push(state, i % 2, float(i), state, False)
        assert len(buffer) == 50
        batch = buffer.sample(32)
        assert batch[0].shape == (32, 4)

        # PER buffer
        per_buffer = PrioritizedReplayBuffer(100)
        for i in range(50):
            per_buffer.push(state, i % 2, float(i), state, False)
        assert len(per_buffer) == 50
        batch = per_buffer.sample(32)
        assert len(batch) == 7
        per_buffer.update_priorities(batch[5], np.random.randn(32))

        # N-step buffer
        n_buffer = NStepReplayBuffer(100, n_steps=3, gamma=0.99)
        for i in range(20):
            n_buffer.push(state, 0, 1.0, state, i == 19)
        assert len(n_buffer) > 0

        print("  PASSED")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1

    # Test 2: NoisyLinear
    print("\n[Test 2] NoisyLinear Layer")
    try:
        layer = NoisyLinear(64, 32)
        x = torch.randn(16, 64)

        # Training mode (with noise)
        layer.train()
        out1 = layer(x)
        layer.reset_noise()
        out2 = layer(x)
        assert out1.shape == (16, 32)
        assert not torch.allclose(out1, out2)  # Noise should differ

        # Eval mode (without noise)
        layer.eval()
        out3 = layer(x)
        out4 = layer(x)
        assert torch.allclose(out3, out4)  # Deterministic

        print("  PASSED")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1

    # Test 3: Network architectures
    print("\n[Test 3] Network Architectures")
    try:
        state_dim, action_dim, batch = 4, 2, 16
        x = torch.randn(batch, state_dim)

        # Standard DQN
        net = DQNNetwork(state_dim, action_dim)
        assert net(x).shape == (batch, action_dim)

        # Dueling
        net = DuelingNetwork(state_dim, action_dim)
        assert net(x).shape == (batch, action_dim)

        # Noisy
        net = NoisyNetwork(state_dim, action_dim)
        assert net(x).shape == (batch, action_dim)

        # Categorical
        net = CategoricalNetwork(state_dim, action_dim, num_atoms=51)
        log_probs = net(x)
        assert log_probs.shape == (batch, action_dim, 51)
        q_values = net.get_q_values(x)
        assert q_values.shape == (batch, action_dim)

        # Rainbow
        net = RainbowNetwork(state_dim, action_dim, num_atoms=51)
        log_probs = net(x)
        assert log_probs.shape == (batch, action_dim, 51)

        print("  PASSED")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1

    # Test 4: Agent variants
    print("\n[Test 4] Agent Variants")
    try:
        config = DQNVariantConfig(state_dim=4, action_dim=2, batch_size=32)

        for variant in DQNVariant:
            agent = DQNVariantAgent(config, variant)

            state = np.random.randn(4).astype(np.float32)
            action = agent.select_action(state, training=True)
            assert 0 <= action < 2

            # Fill buffer
            for _ in range(64):
                s = np.random.randn(4).astype(np.float32)
                agent.store_transition(s, 0, 1.0, s, False)

            # Update
            loss = agent.update()
            assert loss is not None or len(agent.buffer) < config.min_buffer_size

        print("  PASSED")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1

    # Test 5: Environment interaction
    print("\n[Test 5] Environment Interaction")
    if HAS_GYM:
        try:
            env = gym.make("CartPole-v1")
            config = DQNVariantConfig(
                state_dim=4,
                action_dim=2,
                batch_size=32,
                min_buffer_size=64,
            )
            agent = DQNVariantAgent(config, DQNVariant.DOUBLE_DUELING)

            state, _ = env.reset()
            for _ in range(100):
                action = agent.select_action(state, training=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                agent.train_step(state, action, reward, next_state, terminated or truncated)

                if terminated or truncated:
                    state, _ = env.reset()
                else:
                    state = next_state

            env.close()
            print("  PASSED")
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1
    else:
        print("  SKIPPED (gymnasium not installed)")

    # Test 6: Save/Load
    print("\n[Test 6] Model Persistence")
    try:
        import tempfile

        config = DQNVariantConfig(state_dim=4, action_dim=2)
        agent1 = DQNVariantAgent(config, DQNVariant.DOUBLE_DUELING)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            temp_path = f.name

        agent1.save(temp_path)

        agent2 = DQNVariantAgent(config, DQNVariant.DOUBLE_DUELING)
        agent2.load(temp_path)

        os.remove(temp_path)

        for p1, p2 in zip(
            agent1.q_network.parameters(),
            agent2.q_network.parameters(),
        ):
            assert torch.allclose(p1, p2)

        print("  PASSED")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("All tests passed!")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--train":
            if HAS_GYM:
                config = DQNVariantConfig(
                    state_dim=4,
                    action_dim=2,
                    hidden_dim=128,
                    learning_rate=1e-3,
                    batch_size=64,
                    buffer_size=100000,
                    target_update_freq=100,
                    seed=42,
                )
                agent = DQNVariantAgent(config, DQNVariant.RAINBOW)
                rewards, _ = train_agent(agent, num_episodes=500)
            else:
                print("gymnasium not installed")
        elif sys.argv[1] == "--compare":
            if HAS_GYM:
                results = compare_variants(num_episodes=300, seed=42)
                plot_comparison(results, save_path="dqn_variants_comparison.png")
            else:
                print("gymnasium not installed")
        else:
            _run_tests()
    else:
        _run_tests()
