#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Q-Network (DQN) Production-Grade Implementation

============================================================
Core Idea
============================================================
Deep Q-Network revolutionized reinforcement learning by introducing deep neural
networks as function approximators for the action-value function Q(s, a). The key
insight is that neural networks can generalize across similar states, enabling
learning in high-dimensional continuous state spaces where tabular methods fail.

============================================================
Mathematical Foundation
============================================================

1. Bellman Optimality Equation
   The theoretical foundation of Q-learning:

   .. math::
       Q^*(s, a) = \\mathbb{E}\\left[ r + \\gamma \\max_{a'} Q^*(s', a') \\mid s, a \\right]

   where:
   - :math:`Q^*(s, a)`: optimal action-value function
   - :math:`r`: immediate reward
   - :math:`\\gamma \\in [0, 1]`: discount factor
   - :math:`s'`: next state

2. DQN Loss Function (Temporal Difference Error)

   .. math::
       L(\\theta) = \\mathbb{E}_{(s,a,r,s') \\sim \\mathcal{D}}
       \\left[ \\left( y - Q(s, a; \\theta) \\right)^2 \\right]

   TD Target:

   .. math::
       y = r + \\gamma \\max_{a'} Q(s', a'; \\theta^-)

   where:
   - :math:`\\theta`: online network parameters (updated every step)
   - :math:`\\theta^-`: target network parameters (synchronized periodically)
   - :math:`\\mathcal{D}`: experience replay buffer

3. Double DQN (Decoupling Selection and Evaluation)

   Standard DQN suffers from overestimation due to max operator:

   .. math::
       \\mathbb{E}[\\max_a Q(s,a)] \\geq \\max_a \\mathbb{E}[Q(s,a)]

   Double DQN addresses this by using online network for action selection
   and target network for evaluation:

   .. math::
       y^{\\text{Double}} = r + \\gamma Q\\left(s', \\arg\\max_{a'} Q(s', a'; \\theta); \\theta^-\\right)

4. Dueling Architecture (Value-Advantage Decomposition)

   .. math::
       Q(s, a) = V(s) + A(s, a) - \\frac{1}{|\\mathcal{A}|} \\sum_{a'} A(s, a')

   where:
   - :math:`V(s)`: state value function
   - :math:`A(s, a)`: advantage function
   - Mean subtraction ensures identifiability

============================================================
Problem Statement
============================================================

Traditional Q-learning faces fundamental challenges:

1. Curse of Dimensionality:
   - State space grows exponentially: :math:`|S| = O(d^n)` for n-dimensional states
   - Atari games: ~:math:`10^{20000}` possible pixel configurations
   - Q-tables cannot store such massive state spaces

2. Lack of Generalization:
   - Each state is learned independently
   - No knowledge transfer between similar states
   - New states require learning from scratch

3. Raw Sensory Input:
   - Real-world problems often involve images, sensor data
   - Requires automatic feature extraction

DQN Solution:
- Neural networks compress state space into learned representations
- Weight sharing enables automatic generalization
- End-to-end learning from pixels to actions

============================================================
Algorithm Comparison
============================================================

+----------------+------------------+-----------------+-----------------+
| Feature        | Tabular Q        | DQN             | Double DQN      |
+================+==================+=================+=================+
| State Space    | Discrete/Small   | Continuous/High | Continuous/High |
+----------------+------------------+-----------------+-----------------+
| Function Form  | Lookup Table     | Neural Network  | Neural Network  |
+----------------+------------------+-----------------+-----------------+
| Generalization | None             | Strong          | Strong          |
+----------------+------------------+-----------------+-----------------+
| Overestimation | Present          | Severe          | Significantly   |
|                |                  |                 | Reduced         |
+----------------+------------------+-----------------+-----------------+
| Sample         | Low              | High            | High            |
| Efficiency     |                  |                 |                 |
+----------------+------------------+-----------------+-----------------+

============================================================
Complexity Analysis
============================================================

Let:
- d = state dimension
- h = hidden layer width
- |A| = number of actions
- B = batch size
- N = buffer capacity

Space Complexity:
- Network parameters: O(d·h + h² + h·|A|)
- Target network: same
- Replay buffer: O(N · d)
- Total: O(N · d + |θ|)

Time Complexity (per update):
- Forward pass: O(B · |θ|)
- Backward pass: O(B · |θ|)
- Buffer sampling: O(B) for uniform, O(B·log N) for prioritized
- Total: O(B · |θ|)

============================================================
Algorithm Summary
============================================================

DQN Training Loop:
1. Collect experience: agent interacts with environment, stores (s, a, r, s', done)
2. Sample batch: uniform random sampling from replay buffer
3. Compute target: y = r + γ max_{a'} Q(s', a'; θ⁻)
4. Compute loss: L = (y - Q(s, a; θ))²
5. Gradient descent: update online network θ
6. Sync target: periodically θ⁻ ← θ

Key Insights:
- Experience replay transforms RL into supervised learning
- Target network provides stable regression targets
- Deep networks automatically learn state representations

============================================================
References
============================================================
[1] Mnih, V., et al. (2013). Playing Atari with Deep Reinforcement Learning.
    NIPS Deep Learning Workshop.
[2] Mnih, V., et al. (2015). Human-level control through deep reinforcement
    learning. Nature, 518(7540):529-533.
[3] van Hasselt, H., et al. (2016). Deep Reinforcement Learning with Double
    Q-learning. AAAI.
[4] Wang, Z., et al. (2016). Dueling Network Architectures for Deep
    Reinforcement Learning. ICML.
[5] Schaul, T., et al. (2016). Prioritized Experience Replay. ICLR.
[6] Hessel, M., et al. (2018). Rainbow: Combining Improvements in Deep
    Reinforcement Learning. AAAI.

============================================================
Dependencies
============================================================
    pip install torch numpy gymnasium matplotlib

Author: zimingttkx
License: MIT
"""

from __future__ import annotations

import os
import math
import random
import warnings
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Generic,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from numpy.typing import NDArray
from torch import Tensor


# ============================================================
# Type Definitions and Constants
# ============================================================

T = TypeVar("T")
StateType = Union[NDArray[np.floating[Any]], Tuple[int, ...], int]
TensorOrArray = Union[Tensor, NDArray[np.floating[Any]]]

_EPSILON_FLOAT32: float = np.finfo(np.float32).eps


class Transition(NamedTuple):
    """
    Single-step transition data structure.

    Stores the fundamental interaction unit (s, a, r, s', done) in reinforcement learning.
    NamedTuple ensures immutability and memory efficiency through struct-like storage.

    Mathematical Context:
        A transition τ = (s_t, a_t, r_t, s_{t+1}, d_t) represents one step
        of the Markov Decision Process:
        - State s_t from state space S
        - Action a_t from action space A
        - Reward r_t from reward function R(s, a, s')
        - Next state s_{t+1} from transition dynamics P(s'|s, a)
        - Terminal flag d_t ∈ {0, 1}

    Attributes:
        state: Current state s_t, shape (state_dim,)
        action: Executed action a_t ∈ {0, 1, ..., n_actions-1}
        reward: Immediate reward r_t ∈ ℝ
        next_state: Next state s_{t+1}, shape (state_dim,)
        done: Episode termination flag (True indicates terminal state)
    """

    state: NDArray[np.floating[Any]]
    action: int
    reward: float
    next_state: NDArray[np.floating[Any]]
    done: bool


class NetworkType(Enum):
    """
    Q-Network architecture types.

    - STANDARD: Standard MLP architecture, directly outputs Q(s, a) for all actions
    - DUELING: Dueling architecture, separates V(s) and A(s, a) streams
    """

    STANDARD = "standard"
    DUELING = "dueling"


class LossType(Enum):
    """
    Loss function types for TD error optimization.

    - MSE: Mean Squared Error, L = (y - Q)²
      Simple but sensitive to outliers (large TD errors)

    - HUBER: Huber loss (Smooth L1), more robust to outliers
      L = 0.5 * (y - Q)² if |y - Q| < 1 else |y - Q| - 0.5
    """

    MSE = "mse"
    HUBER = "huber"


class ExplorationSchedule(Protocol):
    """Protocol for exploration rate scheduling."""

    def __call__(self, step: int) -> float:
        """Return epsilon value for given training step."""
        ...


# ============================================================
# Configuration
# ============================================================


@dataclass(frozen=False)
class DQNConfig:
    """
    DQN hyperparameter configuration.

    Centralizes all hyperparameters with validation and serialization support.
    Uses dataclass for clean attribute access and automatic __repr__.

    Mathematical Context:
        These hyperparameters control the learning dynamics:
        - γ (gamma): Discount factor in Bellman equation
        - α (learning_rate): Step size for gradient descent
        - ε (epsilon): Exploration probability in ε-greedy policy
        - τ (soft_update_tau): Polyak averaging coefficient if using soft updates

    Attributes:
        state_dim: State space dimensionality
        action_dim: Action space cardinality (number of discrete actions)
        hidden_dims: List of hidden layer dimensions, e.g., [128, 128]
        learning_rate: Adam optimizer learning rate
        gamma: Discount factor γ ∈ [0, 1]
        epsilon_start: Initial exploration rate
        epsilon_end: Final exploration rate
        epsilon_decay: Number of steps for linear decay
        buffer_size: Experience replay buffer capacity
        batch_size: Training batch size
        target_update_freq: Target network synchronization frequency (steps)
        double_dqn: Whether to use Double DQN
        dueling: Whether to use Dueling architecture
        network_type: Network architecture type
        loss_type: Loss function type
        grad_clip: Gradient clipping threshold (None disables clipping)
        device: Compute device ('auto' selects GPU if available)
        seed: Random seed for reproducibility
        min_buffer_size: Minimum buffer size before training starts
        soft_update_tau: Soft update coefficient (None uses hard updates)
    """

    state_dim: int
    action_dim: int
    hidden_dims: List[int] = field(default_factory=lambda: [128, 128])
    learning_rate: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 10000
    buffer_size: int = 100000
    batch_size: int = 64
    target_update_freq: int = 100
    double_dqn: bool = True
    dueling: bool = False
    network_type: NetworkType = NetworkType.STANDARD
    loss_type: LossType = LossType.HUBER
    grad_clip: Optional[float] = 10.0
    device: str = "auto"
    seed: Optional[int] = None
    min_buffer_size: Optional[int] = None
    soft_update_tau: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        self._validate_positive("state_dim", self.state_dim)
        self._validate_positive("action_dim", self.action_dim)

        if not self.hidden_dims:
            raise ValueError("hidden_dims cannot be empty")
        for i, dim in enumerate(self.hidden_dims):
            if dim <= 0:
                raise ValueError(f"hidden_dims[{i}] must be positive, got {dim}")

        self._validate_range("learning_rate", self.learning_rate, 0, 1, exclusive_low=True)
        self._validate_range("gamma", self.gamma, 0, 1)

        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError(
                f"Invalid epsilon range: epsilon_end={self.epsilon_end}, "
                f"epsilon_start={self.epsilon_start}. "
                f"Must satisfy 0 <= epsilon_end <= epsilon_start <= 1"
            )

        self._validate_positive("buffer_size", self.buffer_size)
        self._validate_positive("batch_size", self.batch_size)

        if self.batch_size > self.buffer_size:
            raise ValueError(
                f"batch_size ({self.batch_size}) cannot exceed "
                f"buffer_size ({self.buffer_size})"
            )

        self._validate_positive("target_update_freq", self.target_update_freq)

        if self.dueling:
            object.__setattr__(self, "network_type", NetworkType.DUELING)

        if self.min_buffer_size is None:
            object.__setattr__(self, "min_buffer_size", self.batch_size)

        if self.soft_update_tau is not None:
            self._validate_range("soft_update_tau", self.soft_update_tau, 0, 1)

    def _validate_positive(self, name: str, value: int) -> None:
        """Validate that value is positive."""
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    def _validate_range(
        self,
        name: str,
        value: float,
        low: float,
        high: float,
        exclusive_low: bool = False,
        exclusive_high: bool = False,
    ) -> None:
        """Validate that value is within range."""
        low_ok = value > low if exclusive_low else value >= low
        high_ok = value < high if exclusive_high else value <= high
        if not (low_ok and high_ok):
            low_bracket = "(" if exclusive_low else "["
            high_bracket = ")" if exclusive_high else "]"
            raise ValueError(
                f"{name} must be in {low_bracket}{low}, {high}{high_bracket}, got {value}"
            )

    def get_device(self) -> torch.device:
        """Get compute device."""
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.device)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dims": self.hidden_dims,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "target_update_freq": self.target_update_freq,
            "double_dqn": self.double_dqn,
            "dueling": self.dueling,
            "network_type": self.network_type.value,
            "loss_type": self.loss_type.value,
            "grad_clip": self.grad_clip,
            "device": self.device,
            "seed": self.seed,
            "min_buffer_size": self.min_buffer_size,
            "soft_update_tau": self.soft_update_tau,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DQNConfig":
        """Create configuration from dictionary."""
        data = data.copy()
        if "network_type" in data and isinstance(data["network_type"], str):
            data["network_type"] = NetworkType(data["network_type"])
        if "loss_type" in data and isinstance(data["loss_type"], str):
            data["loss_type"] = LossType(data["loss_type"])
        return cls(**data)


# ============================================================
# Experience Replay Buffers
# ============================================================


class ReplayBuffer:
    """
    Uniform Experience Replay Buffer.

    ============================================================
    Core Idea
    ============================================================
    Stores agent-environment interaction history and samples uniformly for training.
    Breaking temporal correlations makes training more stable by approaching i.i.d.
    assumption required by stochastic gradient descent.

    ============================================================
    Mathematical Foundation
    ============================================================
    Let buffer D = {τ₁, τ₂, ..., τ_N} where τᵢ = (s, a, r, s', done)

    Sampling probability: P(τᵢ) = 1/N (uniform distribution)

    Experience replay transforms RL into supervised learning:
    - Dataset: D
    - Input: state s
    - Target: TD target y = r + γ max_{a'} Q(s', a'; θ⁻)

    ============================================================
    Algorithm Comparison
    ============================================================
    vs. Online Learning:
    + Breaks temporal correlation: consecutive states are highly correlated
    + Sample reuse: each transition can be sampled multiple times
    + More stable: batch gradients have lower variance than single-sample
    - Off-policy: data comes from old policies, potential distribution shift
    - Memory overhead: requires storing large history

    vs. Prioritized Replay (PER):
    + Simple implementation: O(1) storage, O(B) sampling
    + Unbiased: all samples equally weighted
    - Ignores importance: high TD-error samples may be undersampled
    - Inefficient for sparse rewards: successful trajectories diluted

    ============================================================
    Complexity
    ============================================================
    - Space: O(N × d), N = capacity, d = state dimension
    - Push: O(1) amortized (circular overwrite)
    - Sample: O(B), B = batch size
    """

    __slots__ = ("_capacity", "_buffer")

    def __init__(self, capacity: int) -> None:
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum storage capacity. Oldest samples are overwritten when full.

        Raises:
            ValueError: If capacity <= 0
        """
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")
        self._capacity = capacity
        self._buffer: Deque[Transition] = deque(maxlen=capacity)

    @property
    def capacity(self) -> int:
        """Maximum buffer capacity."""
        return self._capacity

    def __len__(self) -> int:
        """Current buffer size."""
        return len(self._buffer)

    def push(
        self,
        state: NDArray[np.floating[Any]],
        action: int,
        reward: float,
        next_state: NDArray[np.floating[Any]],
        done: bool,
    ) -> None:
        """
        Store a single transition.

        Args:
            state: Current state
            action: Executed action
            reward: Immediate reward
            next_state: Next state
            done: Episode termination flag
        """
        self._buffer.append(Transition(state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> Tuple[
        NDArray[np.floating[Any]],
        NDArray[np.int64],
        NDArray[np.floating[Any]],
        NDArray[np.floating[Any]],
        NDArray[np.floating[Any]],
    ]:
        """
        Uniformly sample a mini-batch.

        Args:
            batch_size: Number of samples to draw

        Returns:
            Tuple of (states, actions, rewards, next_states, dones):
            - states: (batch_size, state_dim) float32
            - actions: (batch_size,) int64
            - rewards: (batch_size,) float32
            - next_states: (batch_size, state_dim) float32
            - dones: (batch_size,) float32 (0.0 or 1.0)

        Raises:
            ValueError: If batch_size exceeds buffer size
        """
        if batch_size > len(self._buffer):
            raise ValueError(
                f"Requested batch_size {batch_size} exceeds "
                f"buffer size {len(self._buffer)}"
            )

        batch = random.sample(self._buffer, batch_size)

        states = np.array([t.state for t in batch], dtype=np.float32)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.array([t.next_state for t in batch], dtype=np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return len(self._buffer) >= batch_size

    def clear(self) -> None:
        """Clear all stored transitions."""
        self._buffer.clear()

    def __repr__(self) -> str:
        return f"ReplayBuffer(capacity={self._capacity}, size={len(self)})"


class SumTree:
    """
    Sum Tree data structure for O(log N) prioritized sampling.

    ============================================================
    Core Idea
    ============================================================
    A binary tree where each parent node stores the sum of its children.
    Leaf nodes store priorities, internal nodes store partial sums.
    This enables efficient priority-based sampling in O(log N) time.

    ============================================================
    Mathematical Foundation
    ============================================================
    For a tree with N leaves:
    - Total nodes: 2N - 1
    - Leaf indices: [N-1, 2N-2]
    - Parent of node i: (i - 1) // 2
    - Children of node i: 2i + 1, 2i + 2

    Sampling: Draw uniform random number s ∈ [0, total_priority]
    Traverse tree: go left if s ≤ left_sum, else go right with s -= left_sum

    ============================================================
    Complexity
    ============================================================
    - Space: O(N)
    - Update: O(log N)
    - Sample: O(log N)
    - Get total: O(1)
    """

    __slots__ = ("_capacity", "_tree", "_data", "_write_idx", "_size")

    def __init__(self, capacity: int) -> None:
        """
        Initialize Sum Tree.

        Args:
            capacity: Maximum number of elements (leaves)
        """
        self._capacity = capacity
        self._tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self._data: List[Optional[Transition]] = [None] * capacity
        self._write_idx = 0
        self._size = 0

    @property
    def total_priority(self) -> float:
        """Sum of all priorities (root value)."""
        return float(self._tree[0])

    @property
    def capacity(self) -> int:
        """Maximum capacity."""
        return self._capacity

    def __len__(self) -> int:
        """Current number of stored elements."""
        return self._size

    def add(self, priority: float, data: Transition) -> None:
        """
        Add element with priority.

        Args:
            priority: Priority value (must be positive)
            data: Data to store
        """
        tree_idx = self._write_idx + self._capacity - 1

        self._data[self._write_idx] = data
        self._update(tree_idx, priority)

        self._write_idx = (self._write_idx + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def _update(self, tree_idx: int, priority: float) -> None:
        """Update priority and propagate change to root."""
        delta = priority - self._tree[tree_idx]
        self._tree[tree_idx] = priority

        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self._tree[tree_idx] += delta

    def update_priority(self, tree_idx: int, priority: float) -> None:
        """Update priority at given tree index."""
        self._update(tree_idx, priority)

    def get(self, cumsum: float) -> Tuple[int, float, Transition]:
        """
        Sample element by cumulative sum.

        Args:
            cumsum: Cumulative sum value ∈ [0, total_priority]

        Returns:
            Tuple of (tree_index, priority, data)
        """
        parent_idx = 0

        while True:
            left_child = 2 * parent_idx + 1
            right_child = left_child + 1

            if left_child >= len(self._tree):
                leaf_idx = parent_idx
                break

            if cumsum <= self._tree[left_child]:
                parent_idx = left_child
            else:
                cumsum -= self._tree[left_child]
                parent_idx = right_child

        data_idx = leaf_idx - self._capacity + 1
        data = self._data[data_idx]
        assert data is not None, "Sampled empty slot in SumTree"

        return leaf_idx, self._tree[leaf_idx], data

    def min_priority(self) -> float:
        """Get minimum priority among stored elements."""
        if self._size == 0:
            return 0.0
        start_idx = self._capacity - 1
        priorities = self._tree[start_idx : start_idx + self._size]
        return float(np.min(priorities[priorities > 0]))


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER) Buffer.

    ============================================================
    Core Idea
    ============================================================
    Sample transitions with probability proportional to their "importance"
    (TD error magnitude). High TD-error samples are more informative for
    learning and should be sampled more frequently.

    ============================================================
    Mathematical Foundation
    ============================================================
    Priority assignment:
        pᵢ = |δᵢ| + ε

    where δᵢ is TD error, ε > 0 prevents zero probability.

    Sampling probability (proportional prioritization):
        P(i) = pᵢ^α / Σₖ pₖ^α

    where α ∈ [0, 1] controls prioritization:
    - α = 0: uniform sampling
    - α = 1: fully greedy (only highest priority)

    Importance sampling weights (bias correction):
        wᵢ = (N · P(i))^{-β} / max_j w_j

    where β ∈ [0, 1] controls correction:
    - β = 0: no correction
    - β = 1: full correction

    Typical strategy: β anneals from 0.4 to 1.0 during training.

    ============================================================
    Algorithm Comparison
    ============================================================
    vs. Uniform Replay:
    + Focus on "surprising" samples: high TD error = poor prediction
    + Accelerated learning: ~2x speedup on Atari
    + Sparse reward friendly: rare successes are prioritized
    - Computational overhead: O(log N) sampling with Sum-Tree
    - Hyperparameter sensitive: α, β require tuning
    - Introduces bias: requires importance sampling correction

    ============================================================
    Complexity
    ============================================================
    - Space: O(N)
    - Push: O(log N)
    - Sample: O(B log N)
    - Update priorities: O(B log N)
    """

    __slots__ = (
        "_capacity",
        "_alpha",
        "_beta",
        "_beta_start",
        "_beta_frames",
        "_epsilon",
        "_sum_tree",
        "_max_priority",
        "_frame",
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
        Initialize Prioritized Replay Buffer.

        Args:
            capacity: Maximum capacity
            alpha: Prioritization exponent α ∈ [0, 1]
            beta_start: Initial importance sampling exponent β
            beta_frames: Number of frames to anneal β from beta_start to 1.0
            epsilon: Small constant to prevent zero priority
        """
        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if not 0 <= beta_start <= 1:
            raise ValueError(f"beta_start must be in [0, 1], got {beta_start}")

        self._capacity = capacity
        self._alpha = alpha
        self._beta_start = beta_start
        self._beta = beta_start
        self._beta_frames = beta_frames
        self._epsilon = epsilon

        self._sum_tree = SumTree(capacity)
        self._max_priority: float = 1.0
        self._frame: int = 0

    @property
    def capacity(self) -> int:
        """Maximum capacity."""
        return self._capacity

    @property
    def beta(self) -> float:
        """Current importance sampling exponent."""
        return self._beta

    def __len__(self) -> int:
        """Current buffer size."""
        return len(self._sum_tree)

    def push(
        self,
        state: NDArray[np.floating[Any]],
        action: int,
        reward: float,
        next_state: NDArray[np.floating[Any]],
        done: bool,
    ) -> None:
        """Store transition with maximum priority."""
        transition = Transition(state, action, reward, next_state, done)
        priority = self._max_priority**self._alpha
        self._sum_tree.add(priority, transition)

    def sample(
        self, batch_size: int
    ) -> Tuple[
        NDArray[np.floating[Any]],
        NDArray[np.int64],
        NDArray[np.floating[Any]],
        NDArray[np.floating[Any]],
        NDArray[np.floating[Any]],
        NDArray[np.int64],
        NDArray[np.floating[Any]],
    ]:
        """
        Sample batch with prioritized probabilities.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones, indices, weights):
            - indices: Tree indices for priority updates
            - weights: Importance sampling weights
        """
        buffer_len = len(self._sum_tree)
        if batch_size > buffer_len:
            raise ValueError(f"batch_size {batch_size} exceeds buffer size {buffer_len}")

        self._anneal_beta()

        indices = np.empty(batch_size, dtype=np.int64)
        weights = np.empty(batch_size, dtype=np.float32)
        batch: List[Transition] = []

        total_priority = self._sum_tree.total_priority
        segment_size = total_priority / batch_size

        min_prob = self._sum_tree.min_priority() ** self._alpha / total_priority
        max_weight = (buffer_len * min_prob) ** (-self._beta) if min_prob > 0 else 1.0

        for i in range(batch_size):
            low = segment_size * i
            high = segment_size * (i + 1)
            cumsum = random.uniform(low, high)

            tree_idx, priority, data = self._sum_tree.get(cumsum)

            prob = priority / total_priority
            weight = (buffer_len * prob) ** (-self._beta) / max_weight

            indices[i] = tree_idx
            weights[i] = weight
            batch.append(data)

        states = np.array([t.state for t in batch], dtype=np.float32)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.array([t.next_state for t in batch], dtype=np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones, indices, weights

    def _anneal_beta(self) -> None:
        """Anneal beta from beta_start to 1.0."""
        self._frame += 1
        fraction = min(1.0, self._frame / self._beta_frames)
        self._beta = self._beta_start + fraction * (1.0 - self._beta_start)

    def update_priorities(self, indices: NDArray[np.int64], td_errors: NDArray[np.floating[Any]]) -> None:
        """
        Update priorities based on TD errors.

        Args:
            indices: Tree indices from sampling
            td_errors: Corresponding TD errors
        """
        priorities = (np.abs(td_errors) + self._epsilon) ** self._alpha
        for idx, priority in zip(indices, priorities):
            self._sum_tree.update_priority(int(idx), float(priority))
        self._max_priority = max(self._max_priority, float(np.max(priorities)) ** (1.0 / self._alpha))

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples."""
        return len(self._sum_tree) >= batch_size

    def __repr__(self) -> str:
        return (
            f"PrioritizedReplayBuffer(capacity={self._capacity}, "
            f"size={len(self)}, alpha={self._alpha}, beta={self._beta:.3f})"
        )


# ============================================================
# Neural Network Modules
# ============================================================


def _create_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: List[int],
    activation: Type[nn.Module] = nn.ReLU,
    output_activation: Optional[Type[nn.Module]] = None,
    dropout: float = 0.0,
) -> nn.Sequential:
    """
    Create a Multi-Layer Perceptron (MLP).

    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_dims: List of hidden layer dimensions
        activation: Hidden layer activation function class
        output_activation: Output layer activation (None = no activation)
        dropout: Dropout probability (0 = no dropout)

    Returns:
        nn.Sequential model
    """
    layers: List[nn.Module] = []
    prev_dim = input_dim

    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(activation())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim

    layers.append(nn.Linear(prev_dim, output_dim))
    if output_activation is not None:
        layers.append(output_activation())

    return nn.Sequential(*layers)


def _init_weights_orthogonal(module: nn.Module, gain: float = math.sqrt(2)) -> None:
    """
    Orthogonal initialization for neural network weights.

    Orthogonal initialization maintains gradient scale in deep networks,
    recommended for deep RL networks.

    Mathematical Context:
        For orthogonal matrix W: W^T W = I
        This preserves norm: ||Wx|| = ||x||
        Gradient magnitude remains stable across layers.

    Args:
        module: Module to initialize
        gain: Scaling factor (√2 recommended for ReLU)
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _init_weights_xavier(module: nn.Module) -> None:
    """
    Xavier (Glorot) initialization.

    Maintains variance across layers for sigmoid/tanh activations.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class DQNNetwork(nn.Module):
    """
    Standard DQN Network.

    ============================================================
    Core Idea
    ============================================================
    Multi-layer perceptron mapping states to Q-values for all actions:
        f_θ: ℝ^d → ℝ^{|A|}
        Q(s, a; θ) = f_θ(s)[a]

    ============================================================
    Mathematical Foundation
    ============================================================
    Forward pass:
        h₀ = s
        h_{l+1} = ReLU(W_l h_l + b_l),  l = 0, ..., L-1
        Q(s, ·) = W_L h_L + b_L

    Parameters:
        θ = {W₀, b₀, ..., W_L, b_L}
        |θ| = d·h + (L-1)·h² + h·|A| + Σ(biases)

    ============================================================
    Algorithm Comparison
    ============================================================
    vs. Dueling Network:
    + Simple and direct
    + Fewer parameters
    - Does not separate state value and action advantage
    - All action information mixed together

    vs. Convolutional Network (for Atari):
    + Suitable for low-dimensional states
    + Faster training
    - Not suitable for image inputs
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [128, 128],
    ) -> None:
        """
        Initialize DQN network.

        Args:
            state_dim: State space dimension
            action_dim: Number of actions
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims

        self.network = _create_mlp(state_dim, action_dim, hidden_dims)
        self.apply(_init_weights_orthogonal)

    def forward(self, state: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            state: State tensor, shape (batch_size, state_dim)

        Returns:
            Q-values tensor, shape (batch_size, action_dim)
        """
        return self.network(state)

    def __repr__(self) -> str:
        return (
            f"DQNNetwork(state_dim={self.state_dim}, action_dim={self.action_dim}, "
            f"hidden_dims={self.hidden_dims})"
        )


class DuelingDQNNetwork(nn.Module):
    """
    Dueling DQN Network.

    ============================================================
    Core Idea
    ============================================================
    Decompose Q-function into state value V(s) and action advantage A(s, a):
        Q(s, a) = V(s) + A(s, a) - mean_a A(s, a)

    Intuition:
    - V(s): "How good is this state" - action-independent
    - A(s, a): "How much better is this action than average"

    Separating these allows:
    - State value learns from all action experiences
    - Faster convergence in states where action choice matters little

    ============================================================
    Mathematical Foundation
    ============================================================
    Network structure:
        h = φ(s)                    # Shared feature extraction
        V(s) = V_stream(h)          # Value stream, outputs scalar
        A(s, ·) = A_stream(h)       # Advantage stream, outputs vector

    Aggregation (mean baseline):
        Q(s, a) = V(s) + (A(s, a) - 1/|A| Σ_{a'} A(s, a'))

    Why subtract mean?
    - Identifiability: V and A decomposition is not unique
        Q(s,a) = (V(s) + c) + (A(s,a) - c) for any c
    - Mean subtraction constraint: Σ_a A(s,a) = 0, making V and A unique

    ============================================================
    Algorithm Comparison
    ============================================================
    vs. Standard DQN:
    + Faster convergence: V(s) learns from all actions
    + More stable: value stream gradients smoother
    + ~20% improvement on Atari
    - Slightly more parameters: ~1.5x
    - Extra computation: mean operation

    ============================================================
    Complexity
    ============================================================
    Parameters: O(d·h + h² + h + h·|A|) ≈ 1.5x standard DQN
    Forward: O(|θ|)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [128, 128],
    ) -> None:
        """
        Initialize Dueling network.

        Args:
            state_dim: State space dimension
            action_dim: Number of actions
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims

        if len(hidden_dims) >= 2:
            feature_dims = hidden_dims[:-1]
            stream_input_dim = hidden_dims[-2]
            stream_hidden_dim = hidden_dims[-1]
        else:
            feature_dims = []
            stream_input_dim = state_dim
            stream_hidden_dim = hidden_dims[0] if hidden_dims else 128

        if feature_dims:
            self.feature = _create_mlp(state_dim, stream_input_dim, feature_dims[:-1])
        else:
            self.feature = nn.Identity()
            stream_input_dim = state_dim

        self.value_stream = nn.Sequential(
            nn.Linear(stream_input_dim, stream_hidden_dim),
            nn.ReLU(),
            nn.Linear(stream_hidden_dim, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(stream_input_dim, stream_hidden_dim),
            nn.ReLU(),
            nn.Linear(stream_hidden_dim, action_dim),
        )

        self.apply(_init_weights_orthogonal)

    def forward(self, state: Tensor) -> Tensor:
        """
        Forward pass.

        Aggregation: Q = V + (A - mean(A))

        Args:
            state: State tensor (batch_size, state_dim)

        Returns:
            Q-values (batch_size, action_dim)
        """
        features = self.feature(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

    def __repr__(self) -> str:
        return (
            f"DuelingDQNNetwork(state_dim={self.state_dim}, action_dim={self.action_dim}, "
            f"hidden_dims={self.hidden_dims})"
        )


# ============================================================
# DQN Agent
# ============================================================


class DQNAgent:
    """
    Deep Q-Network Agent.

    ============================================================
    Core Idea
    ============================================================
    Combines deep learning with Q-learning to learn optimal policies
    in high-dimensional state spaces. Uses experience replay and
    target networks for training stability.

    ============================================================
    Mathematical Foundation
    ============================================================
    Training objective: minimize TD error

        L(θ) = E[(y - Q(s, a; θ))²]

    TD target:
        y = r + γ max_{a'} Q(s', a'; θ⁻)  (Standard DQN)
        y = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ⁻)  (Double DQN)

    Policy: ε-greedy
        π(a|s) = { argmax_a Q(s, a; θ), with prob 1 - ε
                 { uniform random,       with prob ε

    ============================================================
    Algorithm Flow
    ============================================================
    1. Initialize: online network θ, target network θ⁻ ← θ, replay buffer D
    2. For each step:
        a. Select action: a = ε-greedy(Q(s, ·; θ))
        b. Execute action: (r, s') = env.step(a)
        c. Store transition: D.push((s, a, r, s', done))
        d. Sample batch: B ~ D
        e. Compute target: y = r + γ (1-done) max_{a'} Q(s', a'; θ⁻)
        f. Update network: θ ← θ - α ∇_θ (y - Q(s, a; θ))²
        g. Sync target: if step % C == 0: θ⁻ ← θ

    ============================================================
    Algorithm Comparison
    ============================================================
    Standard vs Double DQN:
    - Standard: uses max for both selection and evaluation, causes overestimation
    - Double: decouples selection (θ) and evaluation (θ⁻), reduces bias

    vs Policy Gradient:
    + Higher sample efficiency (off-policy replay)
    + Simpler implementation
    - Only supports discrete actions
    - Deterministic policy (requires external exploration)

    ============================================================
    Complexity
    ============================================================
    Per step: O(B × |θ|) forward + backward
    Space: O(N × d + |θ|) buffer + network
    """

    def __init__(self, config: DQNConfig) -> None:
        """
        Initialize DQN agent.

        Args:
            config: Hyperparameter configuration object
        """
        self.config = config
        self.device = config.get_device()

        if config.seed is not None:
            self._set_seed(config.seed)

        self._init_networks()
        self._init_replay_buffer()

        self._epsilon = config.epsilon_start
        self._training_step = 0
        self._update_count = 0

        self._losses: List[float] = []
        self._q_values: List[float] = []
        self._td_errors: List[float] = []

    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def _init_networks(self) -> None:
        """Initialize online and target networks."""
        NetworkClass: Type[nn.Module]
        if self.config.network_type == NetworkType.DUELING:
            NetworkClass = DuelingDQNNetwork
        else:
            NetworkClass = DQNNetwork

        self.q_network: nn.Module = NetworkClass(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dims,
        ).to(self.device)

        self.target_network: nn.Module = NetworkClass(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dims,
        ).to(self.device)

        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        for param in self.target_network.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=self.config.learning_rate,
        )

    def _init_replay_buffer(self) -> None:
        """Initialize experience replay buffer."""
        self.replay_buffer: Union[ReplayBuffer, PrioritizedReplayBuffer] = ReplayBuffer(
            self.config.buffer_size
        )

    @property
    def epsilon(self) -> float:
        """Current exploration rate."""
        return self._epsilon

    @property
    def training_step(self) -> int:
        """Current training step count."""
        return self._training_step

    @property
    def update_count(self) -> int:
        """Current network update count."""
        return self._update_count

    @property
    def losses(self) -> List[float]:
        """Training loss history."""
        return self._losses.copy()

    @property
    def q_values(self) -> List[float]:
        """Average Q-value history."""
        return self._q_values.copy()

    def select_action(
        self,
        state: NDArray[np.floating[Any]],
        training: bool = True,
    ) -> int:
        """
        Select action using ε-greedy policy.

        Args:
            state: Current state
            training: Whether in training mode (enables exploration)

        Returns:
            Action index
        """
        if training and random.random() < self._epsilon:
            return random.randint(0, self.config.action_dim - 1)

        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        return int(q_values.argmax(dim=1).item())

    def store_transition(
        self,
        state: NDArray[np.floating[Any]],
        action: int,
        reward: float,
        next_state: NDArray[np.floating[Any]],
        done: bool,
    ) -> None:
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self) -> Optional[float]:
        """
        Perform one gradient update step.

        Returns:
            Loss value, or None if buffer has insufficient samples
        """
        min_size = self.config.min_buffer_size or self.config.batch_size
        if not self.replay_buffer.is_ready(min_size):
            return None

        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            return self._update_prioritized()
        return self._update_uniform()

    def _update_uniform(self) -> float:
        """Update with uniform replay sampling."""
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.config.batch_size
        )

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        current_q_values = self.q_network(states_t)
        current_q = current_q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.config.double_dqn:
                next_actions = self.q_network(next_states_t).argmax(dim=1)
                next_q = self.target_network(next_states_t).gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze(1)
            else:
                next_q = self.target_network(next_states_t).max(dim=1)[0]

            target_q = rewards_t + self.config.gamma * next_q * (1.0 - dones_t)

        if self.config.loss_type == LossType.HUBER:
            loss = F.smooth_l1_loss(current_q, target_q)
        else:
            loss = F.mse_loss(current_q, target_q)

        self._optimize(loss)

        loss_value = loss.item()
        self._losses.append(loss_value)
        self._q_values.append(current_q.mean().item())

        return loss_value

    def _update_prioritized(self) -> float:
        """Update with prioritized replay sampling."""
        assert isinstance(self.replay_buffer, PrioritizedReplayBuffer)

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            indices,
            weights,
        ) = self.replay_buffer.sample(self.config.batch_size)

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        weights_t = torch.as_tensor(weights, dtype=torch.float32, device=self.device)

        current_q_values = self.q_network(states_t)
        current_q = current_q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.config.double_dqn:
                next_actions = self.q_network(next_states_t).argmax(dim=1)
                next_q = self.target_network(next_states_t).gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze(1)
            else:
                next_q = self.target_network(next_states_t).max(dim=1)[0]

            target_q = rewards_t + self.config.gamma * next_q * (1.0 - dones_t)

        td_errors = target_q - current_q

        if self.config.loss_type == LossType.HUBER:
            element_loss = F.smooth_l1_loss(current_q, target_q, reduction="none")
        else:
            element_loss = F.mse_loss(current_q, target_q, reduction="none")

        loss = (weights_t * element_loss).mean()

        self._optimize(loss)

        self.replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())

        loss_value = loss.item()
        self._losses.append(loss_value)
        self._q_values.append(current_q.mean().item())
        self._td_errors.append(td_errors.abs().mean().item())

        return loss_value

    def _optimize(self, loss: Tensor) -> None:
        """Perform optimization step."""
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
        if self.config.soft_update_tau is not None:
            tau = self.config.soft_update_tau
            for target_param, online_param in zip(
                self.target_network.parameters(),
                self.q_network.parameters(),
            ):
                target_param.data.copy_(
                    tau * online_param.data + (1.0 - tau) * target_param.data
                )
        elif self._update_count % self.config.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self) -> None:
        """Decay exploration rate linearly."""
        self._training_step += 1
        decay_progress = min(1.0, self._training_step / self.config.epsilon_decay)
        self._epsilon = (
            self.config.epsilon_start
            + (self.config.epsilon_end - self.config.epsilon_start) * decay_progress
        )

    def train_step(
        self,
        state: NDArray[np.floating[Any]],
        action: int,
        reward: float,
        next_state: NDArray[np.floating[Any]],
        done: bool,
    ) -> Optional[float]:
        """
        Complete training step: store + update + decay epsilon.

        Args:
            state: Current state
            action: Executed action
            reward: Immediate reward
            next_state: Next state
            done: Termination flag

        Returns:
            Loss value
        """
        self.store_transition(state, action, reward, next_state, done)
        loss = self.update()
        self.decay_epsilon()
        return loss

    def save(self, path: Union[str, Path]) -> None:
        """
        Save model checkpoint.

        Args:
            path: Save path
        """
        checkpoint = {
            "config": self.config.to_dict(),
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self._epsilon,
            "training_step": self._training_step,
            "update_count": self._update_count,
            "losses": self._losses[-1000:],
            "q_values": self._q_values[-1000:],
        }
        torch.save(checkpoint, path)

    def load(self, path: Union[str, Path], load_optimizer: bool = True) -> None:
        """
        Load model checkpoint.

        Args:
            path: Checkpoint path
            load_optimizer: Whether to load optimizer state
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self._epsilon = checkpoint.get("epsilon", self.config.epsilon_end)
        self._training_step = checkpoint.get("training_step", 0)
        self._update_count = checkpoint.get("update_count", 0)
        self._losses = checkpoint.get("losses", [])
        self._q_values = checkpoint.get("q_values", [])

    def get_q_values(self, state: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """
        Get Q-values for state (for debugging and visualization).

        Args:
            state: State

        Returns:
            Q-values for all actions
        """
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.cpu().numpy()[0]

    def set_eval_mode(self) -> None:
        """Set networks to evaluation mode."""
        self.q_network.eval()

    def set_train_mode(self) -> None:
        """Set networks to training mode."""
        self.q_network.train()

    def __repr__(self) -> str:
        return (
            f"DQNAgent(state_dim={self.config.state_dim}, "
            f"action_dim={self.config.action_dim}, "
            f"double_dqn={self.config.double_dqn}, "
            f"dueling={self.config.dueling}, "
            f"device={self.device})"
        )


# ============================================================
# Factory Functions
# ============================================================


def create_dqn_agent(
    state_dim: int,
    action_dim: int,
    double_dqn: bool = True,
    dueling: bool = False,
    prioritized_replay: bool = False,
    **kwargs: Any,
) -> DQNAgent:
    """
    Factory function to create DQN agent.

    Args:
        state_dim: State space dimension
        action_dim: Number of discrete actions
        double_dqn: Use Double DQN
        dueling: Use Dueling architecture
        prioritized_replay: Use Prioritized Experience Replay
        **kwargs: Additional DQNConfig parameters

    Returns:
        Configured DQNAgent
    """
    config = DQNConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        double_dqn=double_dqn,
        dueling=dueling,
        **kwargs,
    )
    agent = DQNAgent(config)

    if prioritized_replay:
        agent.replay_buffer = PrioritizedReplayBuffer(
            capacity=config.buffer_size,
            alpha=kwargs.get("per_alpha", 0.6),
            beta_start=kwargs.get("per_beta_start", 0.4),
            beta_frames=kwargs.get("per_beta_frames", 100000),
        )

    return agent


# ============================================================
# Unit Tests
# ============================================================


def _run_tests() -> bool:
    """
    Run unit tests.

    Returns:
        True if all tests pass
    """
    print("=" * 60)
    print("DQN Core Module Unit Tests")
    print("=" * 60)

    passed = 0
    failed = 0

    print("\n[Test 1] DQNConfig validation")
    try:
        config = DQNConfig(state_dim=4, action_dim=2)
        assert config.state_dim == 4
        assert config.action_dim == 2
        assert config.double_dqn is True

        try:
            DQNConfig(state_dim=0, action_dim=2)
            assert False, "Should raise ValueError"
        except ValueError:
            pass

        try:
            DQNConfig(state_dim=4, action_dim=2, gamma=1.5)
            assert False, "Should raise ValueError"
        except ValueError:
            pass

        print("  PASSED")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1

    print("\n[Test 2] ReplayBuffer")
    try:
        buffer = ReplayBuffer(capacity=100)
        state = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        for i in range(50):
            buffer.push(state, i % 2, float(i), state, False)

        assert len(buffer) == 50
        assert buffer.is_ready(32)

        batch = buffer.sample(32)
        assert len(batch) == 5
        assert batch[0].shape == (32, 4)

        buffer.clear()
        assert len(buffer) == 0

        print("  PASSED")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1

    print("\n[Test 3] SumTree")
    try:
        tree = SumTree(capacity=10)
        state = np.array([1.0, 2.0], dtype=np.float32)

        for i in range(5):
            tree.add(float(i + 1), Transition(state, 0, 1.0, state, False))

        assert len(tree) == 5
        assert abs(tree.total_priority - 15.0) < 1e-6

        # Update leaf node at index: capacity - 1 + data_index = 9 + 4 = 13
        # This changes priority 5 to priority 10, so total becomes 1+2+3+4+10 = 20
        leaf_idx = tree.capacity - 1 + 4
        tree.update_priority(leaf_idx, 10.0)
        assert abs(tree.total_priority - 20.0) < 1e-6

        print("  PASSED")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1

    print("\n[Test 4] PrioritizedReplayBuffer")
    try:
        buffer = PrioritizedReplayBuffer(capacity=100)
        state = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        for i in range(50):
            buffer.push(state, i % 2, float(i), state, False)

        assert len(buffer) == 50
        assert buffer.is_ready(32)

        batch = buffer.sample(32)
        assert len(batch) == 7
        assert batch[0].shape == (32, 4)

        td_errors = np.random.randn(32).astype(np.float32)
        buffer.update_priorities(batch[5], td_errors)

        print("  PASSED")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1

    print("\n[Test 5] DQNNetwork")
    try:
        net = DQNNetwork(state_dim=4, action_dim=2, hidden_dims=[64, 64])
        x = torch.randn(32, 4)
        out = net(x)

        assert out.shape == (32, 2)
        assert not torch.isnan(out).any()

        print("  PASSED")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1

    print("\n[Test 6] DuelingDQNNetwork")
    try:
        net = DuelingDQNNetwork(state_dim=4, action_dim=2, hidden_dims=[64, 64])
        x = torch.randn(32, 4)
        out = net(x)

        assert out.shape == (32, 2)
        assert not torch.isnan(out).any()

        print("  PASSED")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1

    print("\n[Test 7] DQNAgent basic")
    try:
        config = DQNConfig(state_dim=4, action_dim=2, batch_size=32)
        agent = DQNAgent(config)

        state = np.random.randn(4).astype(np.float32)
        action = agent.select_action(state, training=True)
        assert 0 <= action < 2

        for _ in range(64):
            s = np.random.randn(4).astype(np.float32)
            a = random.randint(0, 1)
            r = random.random()
            ns = np.random.randn(4).astype(np.float32)
            d = random.random() > 0.9
            agent.store_transition(s, a, r, ns, d)

        loss = agent.update()
        assert loss is not None
        assert not np.isnan(loss)

        print("  PASSED")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1

    print("\n[Test 8] Double DQN")
    try:
        config = DQNConfig(
            state_dim=4,
            action_dim=2,
            batch_size=32,
            double_dqn=True,
        )
        agent = DQNAgent(config)

        for _ in range(64):
            s = np.random.randn(4).astype(np.float32)
            agent.store_transition(s, 0, 1.0, s, False)

        loss = agent.update()
        assert loss is not None

        print("  PASSED")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1

    print("\n[Test 9] Dueling DQN")
    try:
        config = DQNConfig(
            state_dim=4,
            action_dim=2,
            batch_size=32,
            dueling=True,
        )
        agent = DQNAgent(config)

        for _ in range(64):
            s = np.random.randn(4).astype(np.float32)
            agent.store_transition(s, 0, 1.0, s, False)

        loss = agent.update()
        assert loss is not None

        print("  PASSED")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1

    print("\n[Test 10] Epsilon decay")
    try:
        config = DQNConfig(
            state_dim=4,
            action_dim=2,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=100,
        )
        agent = DQNAgent(config)

        initial_eps = agent.epsilon
        for _ in range(50):
            agent.decay_epsilon()

        assert agent.epsilon < initial_eps
        assert agent.epsilon > config.epsilon_end

        for _ in range(100):
            agent.decay_epsilon()

        assert abs(agent.epsilon - config.epsilon_end) < 0.01

        print("  PASSED")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1

    print("\n[Test 11] Model save/load")
    try:
        import tempfile

        config = DQNConfig(state_dim=4, action_dim=2)
        agent = DQNAgent(config)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            temp_path = f.name

        agent.save(temp_path)

        agent2 = DQNAgent(config)
        agent2.load(temp_path)

        os.remove(temp_path)

        for p1, p2 in zip(
            agent.q_network.parameters(),
            agent2.q_network.parameters(),
        ):
            assert torch.allclose(p1, p2)

        print("  PASSED")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1

    print("\n[Test 12] Factory function")
    try:
        agent = create_dqn_agent(
            state_dim=4,
            action_dim=2,
            double_dqn=True,
            dueling=True,
        )
        assert agent.config.double_dqn
        assert agent.config.dueling

        agent_per = create_dqn_agent(
            state_dim=4,
            action_dim=2,
            prioritized_replay=True,
        )
        assert isinstance(agent_per.replay_buffer, PrioritizedReplayBuffer)

        print("  PASSED")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1

    print("\n[Test 13] Soft update")
    try:
        config = DQNConfig(
            state_dim=4,
            action_dim=2,
            batch_size=32,
            soft_update_tau=0.005,
        )
        agent = DQNAgent(config)

        for _ in range(64):
            s = np.random.randn(4).astype(np.float32)
            agent.store_transition(s, 0, 1.0, s, False)

        online_before = {
            k: v.clone() for k, v in agent.q_network.state_dict().items()
        }
        target_before = {
            k: v.clone() for k, v in agent.target_network.state_dict().items()
        }

        for _ in range(10):
            agent.update()

        for key in online_before:
            online_changed = not torch.allclose(
                online_before[key], agent.q_network.state_dict()[key]
            )
            if online_changed:
                target_changed = not torch.allclose(
                    target_before[key], agent.target_network.state_dict()[key]
                )
                assert target_changed, "Target network should be updated via soft update"
                break

        print("  PASSED")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("All tests passed!")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    import sys

    success = _run_tests()
    sys.exit(0 if success else 1)
