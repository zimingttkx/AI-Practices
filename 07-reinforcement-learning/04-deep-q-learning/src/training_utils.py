#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DQN Training Utilities and Environment Wrappers

============================================================
Core Idea
============================================================
This module provides a complete training infrastructure for DQN:
1. Training loop abstraction with standardized interfaces
2. Environment wrappers for preprocessing and normalization
3. Real-time monitoring with metrics tracking and visualization
4. Hyperparameter scheduling for learning rate and exploration

============================================================
Module Architecture
============================================================

+----------------+     +-----------------+     +------------------+
|  TrainingConfig|---->|   train_dqn()   |---->| TrainingMetrics  |
+----------------+     +-----------------+     +------------------+
                              |
                              v
                       +-------------+
                       |  DQNAgent   |
                       +-------------+
                              |
                              v
                       +-------------+
                       | Environment |
                       +-------------+

Components:
- TrainingConfig: Training hyperparameter configuration
- TrainingMetrics: Metrics recording and statistics
- train_dqn: Main training function with callbacks
- evaluate_agent: Policy evaluation function
- plot_training_curves: Learning curve visualization
- compare_algorithms: Multi-algorithm comparison utility

============================================================
Mathematical Context
============================================================
Training loop implements the standard DQN algorithm:

    for episode in range(num_episodes):
        s = env.reset()
        for t in range(max_steps):
            a = ε-greedy(Q(s, ·; θ))
            s', r, done = env.step(a)
            D.push((s, a, r, s', done))
            θ ← θ - α∇_θ L(θ)  // if |D| ≥ batch_size
            if done: break

Key metrics monitored:
- Episode return: G_t = Σ_{k=0}^{T-t} γ^k r_{t+k}
- Moving average reward: MA_n = (1/n) Σ_{i=1}^n G_i
- TD loss: L = E[(y - Q(s,a;θ))²]

============================================================
References
============================================================
[1] Mnih, V., et al. (2015). Human-level control through deep reinforcement
    learning. Nature.
[2] OpenAI Spinning Up: Deep RL course implementation guidelines.

Author: zimingttkx
License: MIT
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
    cast,
)

import numpy as np
from numpy.typing import NDArray

try:
    import gymnasium as gym
    from gymnasium import Env
    from gymnasium.spaces import Box, Discrete

    HAS_GYM = True
except ImportError:
    HAS_GYM = False
    warnings.warn(
        "gymnasium not installed. Environment features disabled. "
        "Install with: pip install gymnasium"
    )

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn(
        "matplotlib not installed. Plotting disabled. "
        "Install with: pip install matplotlib"
    )

from dqn_core import DQNAgent, DQNConfig, PrioritizedReplayBuffer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# Type Definitions
# ============================================================


class TrainingCallback(Protocol):
    """Protocol for training callbacks."""

    def __call__(
        self,
        episode: int,
        step: int,
        reward: float,
        loss: Optional[float],
        info: Dict[str, Any],
    ) -> bool:
        """
        Called after each episode.

        Args:
            episode: Current episode number
            step: Total training steps
            reward: Episode return
            loss: Average loss (None if no updates)
            info: Additional information

        Returns:
            False to stop training early, True to continue
        """
        ...


# ============================================================
# Training Configuration
# ============================================================


@dataclass
class TrainingConfig:
    """
    Training hyperparameter configuration.

    ============================================================
    Core Idea
    ============================================================
    Centralizes training loop parameters separate from algorithm hyperparameters.
    This separation allows reusing the same agent configuration across different
    training scenarios (e.g., quick debugging vs. full training).

    Attributes:
        num_episodes: Total number of training episodes
        max_steps_per_episode: Maximum steps per episode (truncation limit)
        eval_frequency: Evaluation frequency in episodes
        eval_episodes: Number of evaluation episodes per checkpoint
        log_frequency: Console logging frequency in episodes
        save_frequency: Model checkpoint frequency in episodes
        checkpoint_dir: Directory for saving checkpoints
        render: Whether to render environment during training
        early_stopping_reward: Target reward for early stopping (None disables)
        early_stopping_episodes: Consecutive episodes required to trigger early stop
        warmup_episodes: Episodes before starting evaluation
        gradient_steps_per_episode: Number of gradient updates per episode (None = step count)
    """

    num_episodes: int = 500
    max_steps_per_episode: int = 500
    eval_frequency: int = 50
    eval_episodes: int = 10
    log_frequency: int = 10
    save_frequency: int = 100
    checkpoint_dir: str = "./checkpoints"
    render: bool = False
    early_stopping_reward: Optional[float] = None
    early_stopping_episodes: int = 10
    warmup_episodes: int = 0
    gradient_steps_per_episode: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.num_episodes <= 0:
            raise ValueError(f"num_episodes must be positive, got {self.num_episodes}")
        if self.max_steps_per_episode <= 0:
            raise ValueError(
                f"max_steps_per_episode must be positive, got {self.max_steps_per_episode}"
            )
        if self.eval_frequency <= 0:
            raise ValueError(f"eval_frequency must be positive, got {self.eval_frequency}")
        if self.eval_episodes <= 0:
            raise ValueError(f"eval_episodes must be positive, got {self.eval_episodes}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "num_episodes": self.num_episodes,
            "max_steps_per_episode": self.max_steps_per_episode,
            "eval_frequency": self.eval_frequency,
            "eval_episodes": self.eval_episodes,
            "log_frequency": self.log_frequency,
            "save_frequency": self.save_frequency,
            "checkpoint_dir": self.checkpoint_dir,
            "render": self.render,
            "early_stopping_reward": self.early_stopping_reward,
            "early_stopping_episodes": self.early_stopping_episodes,
            "warmup_episodes": self.warmup_episodes,
            "gradient_steps_per_episode": self.gradient_steps_per_episode,
        }


# ============================================================
# Training Metrics
# ============================================================


@dataclass
class TrainingMetrics:
    """
    Training metrics recording and analysis.

    ============================================================
    Core Idea
    ============================================================
    Records key metrics during training for:
    1. Monitoring learning progress
    2. Diagnosing training issues
    3. Generating visualization reports
    4. Hyperparameter tuning analysis

    ============================================================
    Mathematical Context
    ============================================================
    Tracked statistics:
    - Episode return: G = Σ_{t=0}^T γ^t r_t
    - Moving average: MA_n(G) = (1/n) Σ_{i=max(1,k-n+1)}^k G_i
    - Standard deviation: σ_n = √(E[(G - μ)²])

    Attributes:
        episode_rewards: Episode cumulative rewards
        episode_lengths: Episode step counts
        losses: Training loss history
        epsilon_history: Exploration rate history
        eval_rewards: Evaluation results (episode, mean, std)
        q_values: Average Q-values (monitors overestimation)
        training_time: Total training time in seconds
        total_steps: Total environment steps
        wall_clock_time: Wall clock timestamps
    """

    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    epsilon_history: List[float] = field(default_factory=list)
    eval_rewards: List[Tuple[int, float, float]] = field(default_factory=list)
    q_values: List[float] = field(default_factory=list)
    training_time: float = 0.0
    total_steps: int = 0
    wall_clock_time: List[float] = field(default_factory=list)

    def add_episode(
        self,
        reward: float,
        length: int,
        epsilon: float,
        loss: Optional[float] = None,
        q_value: Optional[float] = None,
    ) -> None:
        """Record episode metrics."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.epsilon_history.append(epsilon)
        self.total_steps += length
        self.wall_clock_time.append(time.time())

        if loss is not None:
            self.losses.append(loss)
        if q_value is not None:
            self.q_values.append(q_value)

    def add_evaluation(self, episode: int, mean_reward: float, std_reward: float) -> None:
        """Record evaluation results."""
        self.eval_rewards.append((episode, mean_reward, std_reward))

    def get_moving_average(
        self, window: int = 100, data: Optional[List[float]] = None
    ) -> NDArray[np.floating[Any]]:
        """
        Compute moving average of rewards.

        Args:
            window: Window size for averaging
            data: Data to average (defaults to episode_rewards)

        Returns:
            Moving average array
        """
        if data is None:
            data = self.episode_rewards

        if len(data) < window:
            return np.array(data, dtype=np.float64)

        kernel = np.ones(window) / window
        return np.convolve(data, kernel, mode="valid")

    def get_statistics(self, last_n: int = 100) -> Dict[str, float]:
        """
        Get statistics for the last N episodes.

        Args:
            last_n: Number of recent episodes to analyze

        Returns:
            Dictionary of statistics
        """
        rewards = self.episode_rewards[-last_n:] if self.episode_rewards else []
        lengths = self.episode_lengths[-last_n:] if self.episode_lengths else []
        losses = self.losses[-last_n:] if self.losses else []

        stats = {
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "std_reward": float(np.std(rewards)) if rewards else 0.0,
            "max_reward": float(np.max(rewards)) if rewards else 0.0,
            "min_reward": float(np.min(rewards)) if rewards else 0.0,
            "median_reward": float(np.median(rewards)) if rewards else 0.0,
            "mean_length": float(np.mean(lengths)) if lengths else 0.0,
            "mean_loss": float(np.mean(losses)) if losses else 0.0,
            "total_episodes": len(self.episode_rewards),
            "total_steps": self.total_steps,
            "training_time": self.training_time,
        }

        if self.q_values:
            stats["mean_q_value"] = float(np.mean(self.q_values[-last_n:]))

        return stats

    def save(self, path: Union[str, Path]) -> None:
        """Save metrics to JSON file."""
        data = {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "losses": self.losses,
            "epsilon_history": self.epsilon_history,
            "eval_rewards": self.eval_rewards,
            "q_values": self.q_values,
            "training_time": self.training_time,
            "total_steps": self.total_steps,
            "wall_clock_time": self.wall_clock_time,
            "statistics": self.get_statistics(),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TrainingMetrics":
        """Load metrics from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        metrics = cls()
        metrics.episode_rewards = data.get("episode_rewards", [])
        metrics.episode_lengths = data.get("episode_lengths", [])
        metrics.losses = data.get("losses", [])
        metrics.epsilon_history = data.get("epsilon_history", [])
        metrics.eval_rewards = [tuple(e) for e in data.get("eval_rewards", [])]
        metrics.q_values = data.get("q_values", [])
        metrics.training_time = data.get("training_time", 0.0)
        metrics.total_steps = data.get("total_steps", 0)
        metrics.wall_clock_time = data.get("wall_clock_time", [])
        return metrics

    def __repr__(self) -> str:
        stats = self.get_statistics(last_n=100)
        return (
            f"TrainingMetrics(episodes={stats['total_episodes']}, "
            f"mean_reward={stats['mean_reward']:.2f}±{stats['std_reward']:.2f}, "
            f"total_steps={stats['total_steps']})"
        )


# ============================================================
# Training Functions
# ============================================================


def train_dqn(
    agent: DQNAgent,
    env_name: str = "CartPole-v1",
    config: Optional[TrainingConfig] = None,
    verbose: bool = True,
    callbacks: Optional[List[TrainingCallback]] = None,
) -> TrainingMetrics:
    """
    Train DQN agent on gymnasium environment.

    ============================================================
    Core Idea
    ============================================================
    Implements the standard DQN training loop with:
    - Experience collection through environment interaction
    - Batch training from replay buffer
    - Periodic evaluation and checkpointing
    - Early stopping based on performance

    ============================================================
    Algorithm Flow
    ============================================================
    ```
    Initialize: agent, environment, metrics
    for episode in range(num_episodes):
        state = env.reset()
        for step in range(max_steps):
            action = agent.select_action(state, training=True)
            next_state, reward, done = env.step(action)
            agent.train_step(state, action, reward, next_state, done)
            state = next_state
            if done: break
        Record metrics
        if episode % eval_freq == 0: Evaluate and checkpoint
        Check early stopping
    Return metrics
    ```

    Args:
        agent: DQN agent to train
        env_name: Gymnasium environment name
        config: Training configuration (uses defaults if None)
        verbose: Whether to print training logs
        callbacks: Optional list of callback functions

    Returns:
        TrainingMetrics containing training history

    Raises:
        ImportError: If gymnasium is not installed
        ValueError: If environment is incompatible with agent
    """
    if not HAS_GYM:
        raise ImportError(
            "gymnasium is required for training. Install with: pip install gymnasium"
        )

    if config is None:
        config = TrainingConfig()

    render_mode = "human" if config.render else None
    env = gym.make(env_name, render_mode=render_mode)

    _validate_env_compatibility(env, agent)

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    metrics = TrainingMetrics()
    reward_window: Deque[float] = deque(maxlen=100)
    start_time = time.time()

    if verbose:
        _print_training_header(agent, env_name, config)

    best_eval_reward = float("-inf")
    callbacks = callbacks or []

    try:
        for episode in range(config.num_episodes):
            episode_metrics = _run_episode(agent, env, config.max_steps_per_episode)

            metrics.add_episode(
                reward=episode_metrics["reward"],
                length=episode_metrics["length"],
                epsilon=agent.epsilon,
                loss=episode_metrics["avg_loss"],
                q_value=episode_metrics.get("avg_q_value"),
            )
            reward_window.append(episode_metrics["reward"])

            if verbose and (episode + 1) % config.log_frequency == 0:
                _print_episode_log(episode + 1, episode_metrics, reward_window, agent.epsilon)

            if (
                episode + 1 > config.warmup_episodes
                and (episode + 1) % config.eval_frequency == 0
            ):
                eval_mean, eval_std = evaluate_agent(
                    agent, env_name, num_episodes=config.eval_episodes, verbose=False
                )
                metrics.add_evaluation(episode + 1, eval_mean, eval_std)

                if verbose:
                    logger.info(f"  [Eval] Mean: {eval_mean:.2f} ± {eval_std:.2f}")

                if eval_mean > best_eval_reward:
                    best_eval_reward = eval_mean
                    agent.save(checkpoint_dir / "best_model.pt")
                    if verbose:
                        logger.info("  [Best] Saved new best model")

            if (episode + 1) % config.save_frequency == 0:
                agent.save(checkpoint_dir / f"checkpoint_{episode + 1}.pt")

            should_continue = _run_callbacks(
                callbacks, episode + 1, metrics.total_steps, episode_metrics
            )
            if not should_continue:
                if verbose:
                    logger.info(f"\n[Callback] Training stopped by callback at episode {episode + 1}")
                break

            if _check_early_stopping(config, reward_window, verbose):
                break

    except KeyboardInterrupt:
        if verbose:
            logger.info("\n[Interrupted] Training interrupted by user")
    finally:
        env.close()

    metrics.training_time = time.time() - start_time

    if verbose:
        _print_training_summary(metrics, best_eval_reward)

    agent.save(checkpoint_dir / "final_model.pt")
    metrics.save(checkpoint_dir / "training_metrics.json")

    _save_training_config(config, agent.config, checkpoint_dir / "config.json")

    return metrics


def _validate_env_compatibility(env: Any, agent: DQNAgent) -> None:
    """Validate environment compatibility with agent."""
    if not hasattr(env, "observation_space") or not hasattr(env, "action_space"):
        raise ValueError("Environment must have observation_space and action_space")

    obs_space = env.observation_space
    act_space = env.action_space

    if isinstance(obs_space, Box):
        state_dim = int(np.prod(obs_space.shape))
    else:
        raise ValueError(f"Unsupported observation space type: {type(obs_space)}")

    if not isinstance(act_space, Discrete):
        raise ValueError(f"DQN requires discrete action space, got {type(act_space)}")

    if state_dim != agent.config.state_dim:
        raise ValueError(
            f"Environment state dim ({state_dim}) != agent state dim ({agent.config.state_dim})"
        )
    if act_space.n != agent.config.action_dim:
        raise ValueError(
            f"Environment action dim ({act_space.n}) != agent action dim ({agent.config.action_dim})"
        )


def _run_episode(
    agent: DQNAgent, env: Any, max_steps: int
) -> Dict[str, Any]:
    """Run single training episode."""
    state, _ = env.reset()
    state = np.asarray(state, dtype=np.float32)

    episode_reward = 0.0
    episode_loss = 0.0
    episode_q_value = 0.0
    loss_count = 0
    q_count = 0

    for step in range(max_steps):
        action = agent.select_action(state, training=True)

        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = np.asarray(next_state, dtype=np.float32)
        done = terminated or truncated

        loss = agent.train_step(state, action, float(reward), next_state, done)

        if loss is not None:
            episode_loss += loss
            loss_count += 1

        episode_reward += reward
        state = next_state

        if done:
            break

    return {
        "reward": episode_reward,
        "length": step + 1,
        "avg_loss": episode_loss / loss_count if loss_count > 0 else None,
        "avg_q_value": episode_q_value / q_count if q_count > 0 else None,
    }


def _run_callbacks(
    callbacks: List[TrainingCallback],
    episode: int,
    total_steps: int,
    episode_metrics: Dict[str, Any],
) -> bool:
    """Run callbacks and return whether to continue training."""
    info = {
        "length": episode_metrics["length"],
        "avg_loss": episode_metrics.get("avg_loss"),
    }
    for callback in callbacks:
        if not callback(
            episode,
            total_steps,
            episode_metrics["reward"],
            episode_metrics.get("avg_loss"),
            info,
        ):
            return False
    return True


def _check_early_stopping(
    config: TrainingConfig, reward_window: Deque[float], verbose: bool
) -> bool:
    """Check early stopping condition."""
    if config.early_stopping_reward is None:
        return False

    if len(reward_window) >= config.early_stopping_episodes:
        recent_rewards = list(reward_window)[-config.early_stopping_episodes :]
        recent_avg = np.mean(recent_rewards)

        if recent_avg >= config.early_stopping_reward:
            if verbose:
                logger.info(
                    f"\n[Early Stopping] Target reward reached: {recent_avg:.2f} >= {config.early_stopping_reward}"
                )
            return True
    return False


def _print_training_header(
    agent: DQNAgent, env_name: str, config: TrainingConfig
) -> None:
    """Print training header."""
    print("=" * 70)
    print(f"DQN Training: {env_name}")
    print("=" * 70)
    print(f"State dim: {agent.config.state_dim}")
    print(f"Action dim: {agent.config.action_dim}")
    print(f"Network: {'Dueling' if agent.config.dueling else 'Standard'}")
    print(f"Double DQN: {agent.config.double_dqn}")
    print(f"Device: {agent.device}")
    print(f"Episodes: {config.num_episodes}")
    print("=" * 70)


def _print_episode_log(
    episode: int,
    metrics: Dict[str, Any],
    reward_window: Deque[float],
    epsilon: float,
) -> None:
    """Print episode log."""
    avg_reward = np.mean(list(reward_window))
    loss_str = f"{metrics['avg_loss']:.4f}" if metrics["avg_loss"] is not None else "N/A"

    print(
        f"Episode {episode:4d} | "
        f"Reward: {metrics['reward']:7.2f} | "
        f"Avg(100): {avg_reward:7.2f} | "
        f"Loss: {loss_str} | "
        f"ε: {epsilon:.3f} | "
        f"Steps: {metrics['length']:4d}"
    )


def _print_training_summary(metrics: TrainingMetrics, best_eval_reward: float) -> None:
    """Print training summary."""
    stats = metrics.get_statistics()
    print("=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Total episodes: {stats['total_episodes']}")
    print(f"Total steps: {stats['total_steps']}")
    print(f"Training time: {metrics.training_time:.2f} seconds")
    print(f"Final avg reward (100 ep): {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"Best evaluation reward: {best_eval_reward:.2f}")
    print("=" * 70)


def _save_training_config(
    training_config: TrainingConfig,
    agent_config: DQNConfig,
    path: Path,
) -> None:
    """Save combined configuration."""
    config_data = {
        "training": training_config.to_dict(),
        "agent": agent_config.to_dict(),
        "timestamp": datetime.now().isoformat(),
    }
    with open(path, "w") as f:
        json.dump(config_data, f, indent=2)


def evaluate_agent(
    agent: DQNAgent,
    env_name: str,
    num_episodes: int = 10,
    max_steps: int = 1000,
    render: bool = False,
    verbose: bool = True,
    deterministic: bool = True,
) -> Tuple[float, float]:
    """
    Evaluate trained agent performance.

    Uses greedy policy (no exploration) to evaluate true learned behavior.

    Args:
        agent: DQN agent to evaluate
        env_name: Environment name
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        render: Whether to render
        verbose: Whether to print logs
        deterministic: Whether to use deterministic (greedy) policy

    Returns:
        Tuple of (mean_reward, std_reward)
    """
    if not HAS_GYM:
        raise ImportError("gymnasium is required for evaluation")

    render_mode = "human" if render else None
    env = gym.make(env_name, render_mode=render_mode)

    agent.set_eval_mode()
    rewards: List[float] = []

    try:
        for episode in range(num_episodes):
            state, _ = env.reset()
            state = np.asarray(state, dtype=np.float32)
            episode_reward = 0.0

            for _ in range(max_steps):
                action = agent.select_action(state, training=not deterministic)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                state = np.asarray(next_state, dtype=np.float32)
                episode_reward += reward

                if done:
                    break

            rewards.append(episode_reward)

            if verbose:
                print(f"Eval Episode {episode + 1}: Reward = {episode_reward:.2f}")

    finally:
        env.close()
        agent.set_train_mode()

    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))

    if verbose:
        print(f"\nEvaluation Summary: {mean_reward:.2f} ± {std_reward:.2f}")

    return mean_reward, std_reward


# ============================================================
# Visualization Utilities
# ============================================================


def plot_training_curves(
    metrics: TrainingMetrics,
    title: str = "DQN Training Progress",
    window: int = 10,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Optional[Figure]:
    """
    Plot training curves.

    Generates four subplots:
    1. Episode rewards with smoothing
    2. Training loss
    3. Exploration rate decay
    4. Evaluation performance

    Args:
        metrics: Training metrics to visualize
        title: Figure title
        window: Smoothing window size
        save_path: Path to save figure (None = don't save)
        show: Whether to display figure

    Returns:
        matplotlib Figure object if matplotlib available, None otherwise
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib required for plotting. Install with: pip install matplotlib")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    _plot_rewards(axes[0, 0], metrics, window)
    _plot_loss(axes[0, 1], metrics)
    _plot_epsilon(axes[1, 0], metrics)
    _plot_evaluation(axes[1, 1], metrics)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Plot saved to: {save_path}")

    if show:
        plt.show()

    return fig


def _plot_rewards(ax: Any, metrics: TrainingMetrics, window: int) -> None:
    """Plot reward curve."""
    episodes = list(range(1, len(metrics.episode_rewards) + 1))

    ax.plot(episodes, metrics.episode_rewards, alpha=0.3, color="blue", label="Raw")

    if len(metrics.episode_rewards) >= window:
        smoothed = metrics.get_moving_average(window)
        smoothed_x = list(range(window, len(episodes) + 1))
        ax.plot(smoothed_x, smoothed, color="blue", linewidth=2, label=f"MA({window})")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Episode Rewards")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)


def _plot_loss(ax: Any, metrics: TrainingMetrics) -> None:
    """Plot loss curve."""
    if not metrics.losses:
        ax.text(0.5, 0.5, "No loss data", ha="center", va="center", fontsize=12)
        ax.set_title("Training Loss")
        return

    ax.plot(metrics.losses, alpha=0.5, color="red", linewidth=0.5)

    window = min(100, len(metrics.losses) // 10) if len(metrics.losses) > 10 else 1
    if window > 1:
        smoothed = np.convolve(metrics.losses, np.ones(window) / window, mode="valid")
        ax.plot(
            range(window - 1, len(metrics.losses)),
            smoothed,
            color="red",
            linewidth=2,
            label=f"MA({window})",
        )
        ax.legend()

    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)


def _plot_epsilon(ax: Any, metrics: TrainingMetrics) -> None:
    """Plot epsilon decay curve."""
    if not metrics.epsilon_history:
        ax.text(0.5, 0.5, "No epsilon data", ha="center", va="center", fontsize=12)
        ax.set_title("Exploration Rate")
        return

    ax.plot(metrics.epsilon_history, color="green", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.set_title("Exploration Rate Decay")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)


def _plot_evaluation(ax: Any, metrics: TrainingMetrics) -> None:
    """Plot evaluation performance."""
    if not metrics.eval_rewards:
        ax.text(0.5, 0.5, "No evaluation data", ha="center", va="center", fontsize=12)
        ax.set_title("Evaluation Performance")
        return

    eval_episodes = [e[0] for e in metrics.eval_rewards]
    eval_means = [e[1] for e in metrics.eval_rewards]
    eval_stds = [e[2] for e in metrics.eval_rewards]

    ax.errorbar(
        eval_episodes,
        eval_means,
        yerr=eval_stds,
        fmt="o-",
        color="purple",
        capsize=3,
        markersize=6,
        label="Mean ± Std",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Evaluation Performance")
    ax.legend()
    ax.grid(True, alpha=0.3)


def compare_algorithms(
    env_name: str = "CartPole-v1",
    num_episodes: int = 300,
    num_seeds: int = 3,
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, List[TrainingMetrics]]:
    """
    Compare DQN algorithm variants.

    Compares:
    1. Standard DQN
    2. Double DQN
    3. Dueling DQN
    4. Double Dueling DQN

    Args:
        env_name: Environment name
        num_episodes: Training episodes per variant
        num_seeds: Number of random seeds per variant
        save_path: Path to save comparison plot
        verbose: Whether to print progress

    Returns:
        Dictionary mapping algorithm names to lists of TrainingMetrics
    """
    if not HAS_GYM:
        raise ImportError("gymnasium is required")

    env = gym.make(env_name)
    state_dim = int(np.prod(env.observation_space.shape))
    action_dim = env.action_space.n
    env.close()

    algorithms = [
        ("Standard DQN", False, False),
        ("Double DQN", True, False),
        ("Dueling DQN", False, True),
        ("Double Dueling", True, True),
    ]

    results: Dict[str, List[TrainingMetrics]] = {}

    training_config = TrainingConfig(
        num_episodes=num_episodes,
        log_frequency=num_episodes + 1,
        eval_frequency=num_episodes // 3,
        save_frequency=num_episodes + 1,
    )

    for name, double_dqn, dueling in algorithms:
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Training: {name}")
            print("=" * 60)

        results[name] = []

        for seed in range(num_seeds):
            if verbose:
                print(f"\n  Seed {seed + 1}/{num_seeds}")

            config = DQNConfig(
                state_dim=state_dim,
                action_dim=action_dim,
                double_dqn=double_dqn,
                dueling=dueling,
                seed=seed,
                epsilon_decay=5000,
            )
            agent = DQNAgent(config)

            metrics = train_dqn(
                agent,
                env_name=env_name,
                config=training_config,
                verbose=False,
            )
            results[name].append(metrics)

            if verbose:
                final_avg = np.mean(metrics.episode_rewards[-50:])
                print(f"    Final avg (50 ep): {final_avg:.2f}")

    if HAS_MATPLOTLIB and (save_path or verbose):
        _plot_algorithm_comparison(results, env_name, save_path)

    return results


def _plot_algorithm_comparison(
    results: Dict[str, List[TrainingMetrics]],
    env_name: str,
    save_path: Optional[str],
) -> None:
    """Plot algorithm comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.tab10.colors
    window = 10

    for idx, (name, metrics_list) in enumerate(results.items()):
        all_rewards = np.array([m.episode_rewards for m in metrics_list])

        mean_rewards = np.mean(all_rewards, axis=0)
        std_rewards = np.std(all_rewards, axis=0)

        if len(mean_rewards) >= window:
            kernel = np.ones(window) / window
            mean_smooth = np.convolve(mean_rewards, kernel, mode="valid")
            std_smooth = np.convolve(std_rewards, kernel, mode="valid")
            x = np.arange(window - 1, len(mean_rewards))

            axes[0].plot(x, mean_smooth, label=name, color=colors[idx], linewidth=2)
            axes[0].fill_between(
                x,
                mean_smooth - std_smooth,
                mean_smooth + std_smooth,
                color=colors[idx],
                alpha=0.2,
            )

    axes[0].set_xlabel("Episode", fontsize=11)
    axes[0].set_ylabel("Total Reward", fontsize=11)
    axes[0].set_title(f"Learning Curves on {env_name}", fontsize=12)
    axes[0].legend(loc="lower right", fontsize=10)
    axes[0].grid(True, alpha=0.3)

    names = list(results.keys())
    final_means = []
    final_stds = []

    for name in names:
        all_final = [np.mean(m.episode_rewards[-50:]) for m in results[name]]
        final_means.append(np.mean(all_final))
        final_stds.append(np.std(all_final))

    x_pos = np.arange(len(names))
    bars = axes[1].bar(
        x_pos,
        final_means,
        yerr=final_stds,
        capsize=5,
        color=[colors[i] for i in range(len(names))],
        alpha=0.8,
    )
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(names, rotation=15, ha="right")
    axes[1].set_ylabel("Average Reward (Last 50 Episodes)", fontsize=11)
    axes[1].set_title("Final Performance Comparison", fontsize=12)
    axes[1].grid(True, alpha=0.3, axis="y")

    for bar, mean in zip(bars, final_means):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Comparison plot saved to: {save_path}")

    plt.show()


# ============================================================
# Command Line Interface
# ============================================================


def main() -> None:
    """Command line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="DQN Training Utilities",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    train_parser = subparsers.add_parser("train", help="Train DQN agent")
    train_parser.add_argument("--env", type=str, default="CartPole-v1", help="Environment name")
    train_parser.add_argument("--episodes", type=int, default=300, help="Number of episodes")
    train_parser.add_argument("--double", action="store_true", help="Use Double DQN")
    train_parser.add_argument("--dueling", action="store_true", help="Use Dueling DQN")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_parser.add_argument("--render", action="store_true", help="Render environment")
    train_parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Checkpoint directory")

    compare_parser = subparsers.add_parser("compare", help="Compare DQN variants")
    compare_parser.add_argument("--env", type=str, default="CartPole-v1", help="Environment name")
    compare_parser.add_argument("--episodes", type=int, default=300, help="Episodes per variant")
    compare_parser.add_argument("--seeds", type=int, default=3, help="Number of seeds")
    compare_parser.add_argument("--output", type=str, default="comparison.png", help="Output plot path")

    eval_parser = subparsers.add_parser("eval", help="Evaluate trained model")
    eval_parser.add_argument("checkpoint", type=str, help="Checkpoint path")
    eval_parser.add_argument("--env", type=str, default="CartPole-v1", help="Environment name")
    eval_parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    eval_parser.add_argument("--render", action="store_true", help="Render environment")

    plot_parser = subparsers.add_parser("plot", help="Plot training metrics")
    plot_parser.add_argument("metrics_path", type=str, help="Path to metrics JSON file")
    plot_parser.add_argument("--output", type=str, help="Output plot path")
    plot_parser.add_argument("--title", type=str, default="DQN Training", help="Plot title")

    args = parser.parse_args()

    if not HAS_GYM:
        print("Error: gymnasium is required. Install with: pip install gymnasium")
        return

    if args.command == "train":
        env = gym.make(args.env)
        state_dim = int(np.prod(env.observation_space.shape))
        action_dim = env.action_space.n
        env.close()

        config = DQNConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            double_dqn=args.double,
            dueling=args.dueling,
            seed=args.seed,
        )
        agent = DQNAgent(config)

        training_config = TrainingConfig(
            num_episodes=args.episodes,
            render=args.render,
            checkpoint_dir=args.checkpoint_dir,
        )

        metrics = train_dqn(agent, env_name=args.env, config=training_config, verbose=True)
        plot_training_curves(metrics, save_path="training_curves.png")

    elif args.command == "compare":
        compare_algorithms(
            env_name=args.env,
            num_episodes=args.episodes,
            num_seeds=args.seeds,
            save_path=args.output,
            verbose=True,
        )

    elif args.command == "eval":
        import torch

        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

        if "config" in checkpoint:
            config_data = checkpoint["config"]
            if isinstance(config_data, dict):
                config = DQNConfig.from_dict(config_data)
            else:
                config = config_data
        else:
            env = gym.make(args.env)
            state_dim = int(np.prod(env.observation_space.shape))
            action_dim = env.action_space.n
            env.close()
            config = DQNConfig(state_dim=state_dim, action_dim=action_dim)

        agent = DQNAgent(config)
        agent.load(args.checkpoint)

        evaluate_agent(
            agent,
            args.env,
            num_episodes=args.episodes,
            render=args.render,
            verbose=True,
        )

    elif args.command == "plot":
        metrics = TrainingMetrics.load(args.metrics_path)
        plot_training_curves(metrics, title=args.title, save_path=args.output)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
