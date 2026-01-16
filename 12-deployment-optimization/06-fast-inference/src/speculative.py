"""
Speculative Decoding: 推测解码实现

============================================================
核心思想 (Core Idea)
============================================================
使用小型 Draft Model 快速生成多个候选 token，然后用大型 Target Model 
并行验证。如果候选 token 被接受，则跳过多次 Target Model 调用，
从而加速推理。

============================================================
关键创新 (Key Innovations)
============================================================
1. Draft-then-Verify: 先猜测后验证的两阶段策略
2. 并行验证: Target Model 一次验证多个候选
3. 拒绝采样: 保证输出分布与 Target Model 完全一致
4. 自适应长度: 根据接受率动态调整推测长度

============================================================
性能提升 (Performance Improvement)
============================================================
- 加速比: 2-3x (取决于 Draft Model 质量)
- 无损质量: 输出分布与原始 Target Model 完全相同
- 内存开销: 需要额外加载 Draft Model

============================================================
参考文献 (References)
============================================================
[1] Leviathan, Y., et al. (2023). Fast Inference from Transformers 
    via Speculative Decoding. ICML 2023.
[2] Chen, C., et al. (2023). Accelerating Large Language Model 
    Decoding with Speculative Sampling. arXiv:2302.01318.
[3] Medusa: https://github.com/FasterDecoding/Medusa
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

__all__ = [
    "SpeculativeConfig",
    "DraftModel",
    "TargetModel",
    "TokenVerifier",
    "SpeculativeDecoder",
    "TreeSpeculation",
    "create_speculative_decoder",
]


# =============================================================================
# 配置
# =============================================================================

@dataclass
class SpeculativeConfig:
    """推测解码配置。
    
    Args:
        num_speculative_tokens: 每次推测的 token 数量 (K)
        max_sequence_length: 最大序列长度
        temperature: 采样温度
        top_p: Top-p 采样参数
        top_k: Top-k 采样参数
        use_tree_speculation: 是否使用树形推测
        tree_width: 树形推测的宽度
        adaptive_k: 是否自适应调整 K
        min_k: 最小推测长度
        max_k: 最大推测长度
        acceptance_threshold: 接受率阈值 (用于自适应)
    """
    num_speculative_tokens: int = 5
    max_sequence_length: int = 4096
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    use_tree_speculation: bool = False
    tree_width: int = 3
    adaptive_k: bool = False
    min_k: int = 1
    max_k: int = 10
    acceptance_threshold: float = 0.8
    
    def __post_init__(self):
        if self.num_speculative_tokens <= 0:
            raise ValueError("num_speculative_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.min_k > self.max_k:
            raise ValueError("min_k must be <= max_k")


# =============================================================================
# 模型接口
# =============================================================================

class DraftModel(ABC):
    """Draft Model 抽象接口。
    
    Draft Model 是一个小型快速模型，用于生成候选 token。
    """
    
    @abstractmethod
    def generate(
        self,
        input_ids: List[int],
        num_tokens: int,
        temperature: float = 1.0
    ) -> Tuple[List[int], List[np.ndarray]]:
        """生成候选 token。
        
        Args:
            input_ids: 输入 token IDs
            num_tokens: 要生成的 token 数量
            temperature: 采样温度
            
        Returns:
            (token_ids, probabilities): 生成的 token IDs 和对应的概率分布
        """
        pass
    
    @abstractmethod
    def get_logits(self, input_ids: List[int]) -> np.ndarray:
        """获取下一个 token 的 logits。
        
        Args:
            input_ids: 输入 token IDs
            
        Returns:
            logits: 形状为 (vocab_size,) 的 logits
        """
        pass


class TargetModel(ABC):
    """Target Model 抽象接口。
    
    Target Model 是大型高质量模型，用于验证候选 token。
    """
    
    @abstractmethod
    def verify(
        self,
        input_ids: List[int],
        candidate_ids: List[int]
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """验证候选 token。
        
        Args:
            input_ids: 原始输入 token IDs
            candidate_ids: 候选 token IDs
            
        Returns:
            (probabilities, next_token_probs): 
                每个位置的概率分布和最后一个位置的概率分布
        """
        pass
    
    @abstractmethod
    def generate_token(
        self,
        input_ids: List[int],
        temperature: float = 1.0
    ) -> Tuple[int, np.ndarray]:
        """生成单个 token。
        
        Args:
            input_ids: 输入 token IDs
            temperature: 采样温度
            
        Returns:
            (token_id, probabilities): 生成的 token ID 和概率分布
        """
        pass


# =============================================================================
# 模拟模型实现 (用于测试)
# =============================================================================

class MockDraftModel(DraftModel):
    """模拟 Draft Model，用于测试。"""
    
    def __init__(self, vocab_size: int = 32000, agreement_rate: float = 0.7):
        self.vocab_size = vocab_size
        self.agreement_rate = agreement_rate
        self._rng = np.random.default_rng(42)
    
    def generate(
        self,
        input_ids: List[int],
        num_tokens: int,
        temperature: float = 1.0
    ) -> Tuple[List[int], List[np.ndarray]]:
        tokens = []
        probs_list = []
        
        for _ in range(num_tokens):
            probs = self._rng.dirichlet(np.ones(self.vocab_size) * 0.1)
            if temperature != 1.0 and temperature > 0:
                logits = np.log(probs + 1e-10) / temperature
                probs = np.exp(logits) / np.sum(np.exp(logits))
            
            token = self._rng.choice(self.vocab_size, p=probs)
            tokens.append(int(token))
            probs_list.append(probs)
        
        return tokens, probs_list
    
    def get_logits(self, input_ids: List[int]) -> np.ndarray:
        return self._rng.standard_normal(self.vocab_size)


class MockTargetModel(TargetModel):
    """模拟 Target Model，用于测试。"""
    
    def __init__(self, vocab_size: int = 32000, agreement_rate: float = 0.7):
        self.vocab_size = vocab_size
        self.agreement_rate = agreement_rate
        self._rng = np.random.default_rng(123)
    
    def verify(
        self,
        input_ids: List[int],
        candidate_ids: List[int]
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        probs_list = []
        
        for i, token in enumerate(candidate_ids):
            probs = self._rng.dirichlet(np.ones(self.vocab_size) * 0.1)
            # 根据配置的一致率决定是否接受
            if self._rng.random() < self.agreement_rate:
                probs[token] = max(probs[token], 0.5)
                probs = probs / probs.sum()
            probs_list.append(probs)
        
        # 最后一个位置的概率
        next_probs = self._rng.dirichlet(np.ones(self.vocab_size) * 0.1)
        
        return probs_list, next_probs
    
    def generate_token(
        self,
        input_ids: List[int],
        temperature: float = 1.0
    ) -> Tuple[int, np.ndarray]:
        probs = self._rng.dirichlet(np.ones(self.vocab_size) * 0.1)
        if temperature != 1.0 and temperature > 0:
            logits = np.log(probs + 1e-10) / temperature
            probs = np.exp(logits) / np.sum(np.exp(logits))
        
        token = self._rng.choice(self.vocab_size, p=probs)
        return int(token), probs


# =============================================================================
# Token Verifier (拒绝采样)
# =============================================================================

class TokenVerifier:
    """Token 验证器：实现拒绝采样算法。
    
    使用拒绝采样确保输出分布与 Target Model 完全一致。
    
    算法:
    1. 对于每个候选 token x，计算接受概率:
       accept_prob = min(1, p_target(x) / p_draft(x))
    2. 以 accept_prob 概率接受 token
    3. 如果拒绝，从修正分布中采样新 token
    """
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        self._rng = np.random.default_rng()
    
    def verify_tokens(
        self,
        draft_tokens: List[int],
        draft_probs: List[np.ndarray],
        target_probs: List[np.ndarray],
        next_token_probs: np.ndarray
    ) -> Tuple[List[int], int]:
        """验证候选 token 并返回接受的 token。
        
        Args:
            draft_tokens: Draft Model 生成的候选 token
            draft_probs: Draft Model 的概率分布
            target_probs: Target Model 的概率分布
            next_token_probs: Target Model 在最后位置的概率分布
            
        Returns:
            (accepted_tokens, num_accepted): 接受的 token 列表和数量
        """
        accepted_tokens = []
        num_accepted = 0
        
        for i, (token, p_draft, p_target) in enumerate(
            zip(draft_tokens, draft_probs, target_probs)
        ):
            # 计算接受概率
            p_d = p_draft[token]
            p_t = p_target[token]
            
            if p_d > 0:
                accept_prob = min(1.0, p_t / p_d)
            else:
                accept_prob = 1.0 if p_t > 0 else 0.0
            
            # 拒绝采样
            if self._rng.random() < accept_prob:
                # 接受
                accepted_tokens.append(token)
                num_accepted += 1
            else:
                # 拒绝：从修正分布中采样
                corrected_token = self._sample_from_residual(
                    p_target, p_draft, token
                )
                accepted_tokens.append(corrected_token)
                # 拒绝后停止验证后续 token
                break
        
        # 如果所有 token 都被接受，从 next_token_probs 采样一个额外 token
        if num_accepted == len(draft_tokens):
            bonus_token = self._sample_from_probs(next_token_probs)
            accepted_tokens.append(bonus_token)
        
        return accepted_tokens, num_accepted
    
    def _sample_from_residual(
        self,
        p_target: np.ndarray,
        p_draft: np.ndarray,
        rejected_token: int
    ) -> int:
        """从修正分布中采样。
        
        修正分布: p_residual(x) = max(0, p_target(x) - p_draft(x)) / Z
        """
        residual = np.maximum(0, p_target - p_draft)
        residual_sum = residual.sum()
        
        if residual_sum > 1e-10:
            residual = residual / residual_sum
            return int(self._rng.choice(len(residual), p=residual))
        else:
            # 回退到 target 分布
            return self._sample_from_probs(p_target)
    
    def _sample_from_probs(self, probs: np.ndarray) -> int:
        """从概率分布中采样。"""
        probs = np.asarray(probs, dtype=np.float64)
        probs = probs / probs.sum()  # 归一化
        return int(self._rng.choice(len(probs), p=probs))


# =============================================================================
# Speculative Decoder 主类
# =============================================================================

@dataclass
class SpeculativeOutput:
    """推测解码输出。"""
    generated_tokens: List[int]
    num_draft_tokens: int
    num_accepted_tokens: int
    num_target_calls: int
    acceptance_rate: float
    speedup_ratio: float
    latency_ms: float


class SpeculativeDecoder:
    """推测解码器：使用 Draft Model 加速 Target Model 推理。
    
    工作流程:
    1. Draft Model 生成 K 个候选 token
    2. Target Model 并行验证所有候选
    3. 使用拒绝采样决定接受哪些 token
    4. 重复直到生成完成
    
    Attributes:
        config: 推测解码配置
        draft_model: Draft Model
        target_model: Target Model
        verifier: Token 验证器
    """
    
    def __init__(
        self,
        config: SpeculativeConfig,
        draft_model: DraftModel,
        target_model: TargetModel
    ):
        self.config = config
        self.draft_model = draft_model
        self.target_model = target_model
        self.verifier = TokenVerifier(config.temperature)
        
        # 自适应 K 的状态
        self._current_k = config.num_speculative_tokens
        self._acceptance_history: List[float] = []
        
        # 统计信息
        self._total_draft_tokens = 0
        self._total_accepted_tokens = 0
        self._total_target_calls = 0
    
    def generate(
        self,
        input_ids: List[int],
        max_tokens: int,
        stop_token_ids: Optional[List[int]] = None
    ) -> SpeculativeOutput:
        """生成 token 序列。
        
        Args:
            input_ids: 输入 token IDs
            max_tokens: 最大生成长度
            stop_token_ids: 停止 token IDs
            
        Returns:
            SpeculativeOutput: 生成结果
        """
        start_time = time.time()
        stop_token_ids = stop_token_ids or []
        
        generated = []
        current_ids = list(input_ids)
        
        num_draft = 0
        num_accepted = 0
        num_target_calls = 0
        
        while len(generated) < max_tokens:
            # 确定本次推测长度
            k = self._get_speculation_length(max_tokens - len(generated))
            
            # Draft Model 生成候选
            draft_tokens, draft_probs = self.draft_model.generate(
                current_ids, k, self.config.temperature
            )
            num_draft += len(draft_tokens)
            
            # Target Model 验证
            target_probs, next_probs = self.target_model.verify(
                current_ids, draft_tokens
            )
            num_target_calls += 1
            
            # 拒绝采样
            accepted, n_accepted = self.verifier.verify_tokens(
                draft_tokens, draft_probs, target_probs, next_probs
            )
            num_accepted += n_accepted
            
            # 更新自适应 K
            if self.config.adaptive_k:
                self._update_k(n_accepted, len(draft_tokens))
            
            # 添加接受的 token
            for token in accepted:
                if token in stop_token_ids:
                    break
                generated.append(token)
                current_ids.append(token)
                if len(generated) >= max_tokens:
                    break
            else:
                continue
            break
        
        # 计算统计信息
        elapsed = time.time() - start_time
        acceptance_rate = num_accepted / num_draft if num_draft > 0 else 0
        # 加速比 = 生成的 token 数 / Target Model 调用次数
        speedup = len(generated) / num_target_calls if num_target_calls > 0 else 1
        
        # 更新全局统计
        self._total_draft_tokens += num_draft
        self._total_accepted_tokens += num_accepted
        self._total_target_calls += num_target_calls
        
        return SpeculativeOutput(
            generated_tokens=generated,
            num_draft_tokens=num_draft,
            num_accepted_tokens=num_accepted,
            num_target_calls=num_target_calls,
            acceptance_rate=acceptance_rate,
            speedup_ratio=speedup,
            latency_ms=elapsed * 1000
        )
    
    def _get_speculation_length(self, remaining: int) -> int:
        """获取本次推测长度。"""
        k = min(self._current_k, remaining)
        return max(1, k)
    
    def _update_k(self, accepted: int, total: int) -> None:
        """自适应更新推测长度 K。"""
        if total == 0:
            return
        
        rate = accepted / total
        self._acceptance_history.append(rate)
        
        # 保留最近 10 次的历史
        if len(self._acceptance_history) > 10:
            self._acceptance_history.pop(0)
        
        avg_rate = np.mean(self._acceptance_history)
        
        # 根据接受率调整 K
        if avg_rate > self.config.acceptance_threshold:
            # 接受率高，增加 K
            self._current_k = min(self._current_k + 1, self.config.max_k)
        elif avg_rate < self.config.acceptance_threshold * 0.5:
            # 接受率低，减少 K
            self._current_k = max(self._current_k - 1, self.config.min_k)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息。"""
        return {
            "total_draft_tokens": self._total_draft_tokens,
            "total_accepted_tokens": self._total_accepted_tokens,
            "total_target_calls": self._total_target_calls,
            "overall_acceptance_rate": (
                self._total_accepted_tokens / self._total_draft_tokens
                if self._total_draft_tokens > 0 else 0
            ),
            "current_k": self._current_k,
            "average_speedup": (
                self._total_accepted_tokens / self._total_target_calls
                if self._total_target_calls > 0 else 1
            ),
        }
    
    def reset_stats(self) -> None:
        """重置统计信息。"""
        self._total_draft_tokens = 0
        self._total_accepted_tokens = 0
        self._total_target_calls = 0
        self._acceptance_history.clear()
        self._current_k = self.config.num_speculative_tokens


# =============================================================================
# Tree Speculation (树形推测)
# =============================================================================

@dataclass
class TreeNode:
    """树形推测的节点。"""
    token_id: int
    probability: float
    children: List["TreeNode"] = field(default_factory=list)
    depth: int = 0
    path: List[int] = field(default_factory=list)


class TreeSpeculation:
    """树形推测：生成多分支候选序列。
    
    相比线性推测，树形推测可以探索更多可能的路径，
    提高至少一条路径被接受的概率。
    
    Attributes:
        config: 推测配置
        draft_model: Draft Model
        width: 每层的分支数
        depth: 树的深度
    """
    
    def __init__(
        self,
        config: SpeculativeConfig,
        draft_model: DraftModel,
        width: int = 3,
        depth: int = 4
    ):
        self.config = config
        self.draft_model = draft_model
        self.width = width
        self.depth = depth
        self._rng = np.random.default_rng()
    
    def generate_tree(
        self,
        input_ids: List[int],
        temperature: float = 1.0
    ) -> TreeNode:
        """生成候选树。
        
        Args:
            input_ids: 输入 token IDs
            temperature: 采样温度
            
        Returns:
            TreeNode: 树的根节点
        """
        root = TreeNode(token_id=-1, probability=1.0, depth=0, path=[])
        self._expand_node(root, input_ids, temperature)
        return root
    
    def _expand_node(
        self,
        node: TreeNode,
        context: List[int],
        temperature: float
    ) -> None:
        """递归展开节点。"""
        if node.depth >= self.depth:
            return
        
        # 获取 logits
        logits = self.draft_model.get_logits(context)
        
        # 应用温度
        if temperature != 1.0 and temperature > 0:
            logits = logits / temperature
        
        # Softmax
        probs = np.exp(logits - np.max(logits))
        probs = probs / probs.sum()
        
        # 选择 top-k 候选
        top_indices = np.argsort(probs)[-self.width:][::-1]
        
        for idx in top_indices:
            child = TreeNode(
                token_id=int(idx),
                probability=float(probs[idx]),
                depth=node.depth + 1,
                path=node.path + [int(idx)]
            )
            node.children.append(child)
            
            # 递归展开
            new_context = context + [int(idx)]
            self._expand_node(child, new_context, temperature)
    
    def get_all_paths(self, root: TreeNode) -> List[List[int]]:
        """获取所有从根到叶的路径。"""
        paths = []
        self._collect_paths(root, [], paths)
        return paths
    
    def _collect_paths(
        self,
        node: TreeNode,
        current_path: List[int],
        all_paths: List[List[int]]
    ) -> None:
        """递归收集路径。"""
        if node.token_id >= 0:
            current_path = current_path + [node.token_id]
        
        if not node.children:
            if current_path:
                all_paths.append(current_path)
        else:
            for child in node.children:
                self._collect_paths(child, current_path, all_paths)
    
    def get_best_path(
        self,
        root: TreeNode,
        target_probs: Dict[Tuple[int, ...], np.ndarray]
    ) -> Tuple[List[int], int]:
        """根据 Target Model 概率选择最佳路径。
        
        Args:
            root: 树的根节点
            target_probs: 每个前缀对应的 Target Model 概率
            
        Returns:
            (best_path, num_accepted): 最佳路径和接受的 token 数
        """
        paths = self.get_all_paths(root)
        
        best_path = []
        best_accepted = 0
        
        for path in paths:
            accepted = self._count_accepted(path, target_probs)
            if accepted > best_accepted:
                best_accepted = accepted
                best_path = path[:accepted]
        
        return best_path, best_accepted
    
    def _count_accepted(
        self,
        path: List[int],
        target_probs: Dict[Tuple[int, ...], np.ndarray]
    ) -> int:
        """计算路径中被接受的 token 数。"""
        accepted = 0
        
        for i, token in enumerate(path):
            prefix = tuple(path[:i])
            if prefix in target_probs:
                probs = target_probs[prefix]
                # 简化的接受判断
                if probs[token] > 0.1:  # 阈值
                    accepted += 1
                else:
                    break
            else:
                break
        
        return accepted


# =============================================================================
# 工厂函数
# =============================================================================

def create_speculative_decoder(
    num_speculative_tokens: int = 5,
    temperature: float = 1.0,
    adaptive_k: bool = False,
    draft_model: Optional[DraftModel] = None,
    target_model: Optional[TargetModel] = None,
    vocab_size: int = 32000,
    **kwargs
) -> SpeculativeDecoder:
    """创建推测解码器的工厂函数。
    
    Args:
        num_speculative_tokens: 每次推测的 token 数量
        temperature: 采样温度
        adaptive_k: 是否自适应调整 K
        draft_model: Draft Model (如果不提供则使用 Mock)
        target_model: Target Model (如果不提供则使用 Mock)
        vocab_size: 词表大小 (用于 Mock Model)
        **kwargs: 其他配置参数
        
    Returns:
        SpeculativeDecoder 实例
        
    Example:
        >>> decoder = create_speculative_decoder(num_speculative_tokens=5)
        >>> output = decoder.generate([1, 2, 3], max_tokens=50)
        >>> print(f"Generated {len(output.generated_tokens)} tokens")
    """
    config = SpeculativeConfig(
        num_speculative_tokens=num_speculative_tokens,
        temperature=temperature,
        adaptive_k=adaptive_k,
        **kwargs
    )
    
    if draft_model is None:
        draft_model = MockDraftModel(vocab_size=vocab_size)
    if target_model is None:
        target_model = MockTargetModel(vocab_size=vocab_size)
    
    return SpeculativeDecoder(config, draft_model, target_model)


def create_tree_speculation(
    draft_model: Optional[DraftModel] = None,
    width: int = 3,
    depth: int = 4,
    vocab_size: int = 32000,
    **kwargs
) -> TreeSpeculation:
    """创建树形推测器的工厂函数。
    
    Args:
        draft_model: Draft Model
        width: 每层分支数
        depth: 树深度
        vocab_size: 词表大小
        **kwargs: 其他配置参数
        
    Returns:
        TreeSpeculation 实例
    """
    config = SpeculativeConfig(**kwargs) if kwargs else SpeculativeConfig()
    
    if draft_model is None:
        draft_model = MockDraftModel(vocab_size=vocab_size)
    
    return TreeSpeculation(config, draft_model, width, depth)


# =============================================================================
# 辅助函数
# =============================================================================

def compute_acceptance_rate(
    draft_probs: List[np.ndarray],
    target_probs: List[np.ndarray],
    draft_tokens: List[int]
) -> float:
    """计算理论接受率。
    
    Args:
        draft_probs: Draft Model 概率分布
        target_probs: Target Model 概率分布
        draft_tokens: Draft 生成的 token
        
    Returns:
        平均接受率
    """
    if not draft_tokens:
        return 0.0
    
    total_accept_prob = 0.0
    
    for token, p_d, p_t in zip(draft_tokens, draft_probs, target_probs):
        if p_d[token] > 0:
            accept_prob = min(1.0, p_t[token] / p_d[token])
        else:
            accept_prob = 1.0 if p_t[token] > 0 else 0.0
        total_accept_prob += accept_prob
    
    return total_accept_prob / len(draft_tokens)


def estimate_speedup(
    acceptance_rate: float,
    k: int,
    draft_cost: float = 0.1,
    target_cost: float = 1.0
) -> float:
    """估算推测解码的加速比。
    
    Args:
        acceptance_rate: 接受率
        k: 推测长度
        draft_cost: Draft Model 相对成本
        target_cost: Target Model 相对成本
        
    Returns:
        估算的加速比
    """
    # 期望接受的 token 数
    expected_accepted = sum(
        acceptance_rate ** i for i in range(1, k + 1)
    )
    
    # 每次迭代的成本
    iteration_cost = k * draft_cost + target_cost
    
    # 基线成本 (每个 token 一次 target 调用)
    baseline_cost = (expected_accepted + 1) * target_cost
    
    return baseline_cost / iteration_cost if iteration_cost > 0 else 1.0
