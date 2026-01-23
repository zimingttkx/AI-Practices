"""
Constitutional AI 实现

基于原则的自我批评与改进机制，通过定义宪法原则引导模型生成安全、有益的输出。

参考文献:
[1] Bai, Y., et al. (2022). Constitutional AI: Harmlessness from AI Feedback.
    https://arxiv.org/abs/2212.08073
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "ConstitutionalPrinciple",
    "ConstitutionalConfig",
    "ConstitutionalAI",
    "SelfCriticTrainer",
    "ConstitutionalBatch",
    "RevisionBatch",
]


@dataclass
class ConstitutionalPrinciple:
    """宪法原则定义"""

    name: str
    critique_request: str
    revision_request: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("原则名称不能为空")
        if not self.critique_request:
            raise ValueError("批评请求不能为空")
        if not self.revision_request:
            raise ValueError("修订请求不能为空")
        if self.weight <= 0:
            raise ValueError(f"权重必须为正数，得到 {self.weight}")


@dataclass
class ConstitutionalConfig:
    """Constitutional AI 配置"""

    principles: list[ConstitutionalPrinciple] = field(default_factory=list)
    max_revisions: int = 3
    critique_temperature: float = 0.7
    revision_temperature: float = 0.9
    min_critique_length: int = 20
    learning_rate: float = 1e-6

    def __post_init__(self) -> None:
        if not self.principles:
            raise ValueError("必须至少定义一个原则")
        if self.max_revisions <= 0:
            raise ValueError(f"max_revisions必须为正数，得到 {self.max_revisions}")
        if not 0 < self.critique_temperature <= 2.0:
            raise ValueError("critique_temperature必须在(0,2]范围内")
        if not 0 < self.revision_temperature <= 2.0:
            raise ValueError("revision_temperature必须在(0,2]范围内")


@dataclass
class ConstitutionalBatch:
    """Constitutional AI 训练批次"""

    prompts: list[str]
    responses: list[str]
    principles: list[ConstitutionalPrinciple]

    def __post_init__(self) -> None:
        if len(self.prompts) != len(self.responses):
            raise ValueError("prompts和responses长度必须相同")
        if not self.principles:
            raise ValueError("必须提供至少一个原则")

    def __len__(self) -> int:
        return len(self.prompts)


@dataclass
class RevisionBatch:
    """修订训练批次"""

    prompts: list[str]
    original_responses: list[str]
    critiques: list[str]
    revised_responses: list[str]

    def __post_init__(self) -> None:
        lengths = [
            len(self.prompts),
            len(self.original_responses),
            len(self.critiques),
            len(self.revised_responses),
        ]
        if len(set(lengths)) != 1:
            raise ValueError("所有列表长度必须相同")

    def __len__(self) -> int:
        return len(self.prompts)


class ConstitutionalAI:
    """Constitutional AI 主类"""

    def __init__(self, config: ConstitutionalConfig):
        self.config = config
        self.critique_history: list[dict[str, str]] = []
        self.revision_history: list[dict[str, str]] = []

    def critique(
        self, prompt: str, response: str, principle: ConstitutionalPrinciple | None = None
    ) -> str:
        """生成批评"""
        if principle is None:
            principle = self._select_principle()

        critique_prompt = self._build_critique_prompt(prompt, response, principle)
        critique = self._generate_critique(critique_prompt)

        self.critique_history.append(
            {
                "prompt": prompt,
                "response": response,
                "principle": principle.name,
                "critique": critique,
            }
        )

        return critique

    def revise(
        self,
        prompt: str,
        response: str,
        critique: str,
        principle: ConstitutionalPrinciple | None = None,
    ) -> str:
        """生成修订版本"""
        if principle is None:
            principle = self._select_principle()

        revision_prompt = self._build_revision_prompt(prompt, response, critique, principle)
        revised = self._generate_revision(revision_prompt)

        self.revision_history.append(
            {
                "prompt": prompt,
                "original": response,
                "critique": critique,
                "revised": revised,
                "principle": principle.name,
            }
        )

        return revised

    def critique_and_revise(
        self, prompt: str, response: str, principle: ConstitutionalPrinciple | None = None
    ) -> tuple[str, str]:
        """批评并修订"""
        critique = self.critique(prompt, response, principle)
        revised = self.revise(prompt, response, critique, principle)
        return critique, revised

    def iterative_revision(
        self, prompt: str, response: str, principle: ConstitutionalPrinciple | None = None
    ) -> list[str]:
        """迭代修订"""
        revisions = [response]
        current = response

        for _i in range(self.config.max_revisions):
            critique = self.critique(prompt, current, principle)
            if not self._needs_revision(critique):
                break

            revised = self.revise(prompt, current, critique, principle)
            revisions.append(revised)
            current = revised

        return revisions

    def train_step(self, batch: ConstitutionalBatch) -> dict[str, float]:
        """训练步骤"""
        total_loss = 0.0
        num_revisions = 0
        critique_lengths = []

        for prompt, response, principle in zip(
            batch.prompts, batch.responses, batch.principles * len(batch)
        ):
            critique = self.critique(prompt, response, principle)
            revised = self.revise(prompt, response, critique, principle)

            loss = self._compute_loss(response, revised)
            total_loss += loss
            num_revisions += 1
            critique_lengths.append(len(critique.split()))

        return {
            "loss": total_loss / num_revisions if num_revisions > 0 else 0.0,
            "num_revisions": num_revisions,
            "avg_critique_length": np.mean(critique_lengths) if critique_lengths else 0.0,
        }

    def _select_principle(self) -> ConstitutionalPrinciple:
        """选择原则（加权随机）"""
        weights = np.array([p.weight for p in self.config.principles])
        weights = weights / weights.sum()
        idx = np.random.choice(len(self.config.principles), p=weights)
        return self.config.principles[idx]

    def _build_critique_prompt(
        self, prompt: str, response: str, principle: ConstitutionalPrinciple
    ) -> str:
        """构建批评提示"""
        return f"""原始问题: {prompt}

模型回答: {response}

请根据以下原则批评上述回答:
{principle.critique_request}

批评:"""

    def _build_revision_prompt(
        self, prompt: str, response: str, critique: str, principle: ConstitutionalPrinciple
    ) -> str:
        """构建修订提示"""
        return f"""原始问题: {prompt}

原始回答: {response}

批评意见: {critique}

请根据以下原则修订回答:
{principle.revision_request}

修订后的回答:"""

    def _generate_critique(self, prompt: str) -> str:
        """生成批评（模拟）"""
        # 实际实现需要调用LLM
        return f"[模拟批评] 基于提示: {prompt[:50]}..."

    def _generate_revision(self, prompt: str) -> str:
        """生成修订（模拟）"""
        # 实际实现需要调用LLM
        return f"[模拟修订] 基于提示: {prompt[:50]}..."

    def _needs_revision(self, critique: str) -> bool:
        """判断是否需要修订"""
        if len(critique.split()) < self.config.min_critique_length:
            return False
        negative_keywords = ["问题", "错误", "不当", "有害", "偏见"]
        return any(kw in critique for kw in negative_keywords)

    def _compute_loss(self, original: str, revised: str) -> float:
        """计算损失（简化版）"""
        # 实际实现需要计算对数似然差异
        return abs(len(revised) - len(original)) / max(len(original), 1)


class SelfCriticTrainer:
    """自我批评训练器"""

    def __init__(self, config: ConstitutionalConfig):
        self.config = config
        self.constitutional_ai = ConstitutionalAI(config)
        self.training_history: list[dict[str, any]] = []

    def generate_critiques(self, responses: list[str], prompts: list[str]) -> list[str]:
        """批量生成批评"""
        critiques = []
        for prompt, response in zip(prompts, responses):
            critique = self.constitutional_ai.critique(prompt, response)
            critiques.append(critique)
        return critiques

    def generate_revisions(
        self, responses: list[str], critiques: list[str], prompts: list[str]
    ) -> list[str]:
        """批量生成修订"""
        revisions = []
        for prompt, response, critique in zip(prompts, responses, critiques):
            revised = self.constitutional_ai.revise(prompt, response, critique)
            revisions.append(revised)
        return revisions

    def train_on_revisions(self, batch: RevisionBatch) -> dict[str, float]:
        """基于修订数据训练"""
        total_loss = 0.0
        improvement_scores = []

        for prompt, original, critique, revised in zip(
            batch.prompts, batch.original_responses, batch.critiques, batch.revised_responses
        ):
            loss = self._compute_revision_loss(original, revised)
            improvement = self._compute_improvement_score(original, revised, critique)

            total_loss += loss
            improvement_scores.append(improvement)

            self.training_history.append(
                {
                    "prompt": prompt,
                    "original": original,
                    "critique": critique,
                    "revised": revised,
                    "loss": loss,
                    "improvement": improvement,
                }
            )

        return {
            "loss": total_loss / len(batch),
            "avg_improvement": np.mean(improvement_scores),
            "num_samples": len(batch),
        }

    def collect_revision_data(
        self, prompts: list[str], responses: list[str]
    ) -> RevisionBatch:
        """收集修订数据"""
        critiques = self.generate_critiques(responses, prompts)
        revisions = self.generate_revisions(responses, critiques, prompts)

        return RevisionBatch(
            prompts=prompts,
            original_responses=responses,
            critiques=critiques,
            revised_responses=revisions,
        )

    def _compute_revision_loss(self, original: str, revised: str) -> float:
        """计算修订损失"""
        # 简化实现：基于长度差异
        return abs(len(revised) - len(original)) / max(len(original), 1)

    def _compute_improvement_score(
        self, original: str, revised: str, critique: str
    ) -> float:
        """计算改进分数"""
        # 简化实现：基于长度和关键词
        length_improvement = len(revised) / max(len(original), 1)
        has_positive_keywords = any(
            kw in revised for kw in ["更好", "改进", "修正", "优化"]
        )
        return length_improvement * (1.5 if has_positive_keywords else 1.0)


# 预定义原则
DEFAULT_PRINCIPLES = [
    ConstitutionalPrinciple(
        name="harmlessness",
        critique_request="请指出回答中可能有害、危险或不当的内容",
        revision_request="请修订回答，使其更加安全、无害",
    ),
    ConstitutionalPrinciple(
        name="helpfulness",
        critique_request="请指出回答中不够有用或信息不足的地方",
        revision_request="请修订回答，使其更加有用和信息丰富",
    ),
    ConstitutionalPrinciple(
        name="honesty",
        critique_request="请指出回答中可能不诚实、误导或虚假的内容",
        revision_request="请修订回答，使其更加诚实和准确",
    ),
]


def create_constitutional_ai(
    principles: list[ConstitutionalPrinciple] | None = None,
    max_revisions: int = 3,
) -> ConstitutionalAI:
    """工厂函数"""
    if principles is None:
        principles = DEFAULT_PRINCIPLES

    config = ConstitutionalConfig(
        principles=principles,
        max_revisions=max_revisions,
    )
    return ConstitutionalAI(config)
