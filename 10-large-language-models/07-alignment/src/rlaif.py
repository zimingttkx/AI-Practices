"""
RLAIF (Reinforcement Learning from AI Feedback)

使用 AI 反馈代替人类反馈进行强化学习对齐。

参考文献:
[1] Lee, H., et al. (2023). RLAIF: Scaling Reinforcement Learning from Human Feedback
    with AI Feedback. https://arxiv.org/abs/2309.00267
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


__all__ = [
    "RLAIFConfig",
    "AIFeedbackGenerator",
    "RLAIFTrainer",
    "create_rlaif_trainer",
]


@dataclass
class RLAIFConfig:
    """RLAIF 配置"""

    critic_model: str = "gpt-4"
    num_samples: int = 4
    temperature: float = 0.7
    max_feedback_length: int = 500
    preference_threshold: float = 0.6
    learning_rate: float = 1e-6

    def __post_init__(self) -> None:
        if self.num_samples < 2:
            raise ValueError(f"num_samples必须>=2，得到 {self.num_samples}")
        if not 0 < self.temperature <= 2.0:
            raise ValueError("temperature必须在(0,2]范围内")
        if not 0 < self.preference_threshold < 1.0:
            raise ValueError("preference_threshold必须在(0,1)范围内")


class AIFeedbackGenerator:
    """AI 反馈生成器"""

    def __init__(self, config: RLAIFConfig):
        self.config = config
        self.feedback_history: list[dict[str, str]] = []

    def generate_preferences(
        self, prompt: str, responses: list[str]
    ) -> tuple[int, int]:
        """生成偏好对 (chosen_idx, rejected_idx)"""
        if len(responses) < 2:
            raise ValueError("至少需要2个响应")

        scores = [self._score_response(prompt, resp) for resp in responses]
        sorted_indices = np.argsort(scores)[::-1]

        chosen_idx = sorted_indices[0]
        rejected_idx = sorted_indices[-1]

        self.feedback_history.append(
            {
                "prompt": prompt,
                "chosen": responses[chosen_idx],
                "rejected": responses[rejected_idx],
                "chosen_score": scores[chosen_idx],
                "rejected_score": scores[rejected_idx],
            }
        )

        return int(chosen_idx), int(rejected_idx)

    def generate_critique(self, prompt: str, response: str) -> str:
        """生成批评"""
        critique_prompt = f"""请评价以下回答的质量:

问题: {prompt}
回答: {response}

请从以下维度评价:
1. 准确性
2. 有用性
3. 安全性
4. 完整性

批评:"""
        critique = self._generate_text(critique_prompt)
        return critique

    def generate_score(self, prompt: str, response: str) -> float:
        """生成评分 [0, 1]"""
        return self._score_response(prompt, response)

    def batch_generate_preferences(
        self, prompts: list[str], responses_list: list[list[str]]
    ) -> list[tuple[int, int]]:
        """批量生成偏好"""
        if len(prompts) != len(responses_list):
            raise ValueError("prompts和responses_list长度必须相同")

        preferences = []
        for prompt, responses in zip(prompts, responses_list):
            chosen, rejected = self.generate_preferences(prompt, responses)
            preferences.append((chosen, rejected))

        return preferences

    def _score_response(self, prompt: str, response: str) -> float:
        """评分响应（模拟）"""
        # 实际实现需要调用LLM
        score = len(response) / 1000.0
        score += 0.5 if "准确" in response or "有用" in response else 0.0
        return min(score, 1.0)

    def _generate_text(self, prompt: str) -> str:
        """生成文本（模拟）"""
        # 实际实现需要调用LLM
        return f"[AI反馈] 基于提示: {prompt[:50]}..."


class RLAIFTrainer:
    """RLAIF 训练器"""

    def __init__(self, config: RLAIFConfig):
        self.config = config
        self.feedback_generator = AIFeedbackGenerator(config)
        self.training_history: list[dict] = []

    def collect_ai_preferences(
        self, prompts: list[str], num_responses_per_prompt: int = 4
    ) -> list[dict[str, str]]:
        """收集AI偏好数据"""
        preference_data = []

        for prompt in prompts:
            responses = self._generate_responses(prompt, num_responses_per_prompt)
            chosen_idx, rejected_idx = self.feedback_generator.generate_preferences(
                prompt, responses
            )

            preference_data.append(
                {
                    "prompt": prompt,
                    "chosen": responses[chosen_idx],
                    "rejected": responses[rejected_idx],
                }
            )

        return preference_data

    def train_reward_model(
        self, preference_data: list[dict[str, str]]
    ) -> dict[str, float]:
        """训练奖励模型"""
        total_loss = 0.0
        num_correct = 0

        for data in preference_data:
            chosen_score = self.feedback_generator.generate_score(
                data["prompt"], data["chosen"]
            )
            rejected_score = self.feedback_generator.generate_score(
                data["prompt"], data["rejected"]
            )

            loss = max(0, rejected_score - chosen_score + 0.1)
            total_loss += loss

            if chosen_score > rejected_score:
                num_correct += 1

        return {
            "loss": total_loss / len(preference_data),
            "accuracy": num_correct / len(preference_data),
            "num_samples": len(preference_data),
        }

    def train_policy(
        self, prompts: list[str], num_iterations: int = 100
    ) -> dict[str, float]:
        """训练策略模型"""
        total_reward = 0.0
        num_samples = 0

        for iteration in range(num_iterations):
            for prompt in prompts:
                response = self._generate_response(prompt)
                reward = self.feedback_generator.generate_score(prompt, response)

                total_reward += reward
                num_samples += 1

                self.training_history.append(
                    {
                        "iteration": iteration,
                        "prompt": prompt,
                        "response": response,
                        "reward": reward,
                    }
                )

        return {
            "avg_reward": total_reward / num_samples,
            "num_iterations": num_iterations,
            "num_samples": num_samples,
        }

    def train_step(
        self, prompts: list[str], responses: list[str]
    ) -> dict[str, float]:
        """单步训练"""
        if len(prompts) != len(responses):
            raise ValueError("prompts和responses长度必须相同")

        total_reward = 0.0
        critiques = []

        for prompt, response in zip(prompts, responses):
            reward = self.feedback_generator.generate_score(prompt, response)
            critique = self.feedback_generator.generate_critique(prompt, response)

            total_reward += reward
            critiques.append(critique)

        return {
            "avg_reward": total_reward / len(prompts),
            "num_samples": len(prompts),
            "avg_critique_length": np.mean([len(c.split()) for c in critiques]),
        }

    def _generate_responses(self, prompt: str, num_responses: int) -> list[str]:
        """生成多个响应（模拟）"""
        return [f"响应{i}: {prompt[:20]}..." for i in range(num_responses)]

    def _generate_response(self, prompt: str) -> str:
        """生成单个响应（模拟）"""
        return f"响应: {prompt[:30]}..."


def create_rlaif_trainer(
    critic_model: str = "gpt-4",
    num_samples: int = 4,
    temperature: float = 0.7,
) -> RLAIFTrainer:
    """工厂函数"""
    config = RLAIFConfig(
        critic_model=critic_model,
        num_samples=num_samples,
        temperature=temperature,
    )
    return RLAIFTrainer(config)
