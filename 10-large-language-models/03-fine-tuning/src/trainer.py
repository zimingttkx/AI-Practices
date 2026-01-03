"""
微调训练器实现

本模块提供统一的微调训练接口，支持：
    - 全量微调
    - LoRA微调
    - 梯度累积
    - 混合精度训练
    - 学习率调度

"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader, Dataset
from torch import Tensor


__all__ = [
    "TrainingConfig",
    "FineTuneTrainer",
]


@dataclass
class TrainingConfig:
    """训练配置。

    参数：
        output_dir: 输出目录
        num_epochs: 训练轮数
        batch_size: 批次大小
        gradient_accumulation_steps: 梯度累积步数
        learning_rate: 学习率
        weight_decay: 权重衰减
        warmup_ratio: 预热比例
        max_grad_norm: 梯度裁剪阈值
        fp16: 是否使用混合精度
        logging_steps: 日志记录间隔
        save_steps: 模型保存间隔
        eval_steps: 评估间隔
        seed: 随机种子
    """
    output_dir: str = "./output"
    num_epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    fp16: bool = True
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    seed: int = 42


class FineTuneTrainer:
    """微调训练器。"""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        data_collator: Optional[Callable] = None,
        compute_metrics: Optional[Callable] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator or self._default_collator
        self.compute_metrics = compute_metrics
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # 设置随机种子
        self._set_seed(config.seed)
        
        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)
        
        # 初始化
        self._setup_training()

    def _set_seed(self, seed: int) -> None:
        """设置随机种子。"""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _default_collator(self, batch: List[Dict]) -> Dict[str, Tensor]:
        """默认数据整理函数。"""
        keys = batch[0].keys()
        return {
            key: torch.stack([item[key] for item in batch])
            for key in keys
        }

    def _setup_training(self) -> None:
        """初始化训练组件。"""
        # 数据加载器
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
        )
        
        if self.eval_dataset:
            self.eval_dataloader = DataLoader(
                self.eval_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=self.data_collator,
            )
        
        # 计算总步数
        num_update_steps_per_epoch = len(self.train_dataloader) // self.config.gradient_accumulation_steps
        self.total_steps = num_update_steps_per_epoch * self.config.num_epochs
        self.warmup_steps = int(self.total_steps * self.config.warmup_ratio)
        
        # 优化器
        self.optimizer = self._create_optimizer()
        
        # 学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 混合精度
        self.scaler = torch.cuda.amp.GradScaler() if self.config.fp16 else None

    def _create_optimizer(self) -> AdamW:
        """创建优化器。"""
        # 分离需要权重衰减的参数
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "layernorm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        return AdamW(optimizer_groups, lr=self.config.learning_rate)

    def _create_scheduler(self) -> LambdaLR:
        """创建学习率调度器（带warmup的余弦退火）。"""
        def lr_lambda(current_step: int) -> float:
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            progress = float(current_step - self.warmup_steps) / float(
                max(1, self.total_steps - self.warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        return LambdaLR(self.optimizer, lr_lambda)

    def train(self) -> Dict[str, float]:
        """执行训练。"""
        self.model.train()
        global_step = 0
        total_loss = 0.0
        logging_loss = 0.0
        
        print(f"开始训练:")
        print(f"  - 总步数: {self.total_steps}")
        print(f"  - 预热步数: {self.warmup_steps}")
        print(f"  - 设备: {self.device}")
        
        for epoch in range(self.config.num_epochs):
            for step, batch in enumerate(self.train_dataloader):
                # 移动到设备
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 前向传播
                if self.config.fp16:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**batch)
                        loss = outputs[1] if isinstance(outputs, tuple) else outputs.loss
                        loss = loss / self.config.gradient_accumulation_steps
                else:
                    outputs = self.model(**batch)
                    loss = outputs[1] if isinstance(outputs, tuple) else outputs.loss
                    loss = loss / self.config.gradient_accumulation_steps
                
                # 反向传播
                if self.config.fp16:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                total_loss += loss.item()
                
                # 梯度累积
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # 梯度裁剪
                    if self.config.fp16:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    
                    # 优化器步进
                    if self.config.fp16:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                    
                    # 日志
                    if global_step % self.config.logging_steps == 0:
                        avg_loss = (total_loss - logging_loss) / self.config.logging_steps
                        lr = self.scheduler.get_last_lr()[0]
                        print(f"Step {global_step}/{self.total_steps} | "
                              f"Loss: {avg_loss:.4f} | LR: {lr:.2e}")
                        logging_loss = total_loss
                    
                    # 评估
                    if self.eval_dataset and global_step % self.config.eval_steps == 0:
                        eval_results = self.evaluate()
                        print(f"Eval @ Step {global_step}: {eval_results}")
                        self.model.train()
                    
                    # 保存
                    if global_step % self.config.save_steps == 0:
                        self._save_checkpoint(global_step)
        
        # 最终保存
        self._save_checkpoint(global_step, final=True)
        
        return {"train_loss": total_loss / global_step}

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """评估模型。"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.eval_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            outputs = self.model(**batch)
            loss = outputs[1] if isinstance(outputs, tuple) else outputs.loss
            
            total_loss += loss.item()
            num_batches += 1
        
        results = {"eval_loss": total_loss / num_batches}
        
        if self.compute_metrics:
            results.update(self.compute_metrics(self.model, self.eval_dataloader))
        
        return results

    def _save_checkpoint(self, step: int, final: bool = False) -> None:
        """保存检查点。"""
        checkpoint_dir = os.path.join(
            self.config.output_dir,
            "final" if final else f"checkpoint-{step}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 保存模型
        torch.save(
            self.model.state_dict(),
            os.path.join(checkpoint_dir, "model.pt")
        )
        
        # 保存优化器状态
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": step,
        }, os.path.join(checkpoint_dir, "trainer_state.pt"))
        
        print(f"检查点已保存到: {checkpoint_dir}")
