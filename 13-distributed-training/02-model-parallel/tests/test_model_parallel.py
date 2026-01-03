"""
02-model-parallel 模块测试
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0] + "/src")

from tensor_parallel import (
    TensorParallelConfig,
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    tensor_parallel_split,
    tensor_parallel_gather,
)
from pipeline_parallel import (
    PipelineConfig,
    PipelineStage,
    GPipeScheduler,
    PipeDreamScheduler,
)
from sequence_parallel import (
    SequenceParallelConfig,
    SequenceParallelAttention,
    SequenceParallelLayerNorm,
    scatter_to_sequence_parallel,
    gather_from_sequence_parallel,
)


# ============== Tensor Parallel 测试 ==============

class TestTensorParallelConfig:
    def test_default_config(self):
        config = TensorParallelConfig()
        assert config.world_size == 1
        assert config.rank == 0
        assert config.sequence_parallel is False

    def test_custom_config(self):
        config = TensorParallelConfig(world_size=4, rank=2)
        assert config.world_size == 4
        assert config.rank == 2


class TestTensorParallelSplit:
    def test_single_gpu(self):
        tensor = torch.randn(4, 8)
        result = tensor_parallel_split(tensor, dim=-1)
        assert torch.equal(result, tensor)

    def test_split_with_config(self):
        tensor = torch.randn(4, 8)
        config = TensorParallelConfig(world_size=2, rank=0)
        result = tensor_parallel_split(tensor, dim=-1, config=config)
        assert result.shape == (4, 4)
        assert torch.equal(result, tensor[:, :4])

    def test_split_rank1(self):
        tensor = torch.randn(4, 8)
        config = TensorParallelConfig(world_size=2, rank=1)
        result = tensor_parallel_split(tensor, dim=-1, config=config)
        assert result.shape == (4, 4)
        assert torch.equal(result, tensor[:, 4:])


class TestColumnParallelLinear:
    def test_init(self):
        config = TensorParallelConfig(world_size=2, rank=0)
        layer = ColumnParallelLinear(10, 8, config=config)
        assert layer.weight.shape == (4, 10)
        assert layer.bias.shape == (4,)

    def test_forward_single_gpu(self):
        config = TensorParallelConfig(world_size=1, rank=0)
        layer = ColumnParallelLinear(10, 8, config=config)
        x = torch.randn(2, 10)
        y = layer(x)
        assert y.shape == (2, 8)


class TestRowParallelLinear:
    def test_init(self):
        config = TensorParallelConfig(world_size=2, rank=0)
        layer = RowParallelLinear(8, 10, config=config)
        assert layer.weight.shape == (10, 4)

    def test_forward_single_gpu(self):
        config = TensorParallelConfig(world_size=1, rank=0)
        layer = RowParallelLinear(8, 10, config=config)
        x = torch.randn(2, 8)
        y = layer(x)
        assert y.shape == (2, 10)


class TestVocabParallelEmbedding:
    def test_init(self):
        config = TensorParallelConfig(world_size=2, rank=0)
        emb = VocabParallelEmbedding(1000, 64, config=config)
        assert emb.weight.shape == (500, 64)
        assert emb.vocab_start_idx == 0
        assert emb.vocab_end_idx == 500

    def test_init_rank1(self):
        config = TensorParallelConfig(world_size=2, rank=1)
        emb = VocabParallelEmbedding(1000, 64, config=config)
        assert emb.vocab_start_idx == 500
        assert emb.vocab_end_idx == 1000


# ============== Pipeline Parallel 测试 ==============

class TestPipelineConfig:
    def test_default_config(self):
        config = PipelineConfig()
        assert config.num_stages == 1
        assert config.num_micro_batches == 1

    def test_custom_config(self):
        config = PipelineConfig(num_stages=4, num_micro_batches=8)
        assert config.num_stages == 4
        assert config.num_micro_batches == 8


class TestPipelineStage:
    def test_init(self):
        module = nn.Linear(10, 10)
        stage = PipelineStage(module, stage_id=0, num_stages=4)
        assert stage.is_first_stage is True
        assert stage.is_last_stage is False

    def test_last_stage(self):
        module = nn.Linear(10, 10)
        stage = PipelineStage(module, stage_id=3, num_stages=4)
        assert stage.is_first_stage is False
        assert stage.is_last_stage is True

    def test_forward(self):
        module = nn.Linear(10, 5)
        stage = PipelineStage(module, stage_id=0, num_stages=1)
        x = torch.randn(2, 10)
        y = stage(x)
        assert y.shape == (2, 5)

    def test_clear_cache(self):
        module = nn.Linear(10, 5)
        stage = PipelineStage(module, stage_id=0, num_stages=1)
        stage._input_cache[0] = torch.randn(2, 10)
        stage.clear_cache()
        assert len(stage._input_cache) == 0


class TestGPipeScheduler:
    def test_schedule(self):
        config = PipelineConfig(num_stages=2, num_micro_batches=4, stage_id=0)
        scheduler = GPipeScheduler(config)
        schedule = scheduler.get_schedule()
        
        forwards = [s for s in schedule if s[0] == "forward"]
        backwards = [s for s in schedule if s[0] == "backward"]
        
        assert len(forwards) == 4
        assert len(backwards) == 4
        assert forwards == [("forward", i) for i in range(4)]
        assert backwards == [("backward", i) for i in [3, 2, 1, 0]]


class TestPipeDreamScheduler:
    def test_schedule_first_stage(self):
        config = PipelineConfig(num_stages=4, num_micro_batches=8, stage_id=0)
        scheduler = PipeDreamScheduler(config)
        schedule = scheduler.get_schedule()
        
        assert len(schedule) == 16  # 8 forward + 8 backward


# ============== Sequence Parallel 测试 ==============

class TestSequenceParallelConfig:
    def test_default_config(self):
        config = SequenceParallelConfig()
        assert config.world_size == 1
        assert config.sequence_dim == 1


class TestScatterGather:
    def test_scatter_single_gpu(self):
        config = SequenceParallelConfig(world_size=1, rank=0)
        tensor = torch.randn(2, 8, 16)
        result = scatter_to_sequence_parallel(tensor, config)
        assert torch.equal(result, tensor)

    def test_scatter_multi_gpu(self):
        config = SequenceParallelConfig(world_size=2, rank=0)
        tensor = torch.randn(2, 8, 16)
        result = scatter_to_sequence_parallel(tensor, config)
        assert result.shape == (2, 4, 16)

    def test_gather_single_gpu(self):
        config = SequenceParallelConfig(world_size=1, rank=0)
        tensor = torch.randn(2, 8, 16)
        result = gather_from_sequence_parallel(tensor, config)
        assert torch.equal(result, tensor)


class TestSequenceParallelLayerNorm:
    def test_forward(self):
        config = SequenceParallelConfig()
        ln = SequenceParallelLayerNorm(64, config=config)
        x = torch.randn(2, 8, 64)
        y = ln(x)
        assert y.shape == x.shape


class TestSequenceParallelAttention:
    def test_init(self):
        config = SequenceParallelConfig()
        attn = SequenceParallelAttention(64, 8, config=config)
        assert attn.num_heads == 8
        assert attn.head_dim == 8

    def test_forward(self):
        config = SequenceParallelConfig(world_size=1)
        attn = SequenceParallelAttention(64, 8, config=config)
        x = torch.randn(2, 16, 64)
        y = attn(x)
        assert y.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
