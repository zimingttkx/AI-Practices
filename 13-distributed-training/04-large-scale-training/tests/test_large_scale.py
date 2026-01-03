"""
04-large-scale-training 模块测试
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import os
from pathlib import Path

import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0] + "/src")

from deepspeed_config import (
    DeepSpeedConfig,
    ZeROStage,
    create_deepspeed_config,
    get_zero_config,
    get_optimizer_config,
    get_fp16_config,
    get_bf16_config,
)
from megatron_core import (
    MegatronConfig,
    MegatronParallelState,
)
from checkpoint_utils import (
    CheckpointConfig,
    DistributedCheckpointer,
)


# ============== DeepSpeed Config 测试 ==============

class TestDeepSpeedConfig:
    def test_default_config(self):
        config = DeepSpeedConfig()
        assert config.train_batch_size == 32
        assert config.zero_stage == 2
        assert config.fp16_enabled is True

    def test_custom_config(self):
        config = DeepSpeedConfig(
            train_batch_size=64,
            zero_stage=3,
            bf16_enabled=True,
            fp16_enabled=False,
        )
        assert config.train_batch_size == 64
        assert config.zero_stage == 3
        assert config.bf16_enabled is True


class TestZeROStage:
    def test_enum_values(self):
        assert ZeROStage.DISABLED == 0
        assert ZeROStage.OPTIMIZER_STATES == 1
        assert ZeROStage.GRADIENTS == 2
        assert ZeROStage.PARAMETERS == 3


class TestGetZeROConfig:
    def test_stage2(self):
        config = get_zero_config(stage=2)
        assert config["stage"] == 2
        assert "offload_optimizer" not in config

    def test_stage3(self):
        config = get_zero_config(stage=3)
        assert config["stage"] == 3
        assert "stage3_prefetch_bucket_size" in config

    def test_with_offload(self):
        config = get_zero_config(stage=2, offload_optimizer=True)
        assert "offload_optimizer" in config
        assert config["offload_optimizer"]["device"] == "cpu"


class TestGetOptimizerConfig:
    def test_default(self):
        config = get_optimizer_config()
        assert config["type"] == "AdamW"
        assert config["params"]["lr"] == 1e-4

    def test_custom(self):
        config = get_optimizer_config(
            optimizer_type="Adam",
            learning_rate=5e-5,
        )
        assert config["type"] == "Adam"
        assert config["params"]["lr"] == 5e-5


class TestGetFP16Config:
    def test_enabled(self):
        config = get_fp16_config(enabled=True)
        assert config["enabled"] is True
        assert "loss_scale" in config


class TestGetBF16Config:
    def test_enabled(self):
        config = get_bf16_config(enabled=True)
        assert config["enabled"] is True


class TestCreateDeepSpeedConfig:
    def test_full_config(self):
        ds_config = DeepSpeedConfig(
            train_batch_size=64,
            zero_stage=2,
            fp16_enabled=True,
        )
        config = create_deepspeed_config(ds_config)
        
        assert config["train_batch_size"] == 64
        assert config["zero_optimization"]["stage"] == 2
        assert config["fp16"]["enabled"] is True


# ============== Megatron Config 测试 ==============

class TestMegatronConfig:
    def test_default_config(self):
        config = MegatronConfig()
        assert config.tensor_model_parallel_size == 1
        assert config.pipeline_model_parallel_size == 1
        assert config.data_parallel_size == 1

    def test_custom_config(self):
        config = MegatronConfig(
            tensor_model_parallel_size=4,
            pipeline_model_parallel_size=2,
        )
        assert config.tensor_model_parallel_size == 4
        assert config.pipeline_model_parallel_size == 2


class TestMegatronParallelState:
    def test_singleton(self):
        state1 = MegatronParallelState()
        state2 = MegatronParallelState()
        assert state1 is state2

    def test_default_values(self):
        state = MegatronParallelState()
        assert state.tensor_model_parallel_rank == 0
        assert state.tensor_model_parallel_world_size == 1


# ============== Checkpoint 测试 ==============

class TestCheckpointConfig:
    def test_default_config(self):
        config = CheckpointConfig()
        assert config.save_interval == 1000
        assert config.keep_last_n == 3

    def test_custom_config(self):
        config = CheckpointConfig(
            save_dir="/tmp/ckpt",
            keep_last_n=5,
        )
        assert config.save_dir == "/tmp/ckpt"
        assert config.keep_last_n == 5


class TestDistributedCheckpointer:
    def test_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_dir=tmpdir)
            checkpointer = DistributedCheckpointer(config)
            assert checkpointer.save_dir.exists()

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_dir=tmpdir, use_distributed=False)
            checkpointer = DistributedCheckpointer(config)
            
            model = nn.Linear(10, 5)
            optimizer = torch.optim.Adam(model.parameters())
            
            # Save
            ckpt_path = checkpointer.save(model, optimizer, step=100)
            assert os.path.exists(ckpt_path)
            
            # Load
            new_model = nn.Linear(10, 5)
            new_optimizer = torch.optim.Adam(new_model.parameters())
            checkpoint = checkpointer.load(ckpt_path, new_model, new_optimizer)
            
            assert checkpoint["step"] == 100

    def test_get_latest_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_dir=tmpdir, use_distributed=False)
            checkpointer = DistributedCheckpointer(config)
            
            model = nn.Linear(10, 5)
            checkpointer.save(model, step=100)
            checkpointer.save(model, step=200)
            
            latest = checkpointer.get_latest_checkpoint()
            assert "200" in latest


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
