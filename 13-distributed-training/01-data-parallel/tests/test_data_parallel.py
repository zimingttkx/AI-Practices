"""
01-data-parallel 模块测试
"""

import pytest
import torch
import torch.nn as nn
import torch.distributed as dist
from unittest.mock import MagicMock, patch

# 导入被测试模块
import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0] + "/src")

from ddp import (
    DDPConfig,
    DDPTrainer,
    get_rank,
    get_world_size,
    is_main_process,
    get_local_rank,
    all_reduce_tensor,
    all_gather_tensor,
    broadcast_tensor,
    reduce_dict,
)
from fsdp import (
    FSDPConfig,
    ShardingStrategy,
    get_fsdp_wrap_policy,
)
from zero import (
    ZeROConfig,
    ZeROStage,
    PartitionedParameter,
)


# ============== DDP 测试 ==============

class TestDDPConfig:
    """DDPConfig 测试"""
    
    def test_default_config(self):
        config = DDPConfig()
        assert config.backend == "nccl"
        assert config.init_method == "env://"
        assert config.find_unused_parameters is False
        assert config.broadcast_buffers is True
    
    def test_custom_config(self):
        config = DDPConfig(
            backend="gloo",
            find_unused_parameters=True,
            bucket_cap_mb=50,
        )
        assert config.backend == "gloo"
        assert config.find_unused_parameters is True
        assert config.bucket_cap_mb == 50


class TestDDPHelpers:
    """DDP 辅助函数测试"""
    
    def test_get_rank_not_initialized(self):
        assert get_rank() == 0
    
    def test_get_world_size_not_initialized(self):
        assert get_world_size() == 1
    
    def test_is_main_process_not_initialized(self):
        assert is_main_process() is True
    
    def test_get_local_rank_default(self):
        assert get_local_rank() == 0
    
    @patch.dict("os.environ", {"LOCAL_RANK": "2"})
    def test_get_local_rank_from_env(self):
        assert get_local_rank() == 2


class TestDDPTrainer:
    """DDPTrainer 测试"""
    
    @pytest.fixture
    def simple_model(self):
        return nn.Linear(10, 5)
    
    def test_trainer_init(self, simple_model):
        trainer = DDPTrainer(simple_model)
        assert trainer.rank == 0
        assert trainer.world_size == 1
        assert trainer._is_wrapped is False
    
    def test_get_model_unwrapped(self, simple_model):
        trainer = DDPTrainer(simple_model)
        model = trainer.get_model()
        assert model is simple_model
    
    def test_get_raw_model_unwrapped(self, simple_model):
        trainer = DDPTrainer(simple_model)
        model = trainer.get_raw_model()
        assert model is simple_model


class TestAllReduceTensor:
    """all_reduce_tensor 测试"""
    
    def test_not_initialized(self):
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = all_reduce_tensor(tensor)
        assert torch.equal(result, tensor)


class TestAllGatherTensor:
    """all_gather_tensor 测试"""
    
    def test_not_initialized(self):
        tensor = torch.tensor([1.0, 2.0])
        result = all_gather_tensor(tensor)
        assert len(result) == 1
        assert torch.equal(result[0], tensor)


class TestBroadcastTensor:
    """broadcast_tensor 测试"""
    
    def test_not_initialized(self):
        tensor = torch.tensor([1.0, 2.0])
        result = broadcast_tensor(tensor)
        assert torch.equal(result, tensor)


class TestReduceDict:
    """reduce_dict 测试"""
    
    def test_not_initialized(self):
        input_dict = {"a": torch.tensor(1.0), "b": torch.tensor(2.0)}
        result = reduce_dict(input_dict)
        assert torch.equal(result["a"], input_dict["a"])
        assert torch.equal(result["b"], input_dict["b"])


# ============== FSDP 测试 ==============

class TestFSDPConfig:
    """FSDPConfig 测试"""
    
    def test_default_config(self):
        config = FSDPConfig()
        assert config.sharding_strategy == ShardingStrategy.FULL_SHARD
        assert config.cpu_offload is False
        assert config.mixed_precision is False
        assert config.auto_wrap_policy == "size_based"
    
    def test_custom_config(self):
        config = FSDPConfig(
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            cpu_offload=True,
            min_num_params=50000,
        )
        assert config.sharding_strategy == ShardingStrategy.SHARD_GRAD_OP
        assert config.cpu_offload is True
        assert config.min_num_params == 50000


class TestShardingStrategy:
    """ShardingStrategy 测试"""
    
    def test_enum_values(self):
        assert ShardingStrategy.FULL_SHARD.value == 1
        assert ShardingStrategy.SHARD_GRAD_OP.value == 2
        assert ShardingStrategy.NO_SHARD.value == 3
        assert ShardingStrategy.HYBRID_SHARD.value == 4


class TestGetFSDPWrapPolicy:
    """get_fsdp_wrap_policy 测试"""
    
    def test_none_policy(self):
        config = FSDPConfig(auto_wrap_policy="none")
        policy = get_fsdp_wrap_policy(config)
        assert policy is None


# ============== ZeRO 测试 ==============

class TestZeROConfig:
    """ZeROConfig 测试"""
    
    def test_default_config(self):
        config = ZeROConfig()
        assert config.stage == ZeROStage.OPTIMIZER
        assert config.overlap_comm is True
        assert config.cpu_offload is False
    
    def test_custom_config(self):
        config = ZeROConfig(
            stage=ZeROStage.PARAMETERS,
            cpu_offload=True,
            reduce_bucket_size=100_000_000,
        )
        assert config.stage == ZeROStage.PARAMETERS
        assert config.cpu_offload is True
        assert config.reduce_bucket_size == 100_000_000


class TestZeROStage:
    """ZeROStage 测试"""
    
    def test_enum_values(self):
        assert ZeROStage.DISABLED == 0
        assert ZeROStage.OPTIMIZER == 1
        assert ZeROStage.GRADIENTS == 2
        assert ZeROStage.PARAMETERS == 3
    
    def test_comparison(self):
        assert ZeROStage.OPTIMIZER < ZeROStage.GRADIENTS
        assert ZeROStage.GRADIENTS < ZeROStage.PARAMETERS
        assert ZeROStage.PARAMETERS >= ZeROStage.OPTIMIZER


class TestPartitionedParameter:
    """PartitionedParameter 测试"""
    
    def test_init(self):
        param = nn.Parameter(torch.randn(100))
        pp = PartitionedParameter(param, rank=0, world_size=4, device=torch.device("cpu"))
        
        assert pp.numel == 100
        assert pp.partition_size == 25
        assert pp.start_idx == 0
        assert pp.end_idx == 25
        assert pp.local_numel == 25
    
    def test_partition(self):
        param = nn.Parameter(torch.randn(100))
        pp = PartitionedParameter(param, rank=1, world_size=4, device=torch.device("cpu"))
        
        local_data = pp.partition()
        assert local_data.shape[0] == 25
        assert pp._is_partitioned is True
    
    def test_partition_uneven(self):
        param = nn.Parameter(torch.randn(10))
        pp = PartitionedParameter(param, rank=3, world_size=4, device=torch.device("cpu"))
        
        assert pp.partition_size == 3
        assert pp.start_idx == 9
        assert pp.end_idx == 10
        assert pp.local_numel == 1
    
    def test_all_gather_not_initialized(self):
        param = nn.Parameter(torch.randn(10))
        pp = PartitionedParameter(param, rank=0, world_size=1, device=torch.device("cpu"))
        
        result = pp.all_gather()
        assert torch.equal(result, param.data)


# ============== 集成测试 ==============

class TestIntegration:
    """集成测试"""
    
    def test_ddp_trainer_with_config(self):
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )
        config = DDPConfig(
            find_unused_parameters=True,
            gradient_as_bucket_view=False,
        )
        trainer = DDPTrainer(model, config)
        
        assert trainer.config.find_unused_parameters is True
        assert trainer.config.gradient_as_bucket_view is False
    
    def test_zero_config_stages(self):
        for stage in [ZeROStage.DISABLED, ZeROStage.OPTIMIZER, 
                      ZeROStage.GRADIENTS, ZeROStage.PARAMETERS]:
            config = ZeROConfig(stage=stage)
            assert config.stage == stage


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
