# 13-distributed-training

> **Prerequisites**: Python basics, PyTorch fundamentals (tensors, autograd, nn.Module), deep learning basics (forward/backward propagation, gradient descent)

## Core Problem: Why Distributed Training?

```
Single GPU Training Bottlenecks
├── Memory Limit: GPT-3 (175B params) requires ~350GB for parameters alone
├── Compute Bottleneck: Single-GPU training takes months or years
└── Data Throughput: Single-GPU data loading becomes the bottleneck

Distributed Training Solutions
├── Data Parallelism: Multiple GPUs process different data, sync gradients
├── Model Parallelism: Split model across GPUs to overcome memory limits
└── Mixed Precision: FP16/BF16 computation reduces memory, accelerates training
```

## Knowledge Architecture

```
Distributed Training
├── 01-Data Parallelism
│   ├── DDP: Full model per GPU, data sharding, AllReduce gradient sync
│   ├── FSDP: Full sharding of params/grads/optimizer states
│   └── ZeRO: DeepSpeed memory optimization, Stage 1/2/3 progressive sharding
│
├── 02-Model Parallelism
│   ├── Tensor Parallel: Split weight matrices within layers (column/row)
│   ├── Pipeline Parallel: Split model by layers, micro-batch pipelining
│   └── Sequence Parallel: Split sequence dimension, complements tensor parallel
│
├── 03-Mixed Precision
│   ├── AMP: Automatic mixed precision, FP16 compute + FP32 accumulation
│   ├── BF16: 8-bit exponent, no scaling needed, Ampere+ GPU support
│   └── Gradient Scaling: Prevent FP16 gradient underflow
│
└── 04-Large-Scale Training
    ├── DeepSpeed: Microsoft framework, ZeRO + mixed precision + offload
    ├── Megatron: NVIDIA framework, 3D parallelism (TP+PP+DP)
    └── Distributed Checkpointing: Sharded save/load, fault tolerance
```

## Technology Selection Guide

| Scenario | Recommended | Rationale |
|----------|-------------|-----------|
| Model fits single GPU | DDP | Simple, minimal communication overhead |
| Model slightly exceeds GPU memory | FSDP/ZeRO-2 | Gradient sharding reduces memory |
| Model far exceeds GPU memory | FSDP/ZeRO-3 + Tensor Parallel | Full sharding + intra-layer split |
| Very large models (100B+) | 3D Parallelism (TP+PP+DP) | Megatron-LM approach |
| Memory constrained | + CPU Offload | Offload params/optimizer to CPU |
| Maximize training speed | + Mixed Precision (BF16) | 2x speedup, no gradient scaling |

## Module Structure

```
13-distributed-training/
├── 01-data-parallel/        # Data Parallelism: DDP, FSDP, ZeRO
│   ├── src/
│   │   ├── ddp.py           # PyTorch DDP wrapper
│   │   ├── fsdp.py          # FSDP trainer
│   │   └── zero.py          # ZeRO optimizer implementation
│   ├── notebooks/           # Interactive tutorials
│   │   ├── 01_ddp_tutorial.ipynb
│   │   ├── 02_fsdp_tutorial.ipynb
│   │   └── 03_zero_tutorial.ipynb
│   └── tests/               # 28 test cases
│
├── 02-model-parallel/       # Model Parallelism: Tensor/Pipeline/Sequence
│   ├── src/
│   │   ├── tensor_parallel.py    # Column/Row/Vocab parallel
│   │   ├── pipeline_parallel.py  # GPipe/PipeDream schedulers
│   │   └── sequence_parallel.py  # Sequence dimension sharding
│   ├── notebooks/
│   │   ├── 01_tensor_parallel_tutorial.ipynb
│   │   ├── 02_pipeline_parallel_tutorial.ipynb
│   │   └── 03_sequence_parallel_tutorial.ipynb
│   └── tests/               # 26 test cases
│
├── 03-mixed-precision/      # Mixed Precision: AMP, BF16, Gradient Scaling
│   ├── src/
│   │   ├── amp.py           # Automatic mixed precision trainer
│   │   ├── bf16_training.py # BF16 training support
│   │   └── gradient_scaling.py  # Dynamic gradient scaling
│   ├── notebooks/
│   │   ├── 01_amp_tutorial.ipynb
│   │   └── 02_bf16_training_tutorial.ipynb
│   └── tests/               # 31 test cases
│
└── 04-large-scale-training/ # Large-Scale Training Tools
    ├── src/
    │   ├── deepspeed_config.py  # DeepSpeed config generator
    │   ├── megatron_core.py     # Megatron parallel state management
    │   └── checkpoint_utils.py  # Distributed checkpointing
    ├── notebooks/
    │   ├── 01_deepspeed_tutorial.ipynb
    │   ├── 02_megatron_tutorial.ipynb
    │   └── 03_checkpoint_tutorial.ipynb
    └── tests/               # 20 test cases
```

## Quick Start

### Requirements

```bash
pip install torch>=2.0.0  # PyTorch 2.0+ for FSDP 2.0
pip install deepspeed     # Optional: DeepSpeed support

# Verify CUDA and NCCL
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, NCCL: {torch.distributed.is_nccl_available()}')"
```

### Run Tests

```bash
# Run all tests (105 total)
pytest 13-distributed-training/ -v

# Run single module tests
pytest 13-distributed-training/01-data-parallel/tests/ -v
```

## Learning Path

```
Beginner → Intermediate → Advanced
    │           │            │
    ↓           ↓            ↓
   DDP  →  FSDP/ZeRO  →  3D Parallelism
    │           │            │
    ↓           ↓            ↓
   AMP  →    BF16    →  DeepSpeed/Megatron
```

1. **Beginner**: Master DDP + AMP, the most common combination
2. **Intermediate**: Learn FSDP/ZeRO for larger models
3. **Advanced**: Understand tensor/pipeline parallelism for very large models

## References

- [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [Megatron-LM Paper](https://arxiv.org/abs/1909.08053)
- [ZeRO Paper](https://arxiv.org/abs/1910.02054)
