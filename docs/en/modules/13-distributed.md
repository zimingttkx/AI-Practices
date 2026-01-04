# 13 - Distributed Training

This module covers parallel strategies and optimization techniques for large-scale model training.

## Module Structure

```
13-distributed-training/
├── 01-data-parallel/           # Data Parallel
├── 02-model-parallel/          # Model Parallel
├── 03-mixed-precision/         # Mixed Precision
└── 04-large-scale-training/    # Large-Scale Training
```

## Core Content

### 01 - Data Parallel

| Technology | Description | Use Case |
|:-----------|:------------|:---------|
| DDP | Distributed Data Parallel | Multi-GPU |
| FSDP | Fully Sharded Data Parallel | Large models |
| ZeRO | Zero Redundancy Optimizer | Memory optimization |

### 02 - Model Parallel

| Technology | Description |
|:-----------|:------------|
| Tensor Parallel | Intra-layer partitioning |
| Pipeline Parallel | Inter-layer partitioning |
| Sequence Parallel | Sequence dimension partitioning |

### 03 - Mixed Precision

| Technology | Description |
|:-----------|:------------|
| AMP | Automatic Mixed Precision |
| BF16 | Brain Float 16 |
| Gradient Scaling | Prevent underflow |

### 04 - Large-Scale Training

| Framework | Features |
|:----------|:---------|
| DeepSpeed | Microsoft, ZeRO optimization |
| Megatron-LM | NVIDIA, 3D parallelism |
| FSDP | PyTorch native |

## Learning Path

```
DDP → FSDP/ZeRO → Tensor Parallel → Pipeline Parallel → Mixed Precision → DeepSpeed
```
