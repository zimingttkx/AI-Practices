# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- 13-distributed-training module (DDP, FSDP, ZeRO, Megatron-LM)

## [2.0.0] - 2026-01-03

### Added
- **10-large-language-models**: Complete LLM module
  - Transformer architecture and tokenizers
  - GPT/BERT/LLaMA implementations
  - LoRA/QLoRA fine-tuning
  - Prompt engineering techniques
  - RAG (Retrieval-Augmented Generation)
  - Agent systems with tool calling
  - RLHF/DPO alignment training
- **11-multimodal-learning**: Multimodal AI module
  - Vision-language models (CLIP, BLIP, LLaVA)
  - Image generation (VAE, Diffusion, Stable Diffusion, ControlNet)
  - Audio models (Whisper ASR, TTS, HiFi-GAN)
- **12-deployment-optimization**: Production deployment module
  - Model optimization (quantization, pruning, distillation, ONNX)
  - Inference engines (TensorRT, vLLM, Triton)
  - Serving systems (FastAPI, gRPC, load balancing)
  - MLOps (experiment tracking, model registry, monitoring)
- 1000+ unit tests across all modules
- Multi-stage Dockerfile with dev/prod/jupyter/test targets
- docker-compose.yml for container orchestration
- GitHub Actions CI/CD pipeline (ci-test.yml)
- pyproject.toml with modern Python packaging

### Changed
- Upgraded project structure to 12 core modules
- Enhanced DEVELOPMENT.md with comprehensive coding guidelines
- Updated requirements.txt with new dependencies
- Improved test coverage configuration in pyproject.toml

## [1.0.0] - 2024-12-13

### Added
- Initial release with 9 learning modules
- 179 Jupyter notebooks covering ML/DL topics
- 19 practical projects across 5 categories
- Comprehensive documentation structure
- GitHub Actions for documentation deployment
- Issue and PR templates
- Code of Conduct and Security Policy

### Modules Included
- **01-foundations**: Machine learning basics (8 sub-modules)
- **02-neural-networks**: Deep learning fundamentals
- **03-computer-vision**: CNN and image processing
- **04-sequence-models**: RNN, LSTM, Transformers
- **05-advanced-topics**: Specialized deep learning topics
- **06-generative-models**: GANs, VAE, Diffusion models
- **07-reinforcement-learning**: RL algorithms and applications
- **08-theory-notes**: Mathematical foundations
- **09-practical-projects**: End-to-end ML projects

---

[Unreleased]: https://github.com/zimingttkx/AI-Practices/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/zimingttkx/AI-Practices/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/zimingttkx/AI-Practices/releases/tag/v1.0.0
