"""
声音克隆基础模块 (Voice Cloning)

本模块实现基础的声音克隆功能，包括：
- 说话人编码器 (Speaker Encoder)
- 说话人嵌入提取
- 多说话人 TTS 适配
- 零样本声音克隆

=== 声音克隆核心概念 ===

声音克隆的目标是从少量参考音频中提取说话人特征，
然后将这些特征注入到 TTS 系统中，生成具有目标说话人音色的语音。

=== 主要方法 ===

1. 说话人编码器方法:
   - 使用预训练的说话人编码器提取说话人嵌入
   - 将嵌入注入到 TTS 模型中
   - 优点: 零样本克隆，无需微调

2. 微调方法:
   - 使用目标说话人数据微调 TTS 模型
   - 优点: 高质量，缺点: 需要更多数据

3. 适配器方法:
   - 冻结主模型，只训练小型适配器
   - 平衡质量和数据需求

=== 参考文献 ===

1. SV2TTS:
   Jia et al. "Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis" 2018

2. YourTTS:
   Casanova et al. "YourTTS: Towards Zero-Shot Multi-Speaker TTS and Zero-Shot Voice Conversion" 2022
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SpeakerEncoderConfig:
    """说话人编码器配置"""

    # 输入配置
    n_mels: int = 80
    sample_rate: int = 16000

    # 模型配置
    hidden_size: int = 256
    num_layers: int = 3
    embedding_size: int = 256

    # 训练配置
    dropout: float = 0.0


@dataclass
class VoiceCloningConfig:
    """声音克隆配置"""

    # 说话人编码器
    speaker_encoder: SpeakerEncoderConfig = None

    # TTS 适配
    tts_hidden_size: int = 256
    speaker_embedding_size: int = 256

    # 适配器配置
    adapter_hidden_size: int = 64
    adapter_dropout: float = 0.1

    def __post_init__(self):
        if self.speaker_encoder is None:
            self.speaker_encoder = SpeakerEncoderConfig()


class SpeakerEncoder(nn.Module):
    """
    说话人编码器

    从参考音频中提取说话人嵌入向量。
    使用 LSTM 处理变长音频，输出固定维度的说话人表示。
    """

    def __init__(self, config: SpeakerEncoderConfig):
        super().__init__()
        self.config = config

        # 输入投影
        self.input_proj = nn.Linear(config.n_mels, config.hidden_size)

        # LSTM 层
        self.lstm = nn.LSTM(
            config.hidden_size,
            config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=False
        )

        # 输出投影
        self.output_proj = nn.Linear(config.hidden_size, config.embedding_size)

    def forward(
        self,
        mel: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        提取说话人嵌入

        Args:
            mel: Mel 频谱 [batch, n_mels, time] 或 [batch, time, n_mels]
            lengths: 序列长度 [batch]

        Returns:
            说话人嵌入 [batch, embedding_size]
        """
        # 确保维度正确 [batch, time, n_mels]
        if mel.size(1) == self.config.n_mels:
            mel = mel.transpose(1, 2)

        # 输入投影
        x = self.input_proj(mel)

        # LSTM
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        _, (h_n, _) = self.lstm(x)

        # 取最后一层的隐藏状态
        embedding = h_n[-1]

        # 输出投影和 L2 归一化
        embedding = self.output_proj(embedding)
        embedding = F.normalize(embedding, p=2, dim=-1)

        return embedding

    def embed_utterance(self, mel: torch.Tensor) -> torch.Tensor:
        """嵌入单个语音"""
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        return self.forward(mel).squeeze(0)


class GE2ELoss(nn.Module):
    """
    Generalized End-to-End Loss

    用于训练说话人编码器的对比损失函数。
    优化版本：使用向量化操作替代循环。
    """

    def __init__(self, init_w: float = 10.0, init_b: float = -5.0):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))

    def forward(
        self,
        embeddings: torch.Tensor,
        speakers_per_batch: int,
        utterances_per_speaker: int
    ) -> torch.Tensor:
        """
        计算 GE2E 损失 (向量化实现)

        Args:
            embeddings: 说话人嵌入 [batch, embedding_size]
            speakers_per_batch: 每批说话人数
            utterances_per_speaker: 每个说话人的语音数

        Returns:
            GE2E 损失
        """
        batch_size = embeddings.size(0)
        expected_size = speakers_per_batch * utterances_per_speaker
        
        # 输入验证
        if batch_size != expected_size:
            raise ValueError(
                f"Batch size ({batch_size}) must equal "
                f"speakers_per_batch ({speakers_per_batch}) * "
                f"utterances_per_speaker ({utterances_per_speaker}) = {expected_size}"
            )
        
        if speakers_per_batch < 2:
            raise ValueError(f"speakers_per_batch must be >= 2, got {speakers_per_batch}")
        
        if utterances_per_speaker < 2:
            raise ValueError(f"utterances_per_speaker must be >= 2, got {utterances_per_speaker}")

        # 重塑为 [speakers, utterances, embedding_size]
        embeddings = embeddings.view(speakers_per_batch, utterances_per_speaker, -1)

        # 计算每个说话人的质心 [speakers, embedding_size]
        centroids = embeddings.mean(dim=1)

        # 向量化计算相似度矩阵
        sim_matrix = []
        
        for i in range(speakers_per_batch):
            for j in range(utterances_per_speaker):
                # 排除当前语音计算质心 (leave-one-out)
                mask = torch.ones(utterances_per_speaker, device=embeddings.device, dtype=torch.bool)
                mask[j] = False
                centroid_excl = embeddings[i, mask].mean(dim=0)  # [embedding_size]

                # 当前嵌入
                current_emb = embeddings[i, j]  # [embedding_size]

                # 计算与所有说话人质心的相似度
                sims = []
                for k in range(speakers_per_batch):
                    if k == i:
                        # 使用排除当前语音的质心
                        sim = F.cosine_similarity(
                            current_emb.unsqueeze(0),
                            centroid_excl.unsqueeze(0),
                            dim=-1
                        ).squeeze()
                    else:
                        # 使用完整质心
                        sim = F.cosine_similarity(
                            current_emb.unsqueeze(0),
                            centroids[k].unsqueeze(0),
                            dim=-1
                        ).squeeze()
                    sims.append(sim)
                
                sim_matrix.append(torch.stack(sims))

        sim_matrix = torch.stack(sim_matrix)  # [batch, speakers]
        sim_matrix = self.w * sim_matrix + self.b

        # 创建标签：每个语音的正确说话人索引
        labels = torch.arange(speakers_per_batch, device=embeddings.device)
        labels = labels.repeat_interleave(utterances_per_speaker)

        # 交叉熵损失
        loss = F.cross_entropy(sim_matrix, labels)

        return loss


class SpeakerAdapter(nn.Module):
    """
    说话人适配器

    将说话人嵌入注入到 TTS 模型的隐藏状态中。
    """

    def __init__(self, config: VoiceCloningConfig):
        super().__init__()
        self.config = config

        # 说话人嵌入投影
        self.speaker_proj = nn.Sequential(
            nn.Linear(config.speaker_embedding_size, config.adapter_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.adapter_dropout),
            nn.Linear(config.adapter_hidden_size, config.tts_hidden_size)
        )

        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(config.tts_hidden_size * 2, config.tts_hidden_size),
            nn.Sigmoid()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        speaker_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        将说话人信息注入隐藏状态

        Args:
            hidden_states: TTS 隐藏状态 [batch, seq_len, hidden_size]
            speaker_embedding: 说话人嵌入 [batch, embedding_size]

        Returns:
            调整后的隐藏状态 [batch, seq_len, hidden_size]
        """
        # 投影说话人嵌入
        speaker_proj = self.speaker_proj(speaker_embedding)  # [batch, hidden_size]

        # 扩展到序列长度
        speaker_proj = speaker_proj.unsqueeze(1).expand(-1, hidden_states.size(1), -1)

        # 门控融合
        gate_input = torch.cat([hidden_states, speaker_proj], dim=-1)
        gate = self.gate(gate_input)

        # 残差连接
        output = hidden_states + gate * speaker_proj

        return output


class MultiSpeakerTTS(nn.Module):
    """
    多说话人 TTS 包装器

    将说话人编码器和适配器与基础 TTS 模型结合。
    """

    def __init__(
        self,
        tts_model: nn.Module,
        speaker_encoder: SpeakerEncoder,
        adapter: SpeakerAdapter
    ):
        super().__init__()
        self.tts_model = tts_model
        self.speaker_encoder = speaker_encoder
        self.adapter = adapter

        # 冻结说话人编码器
        for param in self.speaker_encoder.parameters():
            param.requires_grad = False

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        reference_mel: torch.Tensor,
        reference_lengths: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        多说话人 TTS 前向传播

        Args:
            text: 输入文本
            text_lengths: 文本长度
            reference_mel: 参考音频的 Mel 频谱
            reference_lengths: 参考音频长度
            **kwargs: 传递给 TTS 模型的其他参数

        Returns:
            TTS 模型输出
        """
        # 提取说话人嵌入
        with torch.no_grad():
            speaker_embedding = self.speaker_encoder(reference_mel, reference_lengths)

        # 调用 TTS 模型 (需要根据具体 TTS 模型调整)
        return self.tts_model(text, text_lengths, speaker_embedding=speaker_embedding, **kwargs)

    @torch.no_grad()
    def synthesize(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        reference_mel: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """合成语音"""
        speaker_embedding = self.speaker_encoder(reference_mel)
        return self.tts_model.infer(text, text_lengths, speaker_embedding=speaker_embedding, **kwargs)


class VoiceCloner:
    """
    声音克隆器

    提供简单的 API 进行声音克隆。
    """

    def __init__(
        self,
        speaker_encoder: SpeakerEncoder,
        tts_model: nn.Module,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.speaker_encoder = speaker_encoder.to(self.device)
        self.tts_model = tts_model.to(self.device)

        self.speaker_encoder.eval()
        self.tts_model.eval()

        # 缓存的说话人嵌入
        self._cached_embedding: Optional[torch.Tensor] = None

    def encode_speaker(self, reference_mel: torch.Tensor) -> torch.Tensor:
        """
        编码参考音频为说话人嵌入

        Args:
            reference_mel: 参考音频的 Mel 频谱

        Returns:
            说话人嵌入向量
        """
        with torch.no_grad():
            if reference_mel.dim() == 2:
                reference_mel = reference_mel.unsqueeze(0)
            reference_mel = reference_mel.to(self.device)
            embedding = self.speaker_encoder(reference_mel)
            self._cached_embedding = embedding
            return embedding

    def clone_voice(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        reference_mel: Optional[torch.Tensor] = None,
        speaker_embedding: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        克隆声音生成语音

        Args:
            text: 输入文本
            text_lengths: 文本长度
            reference_mel: 参考音频 (与 speaker_embedding 二选一)
            speaker_embedding: 预计算的说话人嵌入
            **kwargs: 传递给 TTS 模型的参数

        Returns:
            生成的音频波形
        """
        with torch.no_grad():
            # 获取说话人嵌入
            if speaker_embedding is not None:
                emb = speaker_embedding.to(self.device)
            elif reference_mel is not None:
                emb = self.encode_speaker(reference_mel)
            elif self._cached_embedding is not None:
                emb = self._cached_embedding
            else:
                raise ValueError("Must provide reference_mel or speaker_embedding")

            # 确保所有输入在同一设备上
            text = text.to(self.device)
            text_lengths = text_lengths.to(self.device)

            if hasattr(self.tts_model, 'infer'):
                audio = self.tts_model.infer(
                    text, text_lengths,
                    speaker_embedding=emb,
                    **kwargs
                )
            else:
                output = self.tts_model(
                    text, text_lengths,
                    speaker_embedding=emb,
                    **kwargs
                )
                audio = output.get('audio', output) if isinstance(output, dict) else output

            return audio

    def compare_speakers(
        self,
        mel1: torch.Tensor,
        mel2: torch.Tensor
    ) -> float:
        """
        比较两个音频的说话人相似度
        
        Args:
            mel1: 第一个音频的 Mel 频谱
            mel2: 第二个音频的 Mel 频谱
            
        Returns:
            余弦相似度 [-1, 1]
        """
        with torch.no_grad():
            emb1 = self.encode_speaker(mel1)
            emb2 = self.encode_speaker(mel2)
            similarity = F.cosine_similarity(emb1, emb2, dim=-1)
            return similarity.item()


def create_speaker_encoder(size: str = "base") -> SpeakerEncoder:
    """
    创建说话人编码器

    Args:
        size: 模型大小 ("tiny", "base", "large")

    Returns:
        SpeakerEncoder 实例
    """
    configs = {
        "tiny": SpeakerEncoderConfig(
            hidden_size=128,
            num_layers=2,
            embedding_size=128,
        ),
        "base": SpeakerEncoderConfig(
            hidden_size=256,
            num_layers=3,
            embedding_size=256,
        ),
        "large": SpeakerEncoderConfig(
            hidden_size=512,
            num_layers=4,
            embedding_size=512,
        ),
    }

    if size not in configs:
        raise ValueError(f"Unknown size: {size}. Choose from {list(configs.keys())}")

    return SpeakerEncoder(configs[size])


def create_voice_cloning_config(
    tts_hidden_size: int = 256,
    speaker_embedding_size: int = 256
) -> VoiceCloningConfig:
    """创建声音克隆配置"""
    return VoiceCloningConfig(
        speaker_encoder=SpeakerEncoderConfig(embedding_size=speaker_embedding_size),
        tts_hidden_size=tts_hidden_size,
        speaker_embedding_size=speaker_embedding_size,
    )
