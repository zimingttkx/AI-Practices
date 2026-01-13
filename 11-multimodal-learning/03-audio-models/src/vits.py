"""
VITS 端到端语音合成模型 (Variational Inference with adversarial learning for end-to-end Text-to-Speech)

本模块实现 VITS 风格的端到端 TTS 系统，包括：
- 后验编码器 (从音频提取潜在表示)
- 先验编码器 (从文本预测潜在分布)
- 流模型 (学习先验到后验的映射)
- 随机时长预测器
- HiFi-GAN 解码器

=== VITS 核心优势 ===

1. 端到端: 直接从文本生成波形，无需中间 Mel 频谱
2. 高质量: 结合 VAE 和 GAN 的优势
3. 多样性: 随机时长预测器产生自然的韵律变化
4. 快速: 并行生成，推理速度快

=== 模型架构 ===

训练时:
文本 → [先验编码器] → μ_p, σ_p
音频 → [后验编码器] → μ_q, σ_q → z → [解码器] → 重建音频
                    ↓
              [流模型] 学习 p(z|text) 到 q(z|audio) 的映射

推理时:
文本 → [先验编码器] → μ_p, σ_p → [流模型逆变换] → z → [解码器] → 音频

=== 参考文献 ===

Kim et al. "Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech" 2021
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VITSConfig:
    """VITS 模型配置"""

    # 文本配置
    vocab_size: int = 256
    max_seq_len: int = 500

    # 音频配置
    n_mels: int = 80
    sample_rate: int = 22050
    hop_length: int = 256
    win_length: int = 1024
    n_fft: int = 1024

    # 编码器配置
    hidden_channels: int = 192
    filter_channels: int = 768
    n_heads: int = 2
    n_layers: int = 6
    kernel_size: int = 3
    p_dropout: float = 0.1

    # 流模型配置
    n_flows: int = 4
    flow_hidden_channels: int = 192

    # 时长预测器配置
    duration_predictor_channels: int = 256
    duration_predictor_kernel: int = 3
    duration_predictor_dropout: float = 0.5

    # 解码器配置 (HiFi-GAN)
    upsample_rates: Tuple[int, ...] = (8, 8, 2, 2)
    upsample_kernel_sizes: Tuple[int, ...] = (16, 16, 4, 4)
    upsample_initial_channel: int = 512
    resblock_kernel_sizes: Tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes: Tuple[Tuple[int, ...], ...] = (
        (1, 3, 5), (1, 3, 5), (1, 3, 5)
    )

    # 潜在空间配置
    inter_channels: int = 192
    segment_size: int = 32


class LayerNorm(nn.Module):
    """通道维度的层归一化"""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class WaveNetResBlock(nn.Module):
    """WaveNet 风格的残差块"""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        n_layers: int = 2,
        p_dropout: float = 0.0
    ):
        super().__init__()
        self.channels = channels
        self.n_layers = n_layers

        self.dilated_convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(n_layers):
            d = dilation ** i
            padding = (kernel_size * d - d) // 2
            self.dilated_convs.append(nn.Conv1d(
                channels, 2 * channels, kernel_size,
                dilation=d, padding=padding
            ))
            self.norms.append(LayerNorm(2 * channels))

        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for conv, norm in zip(self.dilated_convs, self.norms):
            residual = x
            x = conv(x)
            x = norm(x)

            x_tanh, x_sigmoid = x.chunk(2, dim=1)
            x = torch.tanh(x_tanh) * torch.sigmoid(x_sigmoid)
            x = self.dropout(x)
            x = x + residual

            if x_mask is not None:
                x = x * x_mask

        return x


class TextEncoder(nn.Module):
    """
    文本编码器 (先验编码器)

    将文本序列编码为潜在分布的参数 (μ, σ)
    """

    def __init__(self, config: VITSConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_channels)
        nn.init.normal_(self.embedding.weight, 0.0, config.hidden_channels ** -0.5)

        self.encoder = WaveNetResBlock(
            config.hidden_channels,
            config.kernel_size,
            n_layers=config.n_layers,
            p_dropout=config.p_dropout
        )

        self.proj = nn.Conv1d(config.hidden_channels, config.inter_channels * 2, 1)

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        编码文本序列
        
        Returns:
            x: 编码后的隐藏状态
            m: 先验分布均值
            logs: 先验分布对数标准差
            x_mask: 序列掩码
        """
        x = self.embedding(text).transpose(1, 2)

        x_mask = self._get_mask(text_lengths, x.size(2)).unsqueeze(1).to(x.device)
        x = self.encoder(x * x_mask, x_mask)

        stats = self.proj(x) * x_mask
        m, logs = stats.chunk(2, dim=1)

        return x, m, logs, x_mask

    def _get_mask(self, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        ids = torch.arange(max_len, device=lengths.device)
        return (ids < lengths.unsqueeze(1)).float()


class PosteriorEncoder(nn.Module):
    """
    后验编码器

    从音频频谱提取潜在表示 z ~ q(z|x)
    """

    def __init__(self, config: VITSConfig):
        super().__init__()
        self.config = config

        self.pre = nn.Conv1d(config.n_mels, config.hidden_channels, 1)
        self.encoder = WaveNetResBlock(
            config.hidden_channels,
            config.kernel_size,
            n_layers=config.n_layers,
            p_dropout=config.p_dropout
        )
        self.proj = nn.Conv1d(config.hidden_channels, config.inter_channels * 2, 1)

    def forward(
        self,
        spec: torch.Tensor,
        spec_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_mask = self._get_mask(spec_lengths, spec.size(2)).unsqueeze(1).to(spec.device)

        x = self.pre(spec) * x_mask
        x = self.encoder(x, x_mask)

        stats = self.proj(x) * x_mask
        m, logs = stats.chunk(2, dim=1)

        z = m + torch.randn_like(m) * torch.exp(logs)

        return z, m, logs, x_mask

    def _get_mask(self, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        ids = torch.arange(max_len, device=lengths.device)
        return (ids < lengths.unsqueeze(1)).float()


class ResidualCouplingLayer(nn.Module):
    """
    残差耦合层

    流模型的基本构建块，实现可逆变换。
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int = 5,
        n_layers: int = 4,
        p_dropout: float = 0.0
    ):
        super().__init__()
        self.channels = channels
        self.half_channels = channels // 2

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WaveNetResBlock(hidden_channels, kernel_size, n_layers=n_layers, p_dropout=p_dropout)
        self.post = nn.Conv1d(hidden_channels, self.half_channels, 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        reverse: bool = False
    ) -> torch.Tensor:
        x0, x1 = x.chunk(2, dim=1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask)
        stats = self.post(h) * x_mask
        m = stats

        if not reverse:
            x1 = x1 + m
        else:
            x1 = x1 - m

        return torch.cat([x0, x1], dim=1)


class ResidualCouplingBlock(nn.Module):
    """残差耦合块 (多层流模型)"""

    def __init__(self, config: VITSConfig):
        super().__init__()
        self.flows = nn.ModuleList()

        for _ in range(config.n_flows):
            self.flows.append(ResidualCouplingLayer(
                config.inter_channels,
                config.flow_hidden_channels,
                n_layers=4,
                p_dropout=config.p_dropout
            ))

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        reverse: bool = False
    ) -> torch.Tensor:
        if not reverse:
            for flow in self.flows:
                x = flow(x, x_mask, reverse=False)
                x = torch.flip(x, dims=[1])
        else:
            for flow in reversed(self.flows):
                x = torch.flip(x, dims=[1])
                x = flow(x, x_mask, reverse=True)
        return x


class StochasticDurationPredictor(nn.Module):
    """
    随机时长预测器

    使用流模型预测时长分布，产生自然的韵律变化。
    """

    def __init__(self, config: VITSConfig):
        super().__init__()
        self.config = config

        self.pre = nn.Conv1d(config.hidden_channels, config.duration_predictor_channels, 1)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                config.duration_predictor_channels,
                config.duration_predictor_channels,
                config.duration_predictor_kernel,
                padding=(config.duration_predictor_kernel - 1) // 2
            )
            for _ in range(3)
        ])
        self.norms = nn.ModuleList([
            LayerNorm(config.duration_predictor_channels)
            for _ in range(3)
        ])
        self.proj = nn.Conv1d(config.duration_predictor_channels, 1, 1)
        self.dropout = nn.Dropout(config.duration_predictor_dropout)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        w: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.pre(x.detach())

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x * x_mask)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)

        x = self.proj(x * x_mask) * x_mask

        if w is not None:
            log_w = torch.log(w + 1e-6) * x_mask
            loss = torch.sum((x - log_w) ** 2 * x_mask) / torch.sum(x_mask)
            return loss

        return torch.exp(x) * x_mask

    def infer(self, x: torch.Tensor, x_mask: torch.Tensor, noise_scale: float = 1.0) -> torch.Tensor:
        x = self.pre(x)

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x * x_mask)
            x = norm(x)
            x = F.relu(x)

        x = self.proj(x * x_mask) * x_mask
        w = torch.exp(x) * x_mask

        if noise_scale > 0:
            w = w * (1 + torch.randn_like(w) * noise_scale * 0.1)

        return torch.clamp(w, min=0)


class ResBlock(nn.Module):
    """HiFi-GAN 残差块"""

    def __init__(self, channels: int, kernel_size: int = 3, dilations: Tuple[int, ...] = (1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()

        for d in dilations:
            padding = (kernel_size * d - d) // 2
            self.convs1.append(nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=padding))
            self.convs2.append(nn.Conv1d(channels, channels, kernel_size, dilation=1, padding=(kernel_size - 1) // 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x


class Generator(nn.Module):
    """HiFi-GAN 生成器 (解码器)"""

    def __init__(self, config: VITSConfig):
        super().__init__()
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)

        self.conv_pre = nn.Conv1d(config.inter_channels, config.upsample_initial_channel, 7, padding=3)

        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        ch = config.upsample_initial_channel
        for i, (u, k) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            self.ups.append(nn.ConvTranspose1d(ch, ch // 2, k, stride=u, padding=(k - u) // 2))
            ch = ch // 2

            for j, (k_r, d_r) in enumerate(zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, k_r, d_r))

        self.conv_post = nn.Conv1d(ch, 1, 7, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(x)

        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, 0.1)
            x = up(x)

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x


class VITS(nn.Module):
    """
    VITS 端到端语音合成模型

    主要组件:
    1. 文本编码器: 编码文本为先验分布
    2. 后验编码器: 从音频提取潜在表示
    3. 流模型: 学习先验到后验的映射
    4. 时长预测器: 预测音素时长
    5. 解码器: 从潜在表示生成波形
    """

    def __init__(self, config: VITSConfig):
        super().__init__()
        self.config = config

        self.text_encoder = TextEncoder(config)
        self.posterior_encoder = PosteriorEncoder(config)
        self.flow = ResidualCouplingBlock(config)
        self.duration_predictor = StochasticDurationPredictor(config)
        self.decoder = Generator(config)

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        spec: torch.Tensor,
        spec_lengths: torch.Tensor
    ) -> dict:
        """
        训练前向传播

        Args:
            text: 输入文本 [batch, text_len]
            text_lengths: 文本长度 [batch]
            spec: 线性频谱 [batch, n_mels, spec_len]
            spec_lengths: 频谱长度 [batch]

        Returns:
            包含各种输出和损失的字典
        """
        # 文本编码
        x, m_p, logs_p, x_mask = self.text_encoder(text, text_lengths)

        # 后验编码
        z, m_q, logs_q, y_mask = self.posterior_encoder(spec, spec_lengths)

        # 流模型变换
        z_p = self.flow(z, y_mask)

        # 计算时长
        with torch.no_grad():
            # 简化的时长计算
            w = torch.ones_like(x_mask.squeeze(1)) * (spec_lengths.float() / text_lengths.float()).unsqueeze(1)

        # 时长损失
        duration_loss = self.duration_predictor(x, x_mask, w)

        # 随机采样片段用于解码器训练
        z_slice, ids_slice = self._rand_slice_segments(z, spec_lengths, self.config.segment_size)

        # 解码
        audio = self.decoder(z_slice)

        return {
            "audio": audio,
            "ids_slice": ids_slice,
            "z": z,
            "z_p": z_p,
            "m_p": m_p,
            "logs_p": logs_p,
            "m_q": m_q,
            "logs_q": logs_q,
            "x_mask": x_mask,
            "y_mask": y_mask,
            "duration_loss": duration_loss
        }

    def _rand_slice_segments(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        segment_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """随机切片"""
        b, d, t = x.size()
        ids_start_max = x_lengths - segment_size
        ids_start_max = torch.clamp(ids_start_max, min=0)
        ids_start = (torch.rand(b, device=x.device) * ids_start_max.float()).long()

        ret = torch.zeros(b, d, segment_size, device=x.device, dtype=x.dtype)
        for i in range(b):
            start = ids_start[i].item()
            end = min(start + segment_size, x.size(2))
            ret[i, :, :end-start] = x[i, :, start:end]

        return ret, ids_start

    @torch.no_grad()
    def infer(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        noise_scale: float = 0.667,
        length_scale: float = 1.0,
        noise_scale_w: float = 0.8
    ) -> torch.Tensor:
        """
        推理生成语音

        Args:
            text: 输入文本
            text_lengths: 文本长度
            noise_scale: 噪声缩放因子
            length_scale: 时长缩放因子
            noise_scale_w: 时长预测噪声

        Returns:
            生成的音频波形
        """
        # 文本编码
        x, m_p, logs_p, x_mask = self.text_encoder(text, text_lengths)

        # 预测时长
        w = self.duration_predictor.infer(x, x_mask, noise_scale_w)
        w = w * length_scale

        # 扩展到帧级别
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, dim=[1, 2]), 1).long()
        y_mask = self._get_mask(y_lengths, y_lengths.max()).unsqueeze(1).to(x.device)

        # 扩展先验分布
        m_p_expanded = self._expand_by_duration(m_p, w_ceil, y_lengths.max())
        logs_p_expanded = self._expand_by_duration(logs_p, w_ceil, y_lengths.max())

        # 采样潜在变量
        z_p = m_p_expanded + torch.randn_like(m_p_expanded) * torch.exp(logs_p_expanded) * noise_scale

        # 流模型逆变换
        z = self.flow(z_p, y_mask, reverse=True)

        # 解码
        audio = self.decoder(z * y_mask)

        return audio

    def _get_mask(self, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        ids = torch.arange(max_len, device=lengths.device)
        return (ids < lengths.unsqueeze(1)).float()

    def _expand_by_duration(
        self,
        x: torch.Tensor,
        duration: torch.Tensor,
        max_len: int
    ) -> torch.Tensor:
        """
        根据时长扩展序列
        
        Args:
            x: 输入序列 [batch, channels, seq_len]
            duration: 时长 [batch, 1, seq_len]
            max_len: 最大输出长度
            
        Returns:
            扩展后的序列 [batch, channels, max_len]
        """
        batch_size = x.size(0)
        max_len = max(int(max_len), 1)
        expanded = torch.zeros(batch_size, x.size(1), max_len, device=x.device, dtype=x.dtype)

        for b in range(batch_size):
            # 获取当前样本的时长
            dur = duration[b, 0].long()  # [seq_len]
            dur = torch.clamp(dur, min=0)
            
            total_dur = dur.sum().item()
            if total_dur == 0:
                continue
            
            # 使用 repeat_interleave 进行高效扩展
            expanded_seq = torch.repeat_interleave(x[b], dur, dim=1)  # [channels, expanded_len]
            
            # 截断到 max_len
            actual_len = min(expanded_seq.size(1), max_len)
            expanded[b, :, :actual_len] = expanded_seq[:, :actual_len]

        return expanded


def vits_loss(
    audio_real: torch.Tensor,
    audio_fake: torch.Tensor,
    z_p: torch.Tensor,
    m_p: torch.Tensor,
    logs_p: torch.Tensor,
    m_q: torch.Tensor,
    logs_q: torch.Tensor,
    y_mask: torch.Tensor,
    duration_loss: torch.Tensor
) -> Tuple[torch.Tensor, dict]:
    """
    计算 VITS 损失

    Args:
        audio_real: 真实音频片段
        audio_fake: 生成音频片段
        z_p: 流模型变换后的潜在变量
        m_p: 先验均值
        logs_p: 先验对数方差
        m_q: 后验均值
        logs_q: 后验对数方差
        y_mask: 音频掩码
        duration_loss: 时长预测损失

    Returns:
        总损失和各项损失字典
    """
    # 重建损失 (L1)
    recon_loss = F.l1_loss(audio_fake, audio_real)

    # KL 散度损失
    kl_loss = kl_divergence(z_p, logs_p, m_p, logs_q, m_q, y_mask)

    # 总损失
    total_loss = recon_loss + kl_loss + duration_loss

    return total_loss, {
        "recon_loss": recon_loss,
        "kl_loss": kl_loss,
        "duration_loss": duration_loss
    }


def kl_divergence(
    z_p: torch.Tensor,
    logs_p: torch.Tensor,
    m_p: torch.Tensor,
    logs_q: torch.Tensor,
    m_q: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    计算 KL 散度: KL(q(z|x) || p(z|c))
    
    对于两个高斯分布 q ~ N(m_q, σ_q²) 和 p ~ N(m_p, σ_p²):
    KL(q||p) = log(σ_p/σ_q) + (σ_q² + (m_q - m_p)²) / (2σ_p²) - 0.5
    
    由于我们使用 log(σ) 而非 σ:
    KL = logs_p - logs_q + (exp(2*logs_q) + (m_q - m_p)²) / (2*exp(2*logs_p)) - 0.5
    """
    # 数值稳定性：clamp logs 防止 exp 溢出
    logs_p_clamped = torch.clamp(logs_p, min=-10.0, max=10.0)
    logs_q_clamped = torch.clamp(logs_q, min=-10.0, max=10.0)
    
    kl = logs_p_clamped - logs_q_clamped - 0.5
    kl += 0.5 * (torch.exp(2.0 * logs_q_clamped) + (m_q - m_p) ** 2) * torch.exp(-2.0 * logs_p_clamped)
    kl = torch.sum(kl * mask)
    
    # 避免除零
    mask_sum = torch.sum(mask)
    if mask_sum > 0:
        return kl / mask_sum
    return kl * 0.0


def create_vits_model(size: str = "base") -> VITS:
    """
    创建预定义大小的 VITS 模型

    Args:
        size: 模型大小 ("tiny", "base", "large")

    Returns:
        VITS 模型实例
    """
    configs = {
        "tiny": VITSConfig(
            hidden_channels=128,
            filter_channels=512,
            n_heads=2,
            n_layers=4,
            n_flows=2,
            flow_hidden_channels=128,
            inter_channels=128,
            upsample_initial_channel=256,
        ),
        "base": VITSConfig(
            hidden_channels=192,
            filter_channels=768,
            n_heads=2,
            n_layers=6,
            n_flows=4,
            flow_hidden_channels=192,
            inter_channels=192,
            upsample_initial_channel=512,
        ),
        "large": VITSConfig(
            hidden_channels=256,
            filter_channels=1024,
            n_heads=4,
            n_layers=8,
            n_flows=6,
            flow_hidden_channels=256,
            inter_channels=256,
            upsample_initial_channel=512,
        ),
    }

    if size not in configs:
        raise ValueError(f"Unknown model size: {size}. Choose from {list(configs.keys())}")

    return VITS(configs[size])
