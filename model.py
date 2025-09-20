import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class DWT(nn.Module):
    """Discrete Wavelet Transform"""
    def __init__(self):
        super(DWT, self).__init__()
        self.low = torch.tensor([1., 1.]) / math.sqrt(2)
        self.high = torch.tensor([1., -1.]) / math.sqrt(2)

    def forward(self, x):
        """
        Input: x (B, C, H, W)
        Output: (B, 4*C, H/2, W/2) - [LL, LH, HL, HH] concatenated
        """
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, "Height and Width must be even"
        low = self.low.to(x.device)
        high = self.high.to(x.device)

        ll = torch.outer(low, low).unsqueeze(0).unsqueeze(0)  # (1, 1, 2, 2)
        lh = torch.outer(low, high).unsqueeze(0).unsqueeze(0)
        hl = torch.outer(high, low).unsqueeze(0).unsqueeze(0)
        hh = torch.outer(high, high).unsqueeze(0).unsqueeze(0)

        ll = ll.repeat(C, 1, 1, 1)  # (C, 1, 2, 2)
        lh = lh.repeat(C, 1, 1, 1)
        hl = hl.repeat(C, 1, 1, 1)
        hh = hh.repeat(C, 1, 1, 1)

        x_ll = F.conv2d(x, ll, stride=2, groups=C)
        x_lh = F.conv2d(x, lh, stride=2, groups=C)
        x_hl = F.conv2d(x, hl, stride=2, groups=C)
        x_hh = F.conv2d(x, hh, stride=2, groups=C)

        return torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)


class IWT(nn.Module):
    """Inverse Wavelet Transform"""
    def __init__(self):
        super(IWT, self).__init__()
        self.low = torch.tensor([1., 1.]) / math.sqrt(2)
        self.high = torch.tensor([1., -1.]) / math.sqrt(2)

    def forward(self, x):
        """
        Input: x (B, 4*C, H, W) - [LL, LH, HL, HH] concatenated
        Output: (B, C, 2*H, 2*W)
        """
        B, C4, H, W = x.shape
        assert C4 % 4 == 0, "Channel dimension must be divisible by 4"
        C = C4 // 4

        x_ll = x[:, :C, :, :]
        x_lh = x[:, C:2*C, :, :]
        x_hl = x[:, 2*C:3*C, :, :]
        x_hh = x[:, 3*C:, :, :]

        low = self.low.to(x.device)
        high = self.high.to(x.device)

        ll = torch.outer(low, low).unsqueeze(0).unsqueeze(0) * 2  # (1, 1, 2, 2)
        lh = torch.outer(low, high).unsqueeze(0).unsqueeze(0) * 2
        hl = torch.outer(high, low).unsqueeze(0).unsqueeze(0) * 2
        hh = torch.outer(high, high).unsqueeze(0).unsqueeze(0) * 2

        ll = ll.repeat(C, 1, 1, 1)  # (C, 1, 2, 2)
        lh = lh.repeat(C, 1, 1, 1)
        hl = hl.repeat(C, 1, 1, 1)
        hh = hh.repeat(C, 1, 1, 1)

        x_ll_up = F.conv_transpose2d(x_ll, ll, stride=2, groups=C)
        x_lh_up = F.conv_transpose2d(x_lh, lh, stride=2, groups=C)
        x_hl_up = F.conv_transpose2d(x_hl, hl, stride=2, groups=C)
        x_hh_up = F.conv_transpose2d(x_hh, hh, stride=2, groups=C)

        return x_ll_up + x_lh_up + x_hl_up + x_hh_up


class TimeEmbedding(nn.Module):
    """Positional encoding for timesteps"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResBlock(nn.Module):
    """Residual block with time and emotion conditioning"""
    def __init__(self, in_channels, out_channels, time_dim, emotion_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_mlp = nn.Linear(time_dim, out_channels)
        self.emotion_mlp = nn.Linear(emotion_dim, out_channels)

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.dropout = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, time_emb, emotion_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        time_out = self.time_mlp(F.silu(time_emb))[:, :, None, None]
        emotion_out = self.emotion_mlp(F.silu(emotion_emb))[:, :, None, None]
        h = h + time_out + emotion_out

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class FrequencyBottleneckBlock(nn.Module):
    """Frequency bottleneck block - processes low-freq, passes high-freq"""
    def __init__(self, channels, time_dim, emotion_dim):
        super().__init__()
        self.dwt = DWT()
        self.iwt = IWT()

        self.low_freq_processor = nn.Sequential(
            ResBlock(channels, channels, time_dim, emotion_dim),
            ResBlock(channels, channels, time_dim, emotion_dim)
        )

    def forward(self, x, time_emb, emotion_emb):
        x_freq = self.dwt(x)  # (B, 4*C, H/2, W/2)

        C = x.shape[1]
        x_ll = x_freq[:, :C, :, :]
        x_hi = x_freq[:, C:, :, :]

        x_ll_processed = x_ll
        for layer in self.low_freq_processor:
            x_ll_processed = layer(x_ll_processed, time_emb, emotion_emb)

        x_freq_out = torch.cat([x_ll_processed, x_hi], dim=1)

        return self.iwt(x_freq_out)


class FreqAwareDownsample(nn.Module):
    """Frequency-aware downsampling block"""
    def __init__(self, in_channels, out_channels, time_dim, emotion_dim):
        super().__init__()
        self.dwt = DWT()
        self.conv = nn.Conv2d(4 * in_channels, out_channels, 1)
        self.norm = nn.GroupNorm(8, out_channels)

        self.time_mlp = nn.Linear(time_dim, out_channels)
        self.emotion_mlp = nn.Linear(emotion_dim, out_channels)

    def forward(self, x, time_emb, emotion_emb):
        x_freq = self.dwt(x)  # (B, 4*C, H/2, W/2)

        out = self.conv(x_freq)
        out = self.norm(out)

        time_out = self.time_mlp(F.silu(time_emb))[:, :, None, None]
        emotion_out = self.emotion_mlp(F.silu(emotion_emb))[:, :, None, None]
        out = out + time_out + emotion_out

        C = x.shape[1]
        hi_freq = x_freq[:, C:, :, :]

        return F.silu(out), hi_freq


class FreqAwareUpsample(nn.Module):
    """Frequency-aware upsampling block"""
    def __init__(self, in_channels, out_channels, time_dim, emotion_dim):
        super().__init__()
        self.iwt = IWT()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.norm = nn.GroupNorm(8, out_channels)

        self.time_mlp = nn.Linear(time_dim, out_channels)
        self.emotion_mlp = nn.Linear(emotion_dim, out_channels)

    def forward(self, x, hi_freq_skip, time_emb, emotion_emb):
        x_low = self.conv(x)
        x_low = self.norm(x_low)

        time_out = self.time_mlp(F.silu(time_emb))[:, :, None, None]
        emotion_out = self.emotion_mlp(F.silu(emotion_emb))[:, :, None, None]
        x_low = x_low + time_out + emotion_out
        x_low = F.silu(x_low)

        x_freq = torch.cat([x_low, hi_freq_skip], dim=1)

        return self.iwt(x_freq)


class WaveletResidualConnection(nn.Module):
    """Frequency residual connection using wavelet downsample"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dwt = DWT()
        self.conv = nn.Conv2d(4 * in_channels, out_channels, 1)

    def forward(self, x):
        x_freq = self.dwt(x)  # (B, 4*C, H/2, W/2)
        return self.conv(x_freq)


class WaveletUNet(nn.Module):
    """Wavelet-embedded U-Net for emotion-conditioned diffusion"""
    def __init__(self,
                 in_channels=12,
                 out_channels=12,
                 features=[64, 128, 256, 512],
                 time_dim=256,
                 emotion_dim=64,
                 num_emotions=8):
        super().__init__()

        self.time_embedding = TimeEmbedding(time_dim)
        self.emotion_embedding = nn.Embedding(num_emotions, emotion_dim)

        self.input_conv = nn.Conv2d(in_channels, features[0], 3, padding=1)

        self.aux_expr_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(features[-1], 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_emotions)
        )

        self.aux_va_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(features[-1], 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2)  # valence, arousal
        )

        self.encoder_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()

        for i in range(len(features) - 1):
            self.encoder_blocks.append(nn.ModuleList([
                ResBlock(features[i], features[i], time_dim, emotion_dim),
                ResBlock(features[i], features[i], time_dim, emotion_dim)
            ]))

            self.downsample_blocks.append(
                FreqAwareDownsample(features[i], features[i+1], time_dim, emotion_dim)
            )

        self.bottleneck = nn.ModuleList([
            FrequencyBottleneckBlock(features[-1], time_dim, emotion_dim),
            ResBlock(features[-1], features[-1], time_dim, emotion_dim),
            FrequencyBottleneckBlock(features[-1], time_dim, emotion_dim)
        ])

        self.decoder_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()

        for i in range(len(features) - 1, 0, -1):
            self.upsample_blocks.append(
                FreqAwareUpsample(features[i], features[i-1], time_dim, emotion_dim)
            )

            self.decoder_blocks.append(nn.ModuleList([
                ResBlock(features[i-1] * 2, features[i-1], time_dim, emotion_dim),  # *2 for skip connection
                ResBlock(features[i-1], features[i-1], time_dim, emotion_dim)
            ]))

        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, features[0]),
            nn.SiLU(),
            nn.Conv2d(features[0], out_channels, 3, padding=1)
        )

    def forward(self, x, t, emotion_id, src_image=None, return_aux=False):
        """
        Args:
            x: Noisy wavelet coefficients (B, 12, H/2, W/2)
            t: Timestep (B,)
            emotion_id: Target emotion ID (B,)
            src_image: Source RGB image for residual connections (B, 3, H, W)
            return_aux: Whether to return auxiliary predictions
        """
        time_emb = self.time_embedding(t)
        emotion_emb = self.emotion_embedding(emotion_id)

        x = self.input_conv(x)
        skip_connections = []
        hi_freq_skips = []

        for i, (encoder_block, downsample_block) in enumerate(
            zip(self.encoder_blocks, self.downsample_blocks)
        ):
            for block in encoder_block:
                x = block(x, time_emb, emotion_emb)

            skip_connections.append(x)

            x, hi_freq = downsample_block(x, time_emb, emotion_emb)
            hi_freq_skips.append(hi_freq)

        bottleneck_features = x
        for block in self.bottleneck:
            if isinstance(block, FrequencyBottleneckBlock):
                x = block(x, time_emb, emotion_emb)
            else:
                x = block(x, time_emb, emotion_emb)

        aux_features = x

        for i, (upsample_block, decoder_block) in enumerate(
            zip(self.upsample_blocks, self.decoder_blocks)
        ):
            hi_freq = hi_freq_skips[-(i+1)]
            x = upsample_block(x, hi_freq, time_emb, emotion_emb)

            skip = skip_connections[-(i+1)]
            x = torch.cat([x, skip], dim=1)

            for block in decoder_block:
                x = block(x, time_emb, emotion_emb)

        noise_pred = self.output_conv(x)

        if return_aux:
            expr_pred = self.aux_expr_head(aux_features)
            va_pred = self.aux_va_head(aux_features)
            return noise_pred, expr_pred, va_pred

        return noise_pred


class WaveletDiffusionModel(nn.Module):
    """Complete Wavelet Diffusion Model for Emotion Editing"""
    def __init__(self,
                 num_emotions=8,
                 num_timesteps=1000,
                 beta_start=1e-4,
                 beta_end=2e-2):
        super().__init__()

        self.num_timesteps = num_timesteps
        self.num_emotions = num_emotions

        self.dwt = DWT()
        self.iwt = IWT()

        self.unet = WaveletUNet(num_emotions=num_emotions)

        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

    def forward_process(self, x0, t, noise=None):
        """Forward diffusion process - add noise"""
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise, noise

    def forward(self, x, emotion_id, src_image=None):
        """Training forward pass"""
        x_wavelet = self.dwt(x)  # (B, 12, H/2, W/2)
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device)
        x_noisy, noise = self.forward_process(x_wavelet, t)
        noise_pred = self.unet(x_noisy, t, emotion_id, src_image)
        loss = F.l1_loss(noise_pred, noise)
        return loss

    @torch.no_grad()
    def sample(self, src_image, target_emotion_id, num_steps=50):
        """DDIM sampling for inference - Fixed implementation"""
        device = src_image.device
        B = src_image.shape[0]
        src_wavelet = self.dwt(src_image)
        x = torch.randn_like(src_wavelet)
        timesteps = torch.linspace(self.num_timesteps - 1, 0, num_steps, dtype=torch.long, device=device)

        for i, t in enumerate(timesteps):
            t_tensor = torch.full((B,), t.item(), device=device, dtype=torch.long)
            noise_pred = self.unet(x, t_tensor, target_emotion_id, src_image)
            alpha_t = self.alphas_cumprod[t.item()]
            alpha_prev = self.alphas_cumprod[timesteps[i+1].item()] if i < len(timesteps) - 1 else torch.tensor(1.0, device=device)
            alpha_t = alpha_t.to(device)
            alpha_prev = alpha_prev.to(device)
            pred_x0 = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            pred_x0 = torch.clamp(pred_x0, -3, 3)

            if i < len(timesteps) - 1:
                x = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1 - alpha_prev) * noise_pred
            else:
                x = pred_x0

        result = self.iwt(x)
        return torch.clamp(result, -1, 1)


def create_wavelet_diffusion_model(num_emotions=7):
    """Factory function to create the model"""
    return WaveletDiffusionModel(num_emotions=num_emotions)


if __name__ == "__main__":
    model = create_wavelet_diffusion_model(num_emotions=8)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Tổng số tham số: {total_params:,}")