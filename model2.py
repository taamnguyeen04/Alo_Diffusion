import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DWT(nn.Module):
    """Biến đổi wavelet rời rạc (DWT) + Chuẩn hoá đầu ra"""
    def __init__(self, normalize=True, mean_std=None):
        """
        Args:
            normalize (bool): Bật/tắt chuẩn hoá.
            mean_std (dict or None): Nếu cung cấp, dùng giá trị mean/std cố định
                {'LL': (mean, std), 'LH': (...), 'HL': (...), 'HH': (...)}.
                Nếu None, sẽ chuẩn hoá theo batch động.
        """
        super(DWT, self).__init__()
        self.low = torch.tensor([1., 1.]) / math.sqrt(2)
        self.high = torch.tensor([1., -1.]) / math.sqrt(2)
        self.normalize = normalize
        self.mean_std = {
            'LL': (-0.191473, 1.083653),
            'LH': (-0.000004, 0.069750),
            'HL': (-0.000353, 0.062807),
            'HH': (-0.000007, 0.022190),
        }

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, "Height and Width must be even"

        low = self.low.to(x.device)
        high = self.high.to(x.device)

        ll = torch.outer(low, low).unsqueeze(0).unsqueeze(0)
        lh = torch.outer(low, high).unsqueeze(0).unsqueeze(0)
        hl = torch.outer(high, low).unsqueeze(0).unsqueeze(0)
        hh = torch.outer(high, high).unsqueeze(0).unsqueeze(0)

        ll = ll.repeat(C, 1, 1, 1)
        lh = lh.repeat(C, 1, 1, 1)
        hl = hl.repeat(C, 1, 1, 1)
        hh = hh.repeat(C, 1, 1, 1)

        # Biến đổi
        x_ll = F.conv2d(x, ll, stride=2, groups=C)
        x_lh = F.conv2d(x, lh, stride=2, groups=C)
        x_hl = F.conv2d(x, hl, stride=2, groups=C)
        x_hh = F.conv2d(x, hh, stride=2, groups=C)

        # ====== 🔹 Bước CHUẨN HOÁ ======
        if self.normalize:
            if self.mean_std is not None:
                # 🔸 Chuẩn hoá theo giá trị thống kê cố định (toàn dataset)
                for i, (name, tensor) in enumerate(zip(
                    ['LL', 'LH', 'HL', 'HH'],
                    [x_ll, x_lh, x_hl, x_hh]
                )):
                    mean, std = self.mean_std[name]
                    tensor.sub_(mean).div_(std + 1e-8)
            else:
                # 🔸 Chuẩn hoá động theo batch (z-score)
                for tensor in [x_ll, x_lh, x_hl, x_hh]:
                    mean = tensor.mean(dim=[1,2,3], keepdim=True)
                    std = tensor.std(dim=[1,2,3], keepdim=True)
                    tensor.sub_(mean).div_(std + 1e-8)

        return torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)

class IWT(nn.Module):
    """Phép biến đổi ngược của DWT để tái tạo ảnh từ các thành phần tần số"""
    def __init__(self):
        super(IWT, self).__init__()
        self.low = torch.tensor([1., 1.]) / math.sqrt(2)
        self.high = torch.tensor([1., -1.]) / math.sqrt(2)

    def forward(self, x):
        B, C4, H, W = x.shape
        assert C4 % 4 == 0, "Channel dimension must be divisible by 4"
        C = C4 // 4

        x_ll = x[:, :C, :, :]*2
        x_lh = x[:, C:2*C, :, :]*2
        x_hl = x[:, 2*C:3*C, :, :]*2
        x_hh = x[:, 3*C:, :, :]*2

        low = self.low.to(x.device)
        high = self.high.to(x.device)

        ll = torch.outer(low, low).unsqueeze(0).unsqueeze(0) # (1, 1, 2, 2)
        lh = torch.outer(low, high).unsqueeze(0).unsqueeze(0)
        hl = torch.outer(high, low).unsqueeze(0).unsqueeze(0)
        hh = torch.outer(high, high).unsqueeze(0).unsqueeze(0)

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
    """Mã hóa thông tin timestep (số bước trong diffusion)"""
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


class CrossAttention(nn.Module):
    """Cơ chế cross-attention để trộn đặc trưng ảnh với embedding cảm xúc"""
    def __init__(self, channels, emotion_dim, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = nn.GroupNorm(8, channels)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(emotion_dim, channels)
        self.to_v = nn.Linear(emotion_dim, channels)
        self.to_out = nn.Linear(channels, channels)

    def forward(self, x, emotion_emb):
        B, C, H, W = x.shape

        x_norm = self.norm(x)
        x_flat = x_norm.view(B, C, H*W).transpose(1, 2)  # (B, H*W, C)

        emotion_expanded = emotion_emb.unsqueeze(1).expand(-1, H*W, -1)  # (B, H*W, emotion_dim)

        q = self.to_q(x_flat)  # (B, H*W, C)
        k = self.to_k(emotion_expanded)  # (B, H*W, C)
        v = self.to_v(emotion_expanded)  # (B, H*W, C)

        q = q.view(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, H*W, head_dim)
        k = k.view(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1)

        # Debug attention maps
        # print(f"Attention mean: {attn.abs().mean().item():.6f}")

        out = torch.matmul(attn, v)  # (B, heads, H*W, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, H*W, C)  # (B, H*W, C)
        out = self.to_out(out)  # (B, H*W, C)

        out = out.transpose(1, 2).view(B, C, H, W)  # (B, C, H, W)
        return x + out


class ResBlock(nn.Module):
    """Residual block kết hợp cả timestep và emotion embedding."""
    def __init__(self, in_channels, out_channels, time_dim, emotion_dim, dropout=0.1, use_cross_attn=False, use_film=True, use_adagn=False):
        super().__init__()
        self.use_cross_attn = use_cross_attn
        self.use_film = use_film
        self.use_adagn = use_adagn

        # Cannot use both FiLM and AdaGN simultaneously
        if self.use_film and self.use_adagn:
            raise ValueError("Cannot use both FiLM and AdaGN. Choose one.")

        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_mlp = nn.Linear(time_dim, out_channels)
        self.emotion_mlp = nn.Linear(emotion_dim, out_channels)

        # FiLM layers for emotion conditioning
        if self.use_film:
            self.film_gamma = nn.Linear(emotion_dim, out_channels)
            self.film_beta = nn.Linear(emotion_dim, out_channels)

            # Initialize FiLM to be identity at start: gamma=1, beta=0
            # This ensures stable training - starts with h' = 1*h + 0 = h
            # Then slowly learns to modulate features
            nn.init.zeros_(self.film_gamma.weight)
            nn.init.ones_(self.film_gamma.bias)   # gamma starts at 1
            nn.init.zeros_(self.film_beta.weight)
            nn.init.zeros_(self.film_beta.bias)   # beta starts at 0

        # AdaGN (Adaptive Group Normalization) for emotion conditioning
        if self.use_adagn:
            # Normalization layer without learnable affine parameters
            self.adagn_norm = nn.GroupNorm(8, out_channels, affine=False)

            # Single MLP predicts both gamma and beta (more efficient)
            self.emotion_modulation = nn.Linear(emotion_dim, 2 * out_channels)

            # Initialize AdaGN to be identity: gamma≈0, beta≈0
            # So h' = h_norm * (1 + 0) + 0 = h_norm (normalized features)
            nn.init.zeros_(self.emotion_modulation.weight)
            nn.init.zeros_(self.emotion_modulation.bias)

        if use_cross_attn:
            self.cross_attn = CrossAttention(out_channels, emotion_dim)

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.dropout = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, time_emb, emotion_emb, return_film_params=False):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        time_out = self.time_mlp(F.silu(time_emb))[:, :, None, None]
        emotion_out = self.emotion_mlp(F.silu(emotion_emb))[:, :, None, None]
        h = h + time_out + emotion_out

        # Apply emotion conditioning
        gamma = None
        beta = None

        if self.use_film:
            # FiLM conditioning: h' = gamma * h + beta
            # Initialized with gamma=1, beta=0 for stability
            emotion_emb_norm = F.layer_norm(emotion_emb, (emotion_emb.shape[-1],))
            emotion_feat = F.silu(emotion_emb_norm)

            gamma = self.film_gamma(emotion_feat)[:, :, None, None]
            beta = self.film_beta(emotion_feat)[:, :, None, None]
            h = gamma * h + beta

        elif self.use_adagn:
            # AdaGN conditioning: h' = h_norm * (1 + gamma) + beta
            # Step 1: Normalize features to mean=0, std=1 per group
            h_normalized = self.adagn_norm(h)

            # Step 2: Predict modulation parameters from emotion
            emotion_emb_norm = F.layer_norm(emotion_emb, (emotion_emb.shape[-1],))
            emotion_feat = F.silu(emotion_emb_norm)
            emotion_mod = self.emotion_modulation(emotion_feat)  # (B, 2*C)

            # Split into gamma and beta
            gamma, beta = torch.chunk(emotion_mod, 2, dim=1)  # Each: (B, C)
            gamma = gamma[:, :, None, None]
            beta = beta[:, :, None, None]

            # Step 3: Modulate normalized features
            # h' = h_norm * (1 + gamma) + beta
            # This allows gamma≈0, beta≈0 to produce identity transform
            h = h_normalized * (1 + gamma) + beta

        if self.use_cross_attn:
            h = self.cross_attn(h, emotion_emb)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        output = h + self.shortcut(x)

        if return_film_params:
            return output, gamma, beta
        return output


class FrequencyBottleneckBlock(nn.Module):
    """Xử lý đặc biệt cho tần số thấp, giữ nguyên tần số cao"""
    def __init__(self, channels, time_dim, emotion_dim, use_film=True, use_adagn=False):
        super().__init__()
        self.dwt = DWT()
        self.iwt = IWT()

        self.low_freq_processor = nn.Sequential(
            ResBlock(channels, channels, time_dim, emotion_dim, use_film=use_film, use_adagn=use_adagn),
            ResBlock(channels, channels, time_dim, emotion_dim, use_film=use_film, use_adagn=use_adagn),
            ResBlock(channels, channels, time_dim, emotion_dim, use_film=use_film, use_adagn=use_adagn)
        )

    def forward(self, x, time_emb, emotion_emb, return_film_params=False):
        x_freq = self.dwt(x)  # (B, 4*C, H/2, W/2)

        C = x.shape[1]
        x_ll = x_freq[:, :C, :, :]
        x_hi = x_freq[:, C:, :, :]

        x_ll_processed = x_ll
        film_params = []
        for layer in self.low_freq_processor:
            if return_film_params:
                x_ll_processed, gamma, beta = layer(x_ll_processed, time_emb, emotion_emb, return_film_params=True)
                film_params.append((gamma, beta))
            else:
                x_ll_processed = layer(x_ll_processed, time_emb, emotion_emb)

        x_freq_out = torch.cat([x_ll_processed, x_hi], dim=1)

        if return_film_params:
            return self.iwt(x_freq_out), film_params
        return self.iwt(x_freq_out)


class FreqAwareDownsample(nn.Module):
    """Downsampling có nhận biết tần số."""
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
    """Upsampling có nhận biết tần số"""
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
    """Kết nối tần số từ ảnh gốc (source image) để giữ đặc trưng nhận dạng."""
    def __init__(self, in_channels, out_channels, downsample_level=1):
        super().__init__()
        self.dwt = DWT()
        self.downsample_level = downsample_level

        final_channels = in_channels * (4 ** downsample_level)
        self.conv = nn.Conv2d(final_channels, out_channels, 1)

    def forward(self, x):
        for _ in range(self.downsample_level):
            x = self.dwt(x)  # Each DWT: (B, C, H, W) -> (B, 4*C, H/2, W/2)

        return self.conv(x)


class WaveletUNet(nn.Module):
    """UNet nhúng wavelet, có điều kiện cảm xúc."""
    def __init__(self,
                 in_channels=12,
                 out_channels=12,
                 features=[96, 192, 384, 768],
                 time_dim=256,
                 emotion_dim=64,
                 num_emotions=8,
                 use_film=True,
                 use_adagn=False):
        super().__init__()

        self.time_embedding = TimeEmbedding(time_dim)
        self.num_emotions = num_emotions
        self.emotion_dim = emotion_dim
        self.use_film = use_film
        self.use_adagn = use_adagn

        self.emotion_embedding = nn.Embedding(num_emotions, emotion_dim)

        self.input_conv = nn.Conv2d(in_channels, features[0], 3, padding=1)

        self.encoder_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()
        self.res_blocks = nn.ModuleList()

        for i in range(len(features) - 1):
            use_attn = i >= len(features) // 2

            self.encoder_blocks.append(nn.ModuleList([
                ResBlock(features[i], features[i], time_dim, emotion_dim, use_cross_attn=use_attn, use_film=use_film, use_adagn=use_adagn),
                ResBlock(features[i], features[i], time_dim, emotion_dim, use_cross_attn=use_attn, use_film=use_film, use_adagn=use_adagn),
                ResBlock(features[i], features[i], time_dim, emotion_dim, use_cross_attn=use_attn, use_film=use_film, use_adagn=use_adagn)
            ]))

            self.downsample_blocks.append(
                FreqAwareDownsample(features[i], features[i+1], time_dim, emotion_dim)
            )

            self.res_blocks.append(
                WaveletResidualConnection(3, features[i], downsample_level=i+1)
            )

        self.bottleneck = nn.ModuleList([
            FrequencyBottleneckBlock(features[-1], time_dim, emotion_dim, use_film=use_film, use_adagn=use_adagn),
            ResBlock(features[-1], features[-1], time_dim, emotion_dim, use_film=use_film, use_adagn=use_adagn),
            FrequencyBottleneckBlock(features[-1], time_dim, emotion_dim, use_film=use_film, use_adagn=use_adagn)
        ])

        self.decoder_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()

        for i in range(len(features) - 1, 0, -1):
            self.upsample_blocks.append(
                FreqAwareUpsample(features[i], features[i-1], time_dim, emotion_dim)
            )

            self.decoder_blocks.append(nn.ModuleList([
                ResBlock(features[i-1] * 2, features[i-1], time_dim, emotion_dim, use_film=use_film, use_adagn=use_adagn),
                ResBlock(features[i-1], features[i-1], time_dim, emotion_dim, use_film=use_film, use_adagn=use_adagn),
                ResBlock(features[i-1], features[i-1], time_dim, emotion_dim, use_film=use_film, use_adagn=use_adagn)
            ]))

        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, features[0]),
            nn.SiLU(),
            nn.Conv2d(features[0], out_channels, 3, padding=1)
        )

    def forward(self, x, t, emotion_id, src_image=None, return_aux=False, return_film_params=False):
        """
        Args:
            x: Noisy wavelet coefficients (B, 12, H/2, W/2)
            t: Timestep (B,)
            emotion_id: Target emotion ID (B,)
            src_image: Source RGB image for residual connections (B, 3, H, W)
            return_aux: Whether to return auxiliary predictions
            return_film_params: Whether to return FiLM gamma/beta parameters
        """
        time_emb = self.time_embedding(t)

        emotion_emb = self.emotion_embedding(emotion_id)

        x = self.input_conv(x)
        skip_connections = []
        hi_freq_skips = []
        film_params = []

        for i, (encoder_block, downsample_block) in enumerate(
            zip(self.encoder_blocks, self.downsample_blocks)
        ):
            for block in encoder_block:
                if return_film_params:
                    x, gamma, beta = block(x, time_emb, emotion_emb, return_film_params=True)
                    film_params.append((gamma, beta))
                else:
                    x = block(x, time_emb, emotion_emb)

            if src_image is not None:
                res_feat = self.res_blocks[i](src_image)
                x = x + 0.01 * res_feat  # Reduced from 0.05 to 0.01 for stability

            skip_connections.append(x)

            x, hi_freq = downsample_block(x, time_emb, emotion_emb)
            hi_freq_skips.append(hi_freq)

        for block in self.bottleneck:
            if isinstance(block, FrequencyBottleneckBlock):
                if return_film_params:
                    x, bottleneck_film = block(x, time_emb, emotion_emb, return_film_params=True)
                    film_params.extend(bottleneck_film)
                else:
                    x = block(x, time_emb, emotion_emb)
            else:
                if return_film_params:
                    x, gamma, beta = block(x, time_emb, emotion_emb, return_film_params=True)
                    film_params.append((gamma, beta))
                else:
                    x = block(x, time_emb, emotion_emb)

        for i, (upsample_block, decoder_block) in enumerate(
            zip(self.upsample_blocks, self.decoder_blocks)
        ):
            hi_freq = hi_freq_skips[-(i+1)]
            x = upsample_block(x, hi_freq, time_emb, emotion_emb)

            skip = skip_connections[-(i+1)]
            x = torch.cat([x, skip], dim=1)

            for block in decoder_block:
                if return_film_params:
                    x, gamma, beta = block(x, time_emb, emotion_emb, return_film_params=True)
                    film_params.append((gamma, beta))
                else:
                    x = block(x, time_emb, emotion_emb)

        noise_pred = self.output_conv(x)

        if return_film_params:
            return noise_pred, film_params
        return noise_pred


class WaveletDiffusionModel(nn.Module):
    """Mô hình diffusion hoàn chỉnh để chỉnh sửa cảm xúc."""
    def __init__(self,
                 num_emotions=8,
                 num_timesteps=200,
                 use_emotion_smooth=True,
                 smooth_weight=0.1,
                 use_film=True,
                 use_adagn=False):
        super().__init__()

        self.num_timesteps = num_timesteps
        self.num_emotions = num_emotions
        self.use_emotion_smooth = use_emotion_smooth
        self.smooth_weight = smooth_weight
        self.use_film = use_film
        self.use_adagn = use_adagn

        # ====== Các thành phần chính ======
        self.dwt = DWT()
        self.iwt = IWT()
        self.unet = WaveletUNet(num_emotions=num_emotions, use_film=use_film, use_adagn=use_adagn)

        # ====== Hook theo dõi gradient (nếu cần debug) ======
        def grad_hook(module, grad_in, grad_out):
            if grad_out[0] is not None:
                # print(f"UNet first conv grad mean: {grad_out[0].abs().mean().item():.6f}")
                pass
        self.unet.input_conv.register_full_backward_hook(grad_hook)

        # ============================================================
        # 🔹 COSINE β-SCHEDULE (Improved DDPM)
        # ============================================================
        import math, torch

        s = 0.008  # offset nhỏ để tránh alpha_bar = 0 ở cuối
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps, dtype=torch.float32)

        # ᾱ_t theo cosine (alpha_cumprod)
        alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # chuẩn hóa để ᾱ_0 = 1

        # β_t suy ra từ ᾱ_t
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = betas.clamp(0.0001, 0.9999)

        # α_t và tích lũy ᾱ_t
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)


        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]]))

        # Emotion wheel distances (psychological similarity)
        # 0:neutral, 1:happy, 2:sad, 3:angry, 4:surprised, 5:fearful, 6:disgusted, 7:contempt
        # Create distance matrix based on emotion wheel
        # self.register_buffer('emotion_distances', self._create_emotion_distance_matrix(num_emotions))

    def forward_process(self, x0, t, noise=None):
        """thêm nhiễu vào wavelet ở bước t."""
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise, noise

    def forward(self, x, emotion_id, src_image=None):
        """training → dự đoán nhiễu và tính loss."""
        x_wavelet = self.dwt(x)  # (B, 12, H/2, W/2)
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device)
        x_noisy, noise = self.forward_process(x_wavelet, t)
        noise_pred = self.unet(x_noisy, t, emotion_id, src_image)
        loss = F.l1_loss(noise_pred, noise)
        return loss

    @torch.no_grad()
    def sample(self, src_image, target_emotion_id, num_steps=100, denoising_strength=0.2, return_film_params=False):
        """
        inference → dùng DDIM sampling để sinh ảnh mới theo cảm xúc mục tiêu.

        Args:
            src_image: Ảnh gốc (B, 3, H, W)
            target_emotion_id: ID cảm xúc mục tiêu (B,)
            num_steps: Số bước sampling
            denoising_strength: Mức độ denoising (0.0-1.0).
                              1.0 = thay đổi hoàn toàn, 0.0 = không thay đổi
            return_film_params: Whether to return FiLM gamma/beta parameters from last step
        """
        device = src_image.device
        B = src_image.shape[0]
        src_wavelet = self.dwt(src_image)

        # Tính toán start timestep dựa trên denoising_strength
        # Với strength cao hơn = nhiễu nhiều hơn = thay đổi nhiều hơn
        start_timestep = int(denoising_strength * self.num_timesteps)
        start_timestep = max(1, start_timestep)  # Ít nhất là 1

        # Tạo timesteps từ start_timestep về 0
        timesteps = torch.linspace(start_timestep - 1, 0,
                                 min(num_steps, start_timestep),
                                 dtype=torch.long, device=device)

        # Luôn bắt đầu từ ảnh gốc có nhiễu
        if start_timestep > 0:
            # Thêm nhiễu vào ảnh gốc
            noise = torch.randn_like(src_wavelet)
            alpha_start = self.sqrt_alphas_cumprod[start_timestep]
            sigma_start = self.sqrt_one_minus_alphas_cumprod[start_timestep]
            x = alpha_start * src_wavelet + sigma_start * noise
        else:
            # Trường hợp đặc biệt: không thêm nhiễu (strength = 0)
            x = src_wavelet

        film_params = None
        for i, t in enumerate(timesteps):
            t_tensor = torch.full((B,), t.item(), device=device, dtype=torch.long)
            # Get FiLM params only on the last step
            is_last = (i == len(timesteps) - 1)
            if return_film_params and is_last:
                noise_pred, film_params = self.unet(x, t_tensor, target_emotion_id, src_image, return_film_params=True)
            else:
                noise_pred = self.unet(x, t_tensor, target_emotion_id, src_image)

            alpha_t = self.alphas_cumprod[t.item()]
            alpha_prev = self.alphas_cumprod[timesteps[i+1].item()] if i < len(timesteps) - 1 else torch.tensor(1.0, device=device)
            alpha_t = alpha_t.to(device)
            alpha_prev = alpha_prev.to(device)
            
            pred_x0 = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            # Softer clipping with tanh for smoother gradients
            pred_x0 = torch.tanh(pred_x0 / 3.0) * 5.0  # Maps large values smoothly to [-5, 5]

            if i < len(timesteps) - 1:
                x = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1 - alpha_prev) * noise_pred
            else:
                x = pred_x0

        result = self.iwt(x)

        if return_film_params:
            return result, film_params
        return result

    def sample_full_steps(self, src_image, target_emotion_id, denoising_strength=0.2):
        """
        Full step sampling using all timesteps for debugging/comparison
        """
        device = src_image.device
        B = src_image.shape[0]
        src_wavelet = self.dwt(src_image)

        # Use all timesteps for this version
        start_timestep = int(denoising_strength * self.num_timesteps)
        start_timestep = max(1, start_timestep)

        # Generate all timesteps from start to 0
        timesteps = torch.arange(start_timestep - 1, -1, -1, device=device)

        # Add noise to source image
        if start_timestep > 0:
            noise = torch.randn_like(src_wavelet)
            alpha_start = self.sqrt_alphas_cumprod[start_timestep]
            sigma_start = self.sqrt_one_minus_alphas_cumprod[start_timestep]
            x = alpha_start * src_wavelet + sigma_start * noise
        else:
            x = src_wavelet

        # Full timestep sampling
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
        return result

if __name__ == "__main__":
    # Khởi tạo mô hình
    model = WaveletDiffusionModel(
        num_emotions=8,
        num_timesteps=1000,
        use_film=True,
        use_adagn=False
    )

    # Đếm tổng số tham số
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 60)
    print("Wavelet Diffusion Model - Thống kê tham số")
    print("=" * 60)
    print(f"Tổng số tham số:        {total_params:,}")
    print(f"Tham số huấn luyện:     {trainable_params:,}")
    print(f"Tham số frozen:         {total_params - trainable_params:,}")
    print(f"Kích thước (MB):        {total_params * 4 / (1024**2):.2f}")
    print("=" * 60)

    # Đếm chi tiết theo từng thành phần
    print("\nChi tiết theo thành phần:")
    print("-" * 60)

    # UNet components
    unet_params = sum(p.numel() for p in model.unet.parameters())
    print(f"UNet tổng:              {unet_params:,}")

    # Encoder
    encoder_params = sum(p.numel() for block_list in model.unet.encoder_blocks
                        for block in block_list
                        for p in block.parameters())
    encoder_params += sum(p.numel() for block in model.unet.downsample_blocks
                         for p in block.parameters())
    print(f"  - Encoder:            {encoder_params:,}")

    # Bottleneck
    bottleneck_params = sum(p.numel() for block in model.unet.bottleneck
                           for p in block.parameters())
    print(f"  - Bottleneck:         {bottleneck_params:,}")

    # Decoder
    decoder_params = sum(p.numel() for block_list in model.unet.decoder_blocks
                        for block in block_list
                        for p in block.parameters())
    decoder_params += sum(p.numel() for block in model.unet.upsample_blocks
                         for p in block.parameters())
    print(f"  - Decoder:            {decoder_params:,}")

    # Embeddings
    time_emb_params = sum(p.numel() for p in model.unet.time_embedding.parameters())
    emotion_emb_params = sum(p.numel() for p in model.unet.emotion_embedding.parameters())
    print(f"  - Time embedding:     {time_emb_params:,}")
    print(f"  - Emotion embedding:  {emotion_emb_params:,}")

    # Input/Output conv
    input_conv_params = sum(p.numel() for p in model.unet.input_conv.parameters())
    output_conv_params = sum(p.numel() for p in model.unet.output_conv.parameters())
    print(f"  - Input conv:         {input_conv_params:,}")
    print(f"  - Output conv:        {output_conv_params:,}")

    # Residual connections
    res_params = sum(p.numel() for block in model.unet.res_blocks
                    for p in block.parameters())
    print(f"  - Residual conn:      {res_params:,}")

    print("=" * 60)

    # Test forward pass
    print("\nTest forward pass:")
    print("-" * 60)
    batch_size = 2
    img_size = 256

    x = torch.randn(batch_size, 3, img_size, img_size)
    emotion_id = torch.randint(0, 8, (batch_size,))

    print(f"Input shape:            {x.shape}")
    print(f"Emotion IDs:            {emotion_id.tolist()}")

    # Training forward
    model.train()
    loss = model(x, emotion_id, src_image=x)
    print(f"Training loss:          {loss.item():.6f}")

    # Inference forward
    model.eval()
    with torch.no_grad():
        output = model.sample(x, emotion_id, num_steps=10, denoising_strength=0.2)
    print(f"Output shape:           {output.shape}")
    print(f"Output range:           [{output.min().item():.3f}, {output.max().item():.3f}]")

    print("=" * 60)
    print("✓ Mô hình hoạt động bình thường!")
    print("=" * 60)
