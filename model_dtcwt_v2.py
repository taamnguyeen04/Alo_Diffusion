"""
DirectionalWaveDiffusion: DTCWT-Native Emotion Transfer Model (v2)

Architecture:
  - LL-only diffusion (3ch) + auxiliary Δmag/Δphase prediction
  - Source HF mag/phase conditioning via HFConditionEncoder
  - MagnitudePhaseModulator for per-orientation emotion control
  - UNet backbone with FreqAware down/upsample (DWT on features)

Key properties leveraged from DTCWT:
  - Shift-invariance: no aliasing artifacts
  - 6 directional subbands: emotion-specific edge modulation
  - Magnitude ≈ identity/texture, Phase ≈ edge position (sub-pixel)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pytorch_wavelets.dtcwt.transform2d import DTCWTForward, DTCWTInverse


# ============================================================================
# Color Space Conversion (BT.601)
# ============================================================================

def rgb_to_ycbcr(x):
    """Convert RGB to YCbCr. Input/output: (B, 3, H, W) in [0, 1]."""
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y  =  0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.169 * r - 0.331 * g + 0.500 * b + 0.5
    cr =  0.500 * r - 0.419 * g - 0.081 * b + 0.5
    return torch.cat([y, cb, cr], dim=1)

def ycbcr_to_rgb(x):
    """Convert YCbCr to RGB. Input/output: (B, 3, H, W)."""
    y, cb, cr = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    cb = cb - 0.5
    cr = cr - 0.5
    r = y + 1.403 * cr
    g = y - 0.344 * cb - 0.714 * cr
    b = y + 1.773 * cb
    return torch.cat([r, g, b], dim=1)


# ============================================================================
# DTCWT Wrappers (with YCbCr conversion)
# ============================================================================

class DTCWTWrapper(nn.Module):
    """
    RGB Image → YCbCr → DTCWT → (LL, magnitude, phase)
    
    Converts RGB to YCbCr BEFORE DTCWT so that:
    - Y channel (luminance/structure) is separated from Cb/Cr (color)
    - Diffusion can modify Y freely without causing color shifts
    - Channels 0-5 = Y×6orient, 6-11 = Cb×6orient, 12-17 = Cr×6orient
    
    Output:
        ll:    (B, 3, H, W)      — YCbCr lowpass at original resolution
        mag:   (B, 18, H/2, W/2) — magnitude (Y:0-5, Cb:6-11, Cr:12-17)
        phase: (B, 18, H/2, W/2) — phase (Y:0-5, Cb:6-11, Cr:12-17)
    """
    def __init__(self, biort='near_sym_b', qshift='qshift_b'):
        super().__init__()
        self.dtcwt = DTCWTForward(J=1, biort=biort, qshift=qshift)
    
    def forward(self, x):
        # DTCWT requires fp32 — disable autocast for compatibility
        with torch.amp.autocast('cuda', enabled=False):
            x = x.float()
            
            # Convert RGB → YCbCr before DTCWT
            x_ycbcr = rgb_to_ycbcr(x)
            
            yl, yh = self.dtcwt(x_ycbcr)
            # yl: (B, 3, H, W) — YCbCr lowpass at original resolution
            # yh[0]: (B, 3, 6, H/2, W/2, 2)
            
            hf = yh[0]
            hf_real = hf[..., 0]  # (B, 3, 6, H/2, W/2)
            hf_imag = hf[..., 1]
            
            B, C, D, H, W = hf_real.shape
            hf_real = hf_real.reshape(B, C * D, H, W)
            hf_imag = hf_imag.reshape(B, C * D, H, W)
            
            mag = torch.sqrt(hf_real ** 2 + hf_imag ** 2 + 1e-8)
            phase = torch.atan2(hf_imag, hf_real)
        
        return yl, mag, phase  # yl is YCbCr at H×W, mag/phase at H/2×W/2


class IDTCWTWrapper(nn.Module):
    """
    (LL_YCbCr, HF_real, HF_imag) → IDTCWT → YCbCr → RGB Image
    
    LL and HF are in YCbCr space. After IDTCWT reconstruction,
    converts back to RGB.
    """
    def __init__(self, biort='near_sym_b', qshift='qshift_b'):
        super().__init__()
        self.idtcwt = DTCWTInverse(biort=biort, qshift=qshift)
    
    def forward(self, ll, hf_real, hf_imag):
        """
        Args:
            ll:      (B, 3, H, W)     — YCbCr LL at original resolution
            hf_real: (B, 18, H/2, W/2)
            hf_imag: (B, 18, H/2, W/2)
        Returns:
            image: (B, 3, H, W) — RGB image
        """
        # IDTCWT requires fp32 — disable autocast for compatibility
        with torch.amp.autocast('cuda', enabled=False):
            ll = ll.float()
            hf_real = hf_real.float()
            hf_imag = hf_imag.float()
            
            B = ll.shape[0]
            H_hf, W_hf = hf_real.shape[2], hf_real.shape[3]
            
            hf_real = hf_real.reshape(B, 3, 6, H_hf, W_hf)
            hf_imag = hf_imag.reshape(B, 3, 6, H_hf, W_hf)
            hf = torch.stack([hf_real, hf_imag], dim=-1)
            
            ycbcr = self.idtcwt((ll, [hf]))
            
            # Convert YCbCr → RGB
            return ycbcr_to_rgb(ycbcr)


# ============================================================================
# Core Building Blocks (reused from model.py with adaptations)
# ============================================================================

class DWT(nn.Module):
    """Haar DWT for internal feature map downsampling."""
    def __init__(self):
        super().__init__()
        self.low = torch.tensor([1., 1.]) / math.sqrt(2)
        self.high = torch.tensor([1., -1.]) / math.sqrt(2)

    def forward(self, x):
        B, C, H, W = x.shape
        if H % 2 != 0:
            x = F.pad(x, [0, 0, 0, 1], mode='reflect')
        if W % 2 != 0:
            x = F.pad(x, [0, 1, 0, 0], mode='reflect')
        low = self.low.to(x.device).type(x.dtype)
        high = self.high.to(x.device).type(x.dtype)

        ll = torch.outer(low, low).unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)
        lh = torch.outer(low, high).unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)
        hl = torch.outer(high, low).unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)
        hh = torch.outer(high, high).unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)

        return torch.cat([
            F.conv2d(x, ll, stride=2, groups=C),
            F.conv2d(x, lh, stride=2, groups=C),
            F.conv2d(x, hl, stride=2, groups=C),
            F.conv2d(x, hh, stride=2, groups=C)
        ], dim=1)


class IWT(nn.Module):
    """Haar IWT for internal feature map upsampling."""
    def __init__(self):
        super().__init__()
        self.low = torch.tensor([1., 1.]) / math.sqrt(2)
        self.high = torch.tensor([1., -1.]) / math.sqrt(2)

    def forward(self, x):
        B, C4, H, W = x.shape
        C = C4 // 4
        x_ll, x_lh, x_hl, x_hh = x[:, :C], x[:, C:2*C], x[:, 2*C:3*C], x[:, 3*C:]
        
        low = self.low.to(x.device).type(x.dtype)
        high = self.high.to(x.device).type(x.dtype)
        
        ll = torch.outer(low, low).unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)
        lh = torch.outer(low, high).unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)
        hl = torch.outer(high, low).unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)
        hh = torch.outer(high, high).unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)
        
        return (F.conv_transpose2d(x_ll, ll, stride=2, groups=C) +
                F.conv_transpose2d(x_lh, lh, stride=2, groups=C) +
                F.conv_transpose2d(x_hl, hl, stride=2, groups=C) +
                F.conv_transpose2d(x_hh, hh, stride=2, groups=C))


class TimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


class CrossAttention(nn.Module):
    """Cross-attention between spatial features and emotion embedding."""
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
        x_flat = x_norm.view(B, C, H*W).transpose(1, 2)
        emotion_expanded = emotion_emb.unsqueeze(1).expand(-1, H*W, -1)
        
        q = self.to_q(x_flat).view(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(emotion_expanded).view(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(emotion_expanded).view(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, H*W, C)
        out = self.to_out(out).transpose(1, 2).view(B, C, H, W)
        return x + out


class ResBlock(nn.Module):
    """Residual block with timestep + emotion conditioning (FiLM or AdaGN)."""
    def __init__(self, in_channels, out_channels, time_dim, emotion_dim, 
                 dropout=0.1, use_cross_attn=False, use_film=True, use_adagn=False):
        super().__init__()
        self.use_cross_attn = use_cross_attn
        self.use_film = use_film
        self.use_adagn = use_adagn

        if self.use_film and self.use_adagn:
            raise ValueError("Cannot use both FiLM and AdaGN. Choose one.")

        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_channels)
        self.emotion_mlp = nn.Linear(emotion_dim, out_channels)

        if self.use_film:
            self.film_gamma = nn.Linear(emotion_dim, out_channels)
            self.film_beta = nn.Linear(emotion_dim, out_channels)
            nn.init.zeros_(self.film_gamma.weight)
            nn.init.ones_(self.film_gamma.bias)
            nn.init.zeros_(self.film_beta.weight)
            nn.init.zeros_(self.film_beta.bias)

        if self.use_adagn:
            self.adagn_norm = nn.GroupNorm(8, out_channels, affine=False)
            self.emotion_modulation = nn.Linear(emotion_dim, 2 * out_channels)
            nn.init.zeros_(self.emotion_modulation.weight)
            nn.init.zeros_(self.emotion_modulation.bias)

        if use_cross_attn:
            self.cross_attn = CrossAttention(out_channels, emotion_dim)

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb, emotion_emb):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_mlp(F.silu(time_emb))[:, :, None, None]
        h = h + self.emotion_mlp(F.silu(emotion_emb))[:, :, None, None]

        if self.use_film:
            emotion_feat = F.silu(F.layer_norm(emotion_emb, (emotion_emb.shape[-1],)))
            gamma = self.film_gamma(emotion_feat)[:, :, None, None]
            beta = self.film_beta(emotion_feat)[:, :, None, None]
            h = gamma * h + beta
        elif self.use_adagn:
            h_norm = self.adagn_norm(h)
            emotion_feat = F.silu(F.layer_norm(emotion_emb, (emotion_emb.shape[-1],)))
            gamma, beta = torch.chunk(self.emotion_modulation(emotion_feat), 2, dim=1)
            h = h_norm * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

        if self.use_cross_attn:
            h = self.cross_attn(h, emotion_emb)

        h = F.silu(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.shortcut(x)


# ============================================================================
# Frequency-Aware Down/Upsample (DWT on features, same pattern as model.py)
# ============================================================================

class FrequencyBottleneckBlock(nn.Module):
    """Process LL features only, keep HF features unchanged."""
    def __init__(self, channels, time_dim, emotion_dim, use_film=True, use_adagn=False):
        super().__init__()
        self.dwt = DWT()
        self.iwt = IWT()
        self.low_freq_processor = nn.Sequential(
            ResBlock(channels, channels, time_dim, emotion_dim, use_film=use_film, use_adagn=use_adagn),
            ResBlock(channels, channels, time_dim, emotion_dim, use_film=use_film, use_adagn=use_adagn)
        )

    def forward(self, x, time_emb, emotion_emb):
        x_freq = self.dwt(x)
        C = x.shape[1]
        x_ll = x_freq[:, :C, :, :]
        x_hi = x_freq[:, C:, :, :]
        for layer in self.low_freq_processor:
            x_ll = layer(x_ll, time_emb, emotion_emb)
        return self.iwt(torch.cat([x_ll, x_hi], dim=1))


class FreqAwareDownsample(nn.Module):
    """DWT-based downsampling on feature maps. Stores HF as skip."""
    def __init__(self, in_channels, out_channels, time_dim, emotion_dim):
        super().__init__()
        self.dwt = DWT()
        self.conv = nn.Conv2d(4 * in_channels, out_channels, 1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.time_mlp = nn.Linear(time_dim, out_channels)
        self.emotion_mlp = nn.Linear(emotion_dim, out_channels)

    def forward(self, x, time_emb, emotion_emb):
        x_freq = self.dwt(x)  # (B, 4*C, H/2, W/2)
        out = self.norm(self.conv(x_freq))
        out = out + self.time_mlp(F.silu(time_emb))[:, :, None, None]
        out = out + self.emotion_mlp(F.silu(emotion_emb))[:, :, None, None]
        C = x.shape[1]
        hi_freq = x_freq[:, C:, :, :]
        return F.silu(out), hi_freq


class FreqAwareUpsample(nn.Module):
    """IWT-based upsampling on feature maps. Merges with HF skip."""
    def __init__(self, in_channels, out_channels, time_dim, emotion_dim):
        super().__init__()
        self.iwt = IWT()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.time_mlp = nn.Linear(time_dim, out_channels)
        self.emotion_mlp = nn.Linear(emotion_dim, out_channels)

    def forward(self, x, hi_freq_skip, time_emb, emotion_emb):
        x_low = F.silu(self.norm(self.conv(x)))
        x_low = x_low + self.time_mlp(F.silu(time_emb))[:, :, None, None]
        x_low = x_low + self.emotion_mlp(F.silu(emotion_emb))[:, :, None, None]
        x_freq = torch.cat([x_low, hi_freq_skip], dim=1)
        return self.iwt(x_freq)


class WaveletResidualConnection(nn.Module):
    """Source image features injected at each encoder level."""
    def __init__(self, in_channels, out_channels, downsample_level=1):
        super().__init__()
        self.dwt = DWT()
        self.downsample_level = downsample_level
        final_channels = in_channels * (4 ** downsample_level)
        self.conv = nn.Conv2d(final_channels, out_channels, 1)

    def forward(self, x):
        for _ in range(self.downsample_level):
            x = self.dwt(x)
        return self.conv(x)


# ============================================================================
# DTCWT-Specific Modules (Novelty)
# ============================================================================

class HFConditionEncoder(nn.Module):
    """
    Encode source HF magnitude+phase into features at each UNet level.
    
    HF is at H/2×W/2, but UNet starts at H×W (LL resolution).
    After each FreqAwareDownsample, UNet is at H/2, H/4, H/8...
    So we need features at: H×W (level 0), H/2 (level 1), H/4 (level 2)
    
    We upsample first level to H×W, and pool for deeper levels.
    
    Input: source mag+phase concatenated (B, 36, H/2, W/2)
    Output: list of features matching UNet encoder scales
    """
    def __init__(self, in_channels=36, features=None):
        super().__init__()
        if features is None:
            features = [48, 96, 192, 384]
        
        self.num_levels = len(features) - 1  # Number of encoder levels
        
        # Initial projection at H/2 resolution
        self.init_proj = nn.Sequential(
            nn.Conv2d(in_channels, features[0], 3, padding=1),
            nn.GroupNorm(8, features[0]),
            nn.SiLU()
        )
        
        # Multi-scale projections for each encoder level
        self.level_projs = nn.ModuleList()
        for i in range(self.num_levels):
            self.level_projs.append(nn.Sequential(
                nn.Conv2d(features[0], features[i], 1),
                nn.GroupNorm(8, features[i]),
                nn.SiLU()
            ))
        
        self.pool = nn.AvgPool2d(2)
    
    def forward(self, hf_mag_phase, ll_size):
        """
        Args:
            hf_mag_phase: (B, 36, H/2, W/2)
            ll_size: (H/2, W/2) — spatial size of LL after downsample (= UNet level 0)
        Returns:
            list of features: level 0 at H/2, level 1 at H/4, level 2 at H/8
        """
        # Project at H/2 — same as UNet input resolution
        x = self.init_proj(hf_mag_phase)  # (B, feat[0], H/2, W/2)
        
        features = []
        for i in range(self.num_levels):
            if i == 0:
                # Level 0: UNet is at H/2 — direct projection, no resize needed
                features.append(self.level_projs[i](x))
            else:
                # Level i: UNet is at H/2^(i+1) → pool accordingly
                target_h = ll_size[0] // (2 ** i)
                target_w = ll_size[1] // (2 ** i)
                x_pooled = F.adaptive_avg_pool2d(x, (target_h, target_w))
                features.append(self.level_projs[i](x_pooled))
        
        return features


class MagnitudeModulator(nn.Module):
    """
    Per-orientation magnitude modulation based on emotion.
    
    Key idea from DTCWT theory:
    - Direction attention: each emotion affects different edge orientations
    - 6 orientations at ±15°, ±45°, ±75° corresponding to facial features
    
    Output: Δmag bias added to UNet's magnitude predictions.
    Phase is now handled by Analytic Phase Transport (APT), not regression.
    """
    def __init__(self, emotion_dim, n_orient=6, ch_per_orient=3):
        super().__init__()
        self.n_orient = n_orient
        self.ch_per_orient = ch_per_orient
        
        # Direction attention: emotion → which orientations to change
        self.dir_attn = nn.Sequential(
            nn.Linear(emotion_dim, 32),
            nn.SiLU(),
            nn.Linear(32, n_orient),
            nn.Sigmoid()
        )
        
        # Per-orientation delta predictor: emotion → Δmag per YCbCr
        self.delta_pred = nn.Sequential(
            nn.Linear(emotion_dim, 64),
            nn.SiLU(),
            nn.Linear(64, ch_per_orient)  # 3 Δmag (Y, Cb, Cr)
        )
        
        # Learnable scale, initialized small for stable training
        self.mag_scale = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, emotion_emb):
        """
        Args:
            emotion_emb: (B, emotion_dim)
        Returns:
            delta_mag: (B, 18, 1, 1) — per-orientation magnitude bias
        """
        dir_w = self.dir_attn(emotion_emb)    # (B, 6)
        deltas = self.delta_pred(emotion_emb)  # (B, 3)
        
        d_mag_list = []
        for d in range(self.n_orient):
            w = dir_w[:, d:d+1, None, None]  # (B, 1, 1, 1)
            dm = deltas[:, :self.ch_per_orient, None, None] * w * self.mag_scale
            d_mag_list.append(dm)
        
        delta_mag = torch.cat(d_mag_list, dim=1)  # (B, 18, 1, 1)
        return delta_mag


# ============================================================================
# Analytic Phase Transport (APT)
# ============================================================================

def _wrap_phase(x):
    """Wrap angle to [-π, π]."""
    return torch.atan2(torch.sin(x), torch.cos(x))


class AnalyticPhaseTransport(nn.Module):
    """
    Estimate displacement u(x) from DTCWT phase difference between
    source LL and denoised LL, then warp source HF coefficients.
    
    This replaces black-box Δphase regression with geometry-grounded
    phase transport using the DTCWT shift theorem:
        Δϕ_{s,d}(x) ≈ k_{s,d}^T · u(x)
    
    With 6 orientations providing an overdetermined system (6 eqs, 2 unknowns),
    we solve for u(x) = (uy, ux) per pixel via weighted least-squares.
    """
    
    # DTCWT 6 orientations in radians: 15°, 45°, 75°, 105°, 135°, 165°
    ORIENT_ANGLES = [
        15 * math.pi / 180,
        45 * math.pi / 180,
        75 * math.pi / 180,
        105 * math.pi / 180,
        135 * math.pi / 180,
        165 * math.pi / 180,
    ]
    
    def __init__(self, biort='near_sym_b', qshift='qshift_b', J_apt=2,
                 mag_eps=1e-3, smooth_sigma=1.5):
        super().__init__()
        self.J_apt = J_apt
        self.mag_eps = mag_eps
        
        # DTCWT for displacement estimation (Y channel only, J=2)
        self.dtcwt_apt = DTCWTForward(J=J_apt, biort=biort, qshift=qshift)
        
        # Precompute wave-vector matrix K for each scale
        # K[s] shape (6, 2): each row = k_{s,d} = (ky, kx)
        # At scale s, center frequency ~ π / 2^s
        for s in range(J_apt):
            freq = math.pi / (2 ** (s + 1))
            K_s = torch.zeros(6, 2)
            for d, theta in enumerate(self.ORIENT_ANGLES):
                K_s[d, 0] = freq * math.sin(theta)  # ky
                K_s[d, 1] = freq * math.cos(theta)  # kx
            self.register_buffer(f'K_scale{s}', K_s)
        
        # Gaussian kernel for smoothing displacement field
        if smooth_sigma > 0:
            ks = int(4 * smooth_sigma + 1) | 1  # ensure odd
            ax = torch.arange(ks, dtype=torch.float32) - ks // 2
            gauss = torch.exp(-0.5 * (ax / smooth_sigma) ** 2)
            kernel_1d = gauss / gauss.sum()
            kernel_2d = torch.outer(kernel_1d, kernel_1d)
            kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # (1,1,ks,ks)
            self.register_buffer('smooth_kernel', kernel_2d)
            self.smooth_pad = ks // 2
        else:
            self.smooth_kernel = None
    
    def _extract_y_phases_mags(self, ll_y):
        """
        Run DTCWT on Y channel and extract phases/magnitudes per scale.
        
        Args:
            ll_y: (B, 1, H, W) — Y channel of LL
        Returns:
            phases: list of (B, 6, Hs, Ws) per scale
            mags:   list of (B, 6, Hs, Ws) per scale
        """
        with torch.amp.autocast('cuda', enabled=False):
            ll_y = ll_y.float()
            _, yh = self.dtcwt_apt(ll_y)
            
            phases, mags = [], []
            for s in range(self.J_apt):
                hf = yh[s]  # (B, 1, 6, Hs, Ws, 2)
                real = hf[..., 0].squeeze(1)  # (B, 6, Hs, Ws)
                imag = hf[..., 1].squeeze(1)
                mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
                phase = torch.atan2(imag, real)
                phases.append(phase)
                mags.append(mag)
            
            return phases, mags
    
    def solve_displacement(self, ll_y_src, ll_y_tgt):
        """
        Estimate per-pixel displacement u(x) from phase differences.
        
        Coarse-to-fine: solve at coarsest scale first, then refine.
        
        Args:
            ll_y_src: (B, 1, H, W) — Y channel of source LL
            ll_y_tgt: (B, 1, H, W) — Y channel of target/denoised LL
        Returns:
            u: (B, 2, H1, W1) — displacement field at scale-1 resolution
               u[:,0] = dy, u[:,1] = dx (pixel units)
        """
        phases_src, mags_src = self._extract_y_phases_mags(ll_y_src)
        phases_tgt, mags_tgt = self._extract_y_phases_mags(ll_y_tgt)
        
        u = None  # will be initialized at coarsest scale
        
        # Coarse-to-fine: from scale J-1 (coarsest) to scale 0 (finest)
        for s in range(self.J_apt - 1, -1, -1):
            phase_src = phases_src[s]  # (B, 6, Hs, Ws)
            phase_tgt = phases_tgt[s]
            mag_src = mags_src[s]
            mag_tgt = mags_tgt[s]
            
            B, _, Hs, Ws = phase_src.shape
            K_s = getattr(self, f'K_scale{s}')  # (6, 2)
            
            # Phase difference with wrapping
            dphi = _wrap_phase(phase_tgt - phase_src)  # (B, 6, Hs, Ws)
            
            # If we have a coarser estimate, subtract predicted phase shift
            if u is not None:
                u_up = F.interpolate(u, size=(Hs, Ws), mode='bilinear',
                                     align_corners=False)
                # Predicted phase from current u: k·u for each orientation
                # K_s: (6,2), u_up: (B,2,H,W) → pred: (B,6,H,W)
                pred_phase = (K_s[:, 0].view(1, 6, 1, 1) * u_up[:, 0:1] +
                              K_s[:, 1].view(1, 6, 1, 1) * u_up[:, 1:2])
                dphi = _wrap_phase(dphi - pred_phase)
            
            # Confidence weights: minimum of src/tgt magnitude
            alpha = torch.minimum(mag_src, mag_tgt)  # (B, 6, Hs, Ws)
            alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-8)
            
            # Weighted least-squares per pixel: solve K^T diag(α) K u = K^T diag(α) Δϕ
            # K: (6,2), α: (B,6,H,W), Δϕ: (B,6,H,W)
            # A = K^T diag(α) K → (B, 2, 2, H, W)
            # b = K^T diag(α) Δϕ → (B, 2, H, W)
            
            Ky = K_s[:, 0]  # (6,)
            Kx = K_s[:, 1]  # (6,)
            
            # Weighted components
            aKy = alpha * Ky.view(1, 6, 1, 1)   # (B,6,H,W)
            aKx = alpha * Kx.view(1, 6, 1, 1)
            
            # 2x2 normal matrix entries
            a11 = (aKy * Ky.view(1, 6, 1, 1)).sum(dim=1)  # (B,H,W)
            a12 = (aKy * Kx.view(1, 6, 1, 1)).sum(dim=1)
            a22 = (aKx * Kx.view(1, 6, 1, 1)).sum(dim=1)
            
            # RHS
            b1 = (aKy * dphi).sum(dim=1)  # (B,H,W)
            b2 = (aKx * dphi).sum(dim=1)
            
            # Solve 2x2 system: det = a11*a22 - a12^2
            det = a11 * a22 - a12 * a12 + 1e-8
            du_y = (a22 * b1 - a12 * b2) / det
            du_x = (a11 * b2 - a12 * b1) / det
            
            du = torch.stack([du_y, du_x], dim=1)  # (B, 2, Hs, Ws)
            
            if u is not None:
                u = u_up + du
            else:
                u = du
        
        # Smooth displacement field
        if self.smooth_kernel is not None:
            pad = self.smooth_pad
            kernel = self.smooth_kernel  # (1,1,ks,ks)
            u_y = F.conv2d(F.pad(u[:, 0:1], [pad]*4, mode='reflect'),
                           kernel)
            u_x = F.conv2d(F.pad(u[:, 1:2], [pad]*4, mode='reflect'),
                           kernel)
            u = torch.cat([u_y, u_x], dim=1)
        
        return u
    
    def _make_warp_grid(self, u):
        """
        Convert pixel displacement u to normalized grid for F.grid_sample.
        
        Args:
            u: (B, 2, H, W) — displacement in pixels (dy, dx)
        Returns:
            grid: (B, H, W, 2) — normalized grid for grid_sample (x, y order)
        """
        B, _, H, W = u.shape
        # Base grid: identity mapping
        yy = torch.arange(H, device=u.device, dtype=u.dtype)
        xx = torch.arange(W, device=u.device, dtype=u.dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')
        
        # Add displacement
        new_y = grid_y.unsqueeze(0) + u[:, 0]  # (B, H, W)
        new_x = grid_x.unsqueeze(0) + u[:, 1]
        
        # Normalize to [-1, 1] for grid_sample
        new_y = 2.0 * new_y / max(H - 1, 1) - 1.0
        new_x = 2.0 * new_x / max(W - 1, 1) - 1.0
        
        # grid_sample expects (x, y) order
        grid = torch.stack([new_x, new_y], dim=-1)  # (B, H, W, 2)
        return grid
    
    def warp_coefficients(self, src_mag, src_phase, u):
        """
        Warp HF magnitude and phase using displacement field u.
        
        Phase gets additional correction from shift theorem: ϕ' = warp(ϕ) + k·u
        
        Args:
            src_mag:   (B, 18, H/2, W/2) — source HF magnitude
            src_phase: (B, 18, H/2, W/2) — source HF phase
            u:         (B, 2, Hu, Wu)    — displacement field
        Returns:
            warped_mag:   (B, 18, H/2, W/2)
            warped_phase: (B, 18, H/2, W/2)
        """
        _, _, Hm, Wm = src_mag.shape
        
        # Resize u to match HF resolution if needed
        if u.shape[2] != Hm or u.shape[3] != Wm:
            u = F.interpolate(u, size=(Hm, Wm), mode='bilinear',
                              align_corners=False)
            # Scale displacement proportionally
            u = u * (Hm / u.shape[2]) if u.shape[2] != Hm else u
        
        grid = self._make_warp_grid(u)
        
        # Warp magnitude (bilinear)
        warped_mag = F.grid_sample(src_mag, grid, mode='bilinear',
                                   padding_mode='border', align_corners=False)
        
        # Warp phase (bilinear on sin/cos to avoid discontinuity)
        sin_phase = torch.sin(src_phase)
        cos_phase = torch.cos(src_phase)
        warped_sin = F.grid_sample(sin_phase, grid, mode='bilinear',
                                   padding_mode='border', align_corners=False)
        warped_cos = F.grid_sample(cos_phase, grid, mode='bilinear',
                                   padding_mode='border', align_corners=False)
        warped_phase = torch.atan2(warped_sin, warped_cos)
        
        # Phase correction from shift theorem: add k·u for scale 1
        K_1 = self.K_scale0  # (6, 2) — scale 0 = finest scale
        # Apply correction per orientation, for all 3 color channels (Y,Cb,Cr)
        for d in range(6):
            phase_corr = K_1[d, 0] * u[:, 0:1] + K_1[d, 1] * u[:, 1:2]  # (B,1,H,W)
            # Channels d, d+6, d+12 correspond to Y, Cb, Cr for orientation d
            warped_phase[:, d:d+1] = warped_phase[:, d:d+1] + phase_corr
            warped_phase[:, d+6:d+7] = warped_phase[:, d+6:d+7] + phase_corr
            warped_phase[:, d+12:d+13] = warped_phase[:, d+12:d+13] + phase_corr
        
        # Wrap to [-π, π]
        warped_phase = _wrap_phase(warped_phase)
        
        return warped_mag, warped_phase
    
    def warp_haar_detail(self, ll_hf_skip, u):
        """
        Warp Haar detail bands (LH+HL+HH) using displacement field.
        
        Args:
            ll_hf_skip: (B, 9, H/2, W/2) — Haar HF skip connection
            u:          (B, 2, Hu, Wu)    — displacement field
        Returns:
            warped_skip: (B, 9, H/2, W/2)
        """
        _, _, Hs, Ws = ll_hf_skip.shape
        
        if u.shape[2] != Hs or u.shape[3] != Ws:
            u = F.interpolate(u, size=(Hs, Ws), mode='bilinear',
                              align_corners=False)
        
        grid = self._make_warp_grid(u)
        warped_skip = F.grid_sample(ll_hf_skip, grid, mode='bilinear',
                                    padding_mode='border', align_corners=False)
        return warped_skip


# ============================================================================
# Main UNet
# ============================================================================

class DirectionalWaveDiffusionUNet(nn.Module):
    """
    UNet for LL-only diffusion with auxiliary Δmag head.
    
    Input: noisy LL (3ch) + emotion map (Ech) 
    Conditioning: source HF features injected at each encoder level
    Output: noise_pred (3ch), Δmag (18ch)
    Phase is handled by Analytic Phase Transport (APT), not regression.
    """
    def __init__(self,
                 in_channels=11,    # 3 LL + 8 emotion one-hot
                 features=None,
                 time_dim=256,
                 emotion_dim=64,
                 num_emotions=8,
                 use_film=True,
                 use_adagn=False):
        super().__init__()
        if features is None:
            features = [48, 96, 192, 384]
        
        self.time_embedding = TimeEmbedding(time_dim)
        self.num_emotions = num_emotions
        self.emotion_dim = emotion_dim
        
        # Emotion embedding: id → learned vector
        self.emotion_embedding = nn.Embedding(num_emotions, emotion_dim)
        
        # Input conv: (3 + num_emotions) → features[0]
        self.input_conv = nn.Conv2d(in_channels, features[0], 3, padding=1)
        
        # HF condition encoder
        self.hf_encoder = HFConditionEncoder(in_channels=36, features=features)
        self.hf_inject_scale = 0.05
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()
        self.res_blocks = nn.ModuleList()
        
        for i in range(len(features) - 1):
            use_attn = i >= len(features) // 2
            self.encoder_blocks.append(nn.ModuleList([
                ResBlock(features[i], features[i], time_dim, emotion_dim,
                         use_cross_attn=use_attn, use_film=use_film, use_adagn=use_adagn),
                ResBlock(features[i], features[i], time_dim, emotion_dim,
                         use_cross_attn=use_attn, use_film=use_film, use_adagn=use_adagn)
            ]))
            self.downsample_blocks.append(
                FreqAwareDownsample(features[i], features[i+1], time_dim, emotion_dim)
            )
            # Source image residual: UNet level 0 is at H/2 (after LL downsample)
            # So level 0 needs DWT×1 (H→H/4? no, DWT on src_image H→H/2)
            # level i needs (i+1) DWTs: src H→H/2^(i+1) to match UNet at H/2^(i+1)
            self.res_blocks.append(
                WaveletResidualConnection(3, features[i], downsample_level=i+1)
            )
        
        # Bottleneck
        self.bottleneck = nn.ModuleList([
            FrequencyBottleneckBlock(features[-1], time_dim, emotion_dim,
                                     use_film=use_film, use_adagn=use_adagn),
            ResBlock(features[-1], features[-1], time_dim, emotion_dim,
                     use_film=use_film, use_adagn=use_adagn),
            FrequencyBottleneckBlock(features[-1], time_dim, emotion_dim,
                                     use_film=use_film, use_adagn=use_adagn)
        ])
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        
        for i in range(len(features) - 1, 0, -1):
            self.upsample_blocks.append(
                FreqAwareUpsample(features[i], features[i-1], time_dim, emotion_dim)
            )
            self.decoder_blocks.append(nn.ModuleList([
                ResBlock(features[i-1] * 2, features[i-1], time_dim, emotion_dim,
                         use_film=use_film, use_adagn=use_adagn),
                ResBlock(features[i-1], features[i-1], time_dim, emotion_dim,
                         use_film=use_film, use_adagn=use_adagn)
            ]))
        
        # Output heads (3 separate for interpretability)
        shared_out = nn.Sequential(
            nn.GroupNorm(8, features[0]),
            nn.SiLU()
        )
        self.shared_output = shared_out
        
        # LL noise prediction head
        self.output_noise = nn.Conv2d(features[0], 3, 3, padding=1)
        
        # Δmagnitude head (init to zero → identity at start)
        # UNet is at H/2 which matches HF resolution — no pooling needed
        self.output_delta_mag = nn.Conv2d(features[0], 18, 3, padding=1)
        nn.init.zeros_(self.output_delta_mag.weight)
        nn.init.zeros_(self.output_delta_mag.bias)
        
        # NOTE: Δphase head removed — phase handled by APT
    
    def forward(self, x, t, emotion_id, hf_condition_features, src_image=None):
        """
        Args:
            x: (B, 3+E, H/2, W/2) — noisy LL (downsampled) + emotion map
            t: (B,) — timesteps
            emotion_id: (B,) — emotion class indices
            hf_condition_features: list of features from HFConditionEncoder
            src_image: (B, 3, H, W) — source RGB for residual connections
        Returns:
            noise_pred:  (B, 3, H/2, W/2) — noise prediction
            delta_mag:   (B, 18, H/2, W/2) — HF magnitude delta
        """
        time_emb = self.time_embedding(t).to(dtype=x.dtype)
        emotion_emb = self.emotion_embedding(emotion_id)
        
        x = self.input_conv(x)
        skip_connections = []
        hi_freq_skips = []
        
        # Encoder
        for i, (encoder_block, downsample_block) in enumerate(
            zip(self.encoder_blocks, self.downsample_blocks)
        ):
            for block in encoder_block:
                x = block(x, time_emb, emotion_emb)
            
            # Inject source image features (identity preservation)
            if src_image is not None:
                res_feat = self.res_blocks[i](src_image)
                x = x + 0.05 * res_feat
            
            # Inject HF condition features
            x = x + self.hf_inject_scale * hf_condition_features[i]
            
            skip_connections.append(x)
            x, hi_freq = downsample_block(x, time_emb, emotion_emb)
            hi_freq_skips.append(hi_freq)
        
        # Bottleneck
        for block in self.bottleneck:
            if isinstance(block, FrequencyBottleneckBlock):
                x = block(x, time_emb, emotion_emb)
            else:
                x = block(x, time_emb, emotion_emb)
        
        # Decoder
        for i, (upsample_block, decoder_block) in enumerate(
            zip(self.upsample_blocks, self.decoder_blocks)
        ):
            hi_freq = hi_freq_skips[-(i+1)]
            x = upsample_block(x, hi_freq, time_emb, emotion_emb)
            skip = skip_connections[-(i+1)]
            x = torch.cat([x, skip], dim=1)
            for block in decoder_block:
                x = block(x, time_emb, emotion_emb)
        
        # Output heads
        x = self.shared_output(x)
        noise_pred = self.output_noise(x)
        delta_mag = self.output_delta_mag(x)
        
        return noise_pred, delta_mag


# ============================================================================
# Full Diffusion Model
# ============================================================================

class DirectionalWaveDiffusionModel(nn.Module):
    """
    Complete diffusion model with Analytic Phase Transport (APT).
    
    Pipeline: DTCWT decomposition → LL diffusion → APT displacement estimation
    → warp source HF + Δmag modulation → IDTCWT reconstruction.
    
    Phase is no longer regressed by the UNet. Instead, the displacement field
    u(x) is estimated from DTCWT phase differences between source and denoised
    LL, and source HF is warped geometrically.
    """
    def __init__(self,
                 num_emotions=8,
                 num_timesteps=1000,
                 beta_start=1e-4,
                 beta_end=2e-2,
                 features=None,
                 use_film=True,
                 use_adagn=False):
        super().__init__()
        if features is None:
            features = [48, 96, 192, 384]
        
        self.num_timesteps = num_timesteps
        self.num_emotions = num_emotions
        
        # DTCWT decomposition / reconstruction
        self.dtcwt = DTCWTWrapper()
        self.idtcwt = IDTCWTWrapper()
        
        # LL downsample/upsample (H×W ↔ H/2×W/2)
        self.ll_dwt = DWT()
        self.ll_iwt = IWT()
        
        # Analytic Phase Transport — replaces Δphase regression
        self.apt = AnalyticPhaseTransport()
        
        # Magnitude Modulator (emotion × direction, phase no longer modulated)
        emotion_dim = 64
        self.mag_mod = MagnitudeModulator(emotion_dim)
        
        # UNet: input = 3 LL + num_emotions one-hot
        self.unet = DirectionalWaveDiffusionUNet(
            in_channels=3 + num_emotions,
            features=features,
            emotion_dim=emotion_dim,
            num_emotions=num_emotions,
            use_film=use_film,
            use_adagn=use_adagn
        )
        
        # Noise schedule
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
    
    def ll_downsample(self, ll):
        """
        Downsample LL from H×W to H/2×W/2 using Haar DWT.
        
        Returns:
            ll_sub: (B, 3, H/2, W/2) — LL subband (coarse structure)
            ll_hf_skip: (B, 9, H/2, W/2) — LH+HL+HH subbands (detail skip)
        """
        ll_dwt = self.ll_dwt(ll)  # (B, 12, H/2, W/2) = 3ch × 4 subbands
        C = ll.shape[1]  # 3
        ll_sub = ll_dwt[:, :C, :, :]       # (B, 3, H/2, W/2) — LL only
        ll_hf_skip = ll_dwt[:, C:, :, :]   # (B, 9, H/2, W/2) — LH+HL+HH
        return ll_sub, ll_hf_skip
    
    def ll_upsample(self, ll_sub, ll_hf_skip):
        """
        Upsample LL from H/2×W/2 back to H×W using Haar IWT.
        
        Args:
            ll_sub: (B, 3, H/2, W/2) — predicted/denoised LL subband
            ll_hf_skip: (B, 9, H/2, W/2) — LH+HL+HH from source
        Returns:
            ll_full: (B, 3, H, W) — reconstructed LL at original resolution
        """
        ll_dwt = torch.cat([ll_sub, ll_hf_skip], dim=1)  # (B, 12, H/2, W/2)
        return self.ll_iwt(ll_dwt)  # (B, 3, H, W)
    
    def _inject_emotion(self, x_ll, emotion_id):
        """
        StarGAN-style injection: concatenate one-hot emotion map to LL.
        
        Args:
            x_ll: (B, 3, H/2, W/2) — noisy LL coefficients (downsampled)
            emotion_id: (B,)
        Returns:
            (B, 3 + num_emotions, H/2, W/2)
        """
        B, C, H, W = x_ll.shape
        emotion_onehot = F.one_hot(emotion_id, num_classes=self.num_emotions).to(dtype=x_ll.dtype)
        emotion_map = emotion_onehot[:, :, None, None].expand(B, self.num_emotions, H, W)
        return torch.cat([x_ll, emotion_map], dim=1)
    
    def forward_process(self, x0, t, noise=None):
        """Add noise to LL coefficients at timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise, noise
    
    def _predict_x0(self, x_noisy, noise_pred, t):
        """Predict clean x0 from noisy input and predicted noise."""
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        pred_x0 = (x_noisy - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha
        return torch.clamp(pred_x0, -3, 3)
    
    def forward(self, x, emotion_id, src_image=None):
        """
        Training forward: compute losses.
        
        Pipeline:
        1. DTCWT decompose target → LL, mag, phase
        2. UNet denoises LL → noise_pred, Δmag
        3. APT: estimate displacement u from phase diff(src_LL_Y, pred_LL_Y)
        4. Warp source HF + Haar detail by u, apply Δmag
        5. IDTCWT reconstruct
        
        Args:
            x: (B, 3, H, W) — target image
            emotion_id: (B,) — emotion class
            src_image: (B, 3, H, W) — source image for residual connection
        Returns:
            dict with losses and predicted image for DAN loss
        """
        # 1. DTCWT decompose target
        ll_full, mag, phase = self.dtcwt(x)  # ll_full:(B,3,H,W) YCbCr
        
        # Also decompose source for APT (if src_image provided, else self-supervised)
        src_for_apt = src_image if src_image is not None else x
        src_ll_full, src_mag, src_phase = self.dtcwt(src_for_apt)
        
        # 2. Downsample LL for UNet processing
        ll, ll_hf_skip = self.ll_downsample(ll_full)
        src_ll, src_ll_hf_skip = self.ll_downsample(src_ll_full)
        
        # 3. Add noise to downsampled LL
        B = x.shape[0]
        t = torch.randint(0, self.num_timesteps, (B,), device=x.device)
        noisy_ll, noise = self.forward_process(ll, t)
        
        # 4. Prepare HF conditioning (from source)
        hf_condition = torch.cat([src_mag, src_phase], dim=1)
        ll_size = (ll.shape[2], ll.shape[3])
        hf_features = self.unet.hf_encoder(hf_condition, ll_size)
        
        # 5. StarGAN emotion injection on LL
        unet_input = self._inject_emotion(noisy_ll, emotion_id)
        
        # 6. UNet forward → noise_pred + Δmag (no Δphase)
        noise_pred, delta_mag = self.unet(
            unet_input, t, emotion_id, hf_features, src_for_apt
        )
        
        # 7. MagnitudeModulator bias
        emotion_emb = self.unet.emotion_embedding(emotion_id)
        mod_dmag = self.mag_mod(emotion_emb)
        delta_mag = delta_mag + mod_dmag
        
        # Clamp Δmag
        delta_mag = 0.1 * torch.tanh(delta_mag)
        
        # Chrominance damping on Δmag
        chroma_scale = 0.1
        delta_mag[:, 6:, :, :] = delta_mag[:, 6:, :, :] * chroma_scale
        
        # 8. Predict clean LL
        pred_ll_half = self._predict_x0(noisy_ll, noise_pred, t)
        
        # 9. APT: estimate displacement from Y channel phase difference
        #    src LL Y at full res, pred LL Y upsampled to full res
        with torch.amp.autocast('cuda', enabled=False):
            src_ll_Y = src_ll_full[:, 0:1].float()  # Y channel at H×W
            pred_ll_up = self.ll_upsample(pred_ll_half, src_ll_hf_skip)
            pred_ll_Y = pred_ll_up[:, 0:1].float()
            displacement = self.apt.solve_displacement(src_ll_Y, pred_ll_Y)
        
        # 10. Warp source HF coefficients by displacement
        warped_mag, warped_phase = self.apt.warp_coefficients(
            src_mag, src_phase, displacement
        )
        
        # 11. Warp Haar detail bands by displacement
        warped_hf_skip = self.apt.warp_haar_detail(src_ll_hf_skip, displacement)
        
        # 12. Apply Δmag to warped magnitude
        pred_mag = warped_mag * (1 + delta_mag)
        pred_phase = warped_phase  # phase fully determined by APT geometry
        
        # 13. Compute losses
        ddpm_loss = F.l1_loss(noise_pred, noise)
        mag_loss = F.l1_loss(delta_mag, torch.zeros_like(delta_mag))
        chroma_loss = F.l1_loss(pred_ll_half[:, 1:3], ll[:, 1:3])
        
        # Displacement smoothness loss (TV regularization)
        u_smooth_loss = (torch.mean(torch.abs(displacement[:,:,:,1:] - displacement[:,:,:,:-1])) +
                         torch.mean(torch.abs(displacement[:,:,1:,:] - displacement[:,:,:-1,:])))
        
        # 14. Reconstruct image for DAN loss
        pred_ll_full = self.ll_upsample(pred_ll_half, warped_hf_skip)
        pred_real = pred_mag * torch.cos(pred_phase)
        pred_imag = pred_mag * torch.sin(pred_phase)
        pred_img = self.idtcwt(pred_ll_full, pred_real, pred_imag)
        
        return {
            'ddpm_loss': ddpm_loss,
            'mag_loss': mag_loss,
            'chroma_loss': chroma_loss,
            'u_smooth_loss': u_smooth_loss,
            'pred_img': pred_img,
            'pred_ll': pred_ll_full,
            'delta_mag': delta_mag,
            'displacement': displacement,
        }
    
    @torch.no_grad()
    def sample(self, src_image, target_emotion_id, num_steps=50, denoising_strength=0.3):
        """
        Inference: DDIM sampling + APT phase transport.
        
        Pipeline:
        1. DTCWT decompose source
        2. DDIM denoise LL → denoised LL'
        3. APT: phase diff(src_LL_Y, LL'_Y) → displacement u
        4. Warp source HF + Haar detail by u, apply Δmag
        5. IDTCWT reconstruct
        """
        device = src_image.device
        B = src_image.shape[0]
        
        # 1. DTCWT decompose source
        src_ll_full, src_mag, src_phase = self.dtcwt(src_image)
        
        # 2. Downsample LL
        src_ll, ll_hf_skip = self.ll_downsample(src_ll_full)
        
        # 3. Prepare HF conditioning
        hf_condition = torch.cat([src_mag, src_phase], dim=1)
        ll_size = (src_ll.shape[2], src_ll.shape[3])
        hf_features = self.unet.hf_encoder(hf_condition, ll_size)
        
        # 4. Add noise to source LL
        start_timestep = max(1, int(denoising_strength * self.num_timesteps))
        timesteps = torch.linspace(start_timestep - 1, 0,
                                   min(num_steps, start_timestep),
                                   dtype=torch.long, device=device)
        
        noise = torch.randn_like(src_ll)
        alpha_start = self.sqrt_alphas_cumprod[start_timestep]
        sigma_start = self.sqrt_one_minus_alphas_cumprod[start_timestep]
        x = alpha_start * src_ll + sigma_start * noise
        
        # 5. DDIM denoising loop
        final_dmag = None
        emotion_emb = self.unet.emotion_embedding(target_emotion_id)
        
        for i, t in enumerate(timesteps):
            t_tensor = torch.full((B,), t.item(), device=device, dtype=torch.long)
            
            unet_input = self._inject_emotion(x, target_emotion_id)
            noise_pred, delta_mag = self.unet(
                unet_input, t_tensor, target_emotion_id, hf_features, src_image
            )
            
            # MagnitudeModulator bias
            mod_dm = self.mag_mod(emotion_emb)
            delta_mag = delta_mag + mod_dm
            delta_mag = 0.1 * torch.tanh(delta_mag)
            delta_mag[:, 6:, :, :] = delta_mag[:, 6:, :, :] * 0.1
            
            final_dmag = delta_mag
            
            # DDIM step
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
        
        # 6. APT: estimate displacement from denoised LL vs source LL
        denoised_ll_full = self.ll_upsample(x, ll_hf_skip)
        src_ll_Y = src_ll_full[:, 0:1].float()
        pred_ll_Y = denoised_ll_full[:, 0:1].float()
        displacement = self.apt.solve_displacement(src_ll_Y, pred_ll_Y)
        
        # 7. Warp source HF by displacement
        warped_mag, warped_phase = self.apt.warp_coefficients(
            src_mag, src_phase, displacement
        )
        
        # 8. Warp Haar detail bands
        warped_hf_skip = self.apt.warp_haar_detail(ll_hf_skip, displacement)
        
        # 9. Apply Δmag + reconstruct
        new_mag = warped_mag * (1 + final_dmag)
        new_real = new_mag * torch.cos(warped_phase)
        new_imag = new_mag * torch.sin(warped_phase)
        
        x_full = self.ll_upsample(x, warped_hf_skip)
        output = self.idtcwt(x_full, new_real, new_imag)
        return output


if __name__ == '__main__':
    # Quick sanity check
    print("Testing DirectionalWaveDiffusionModel with APT...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = DirectionalWaveDiffusionModel(num_emotions=8).to(device)
    
    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    
    # Test forward
    x = torch.randn(2, 3, 224, 224).to(device)
    emotion = torch.randint(0, 8, (2,)).to(device)
    
    outputs = model(x, emotion, src_image=x)
    print(f"DDPM loss: {outputs['ddpm_loss'].item():.4f}")
    print(f"Mag loss: {outputs['mag_loss'].item():.4f}")
    print(f"Displacement shape: {outputs['displacement'].shape}")
    print(f"Displacement mean: {outputs['displacement'].abs().mean().item():.4f}")
    print(f"U smooth loss: {outputs['u_smooth_loss'].item():.4f}")
    print(f"Pred image shape: {outputs['pred_img'].shape}")
    
    # Test sampling
    out = model.sample(x[:1], emotion[:1], num_steps=5)
    print(f"Sample output shape: {out.shape}")
    assert not torch.isnan(out).any(), "NaN in sample output!"
    
    # DTCWT roundtrip test
    wrapper = DTCWTWrapper().to(device)
    iwrapper = IDTCWTWrapper().to(device)
    ll, mag, phase = wrapper(x)
    real = mag * torch.cos(phase)
    imag = mag * torch.sin(phase)
    recon = iwrapper(ll, real, imag)
    mse = ((x - recon) ** 2).mean().item()
    print(f"DTCWT roundtrip MSE: {mse:.8f}")
    
    # APT zero-displacement test
    apt = AnalyticPhaseTransport().to(device)
    with torch.no_grad():
        ll_y = ll[:1, 0:1]  # Y channel
        u = apt.solve_displacement(ll_y, ll_y)
        print(f"APT zero-disp test: max |u| = {u.abs().max().item():.6f} (expect ~0)")
    
    print("All tests passed!" if mse < 1e-4 else f"WARNING: roundtrip MSE too high: {mse}")
