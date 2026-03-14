import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pytorch_wavelets.dtcwt.transform2d import DTCWTForward, DTCWTInverse
from pytorch_wavelets.dwt.transform2d import DWTForward, DWTInverse


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
        ll:    (B, 3, H, W)      — YCbCr lowpass at J=1 resolution (identical to input spatial shape)
        mag:   (B, 18, H/2, W/2) — magnitude of J=1 (Y:0-5, Cb:6-11, Cr:12-17)
        phase: (B, 18, H/2, W/2) — phase (Y:0-5, Cb:6-11, Cr:12-17)
    """
    def __init__(self, biort='near_sym_b', qshift='qshift_b'):
        super().__init__()
        # Use J=1 to get LL at 224x224 natively, UNet will process this 224x224 resolution
        self.dtcwt = DTCWTForward(J=1, biort=biort, qshift=qshift)
    
    def forward(self, x):
        # DTCWT requires fp32 — disable autocast for compatibility
        with torch.amp.autocast('cuda', enabled=False):
            x = x.float()
            
            # Convert RGB → YCbCr before DTCWT
            x_ycbcr = rgb_to_ycbcr(x)
            
            yl, yh = self.dtcwt(x_ycbcr)
            # yl: (B, 3, H, W) — YCbCr lowpass at original resolution
            
            # Pass yl (224x224) directly without pooling for perfect J=1 reconstruction
            
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
            ll:      (B, 3, H, W)         — YCbCr LL at original resolution
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
            # Since ll is already 224x224 and HF is 112x112, we can IDTCWT directly
            ycbcr = self.idtcwt((ll, [hf]))
            
            # Convert YCbCr → RGB
            return ycbcr_to_rgb(ycbcr)


# ============================================================================
# Core Building Blocks (reused from model.py with adaptations)
# ============================================================================


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
        
        attn = torch.softmax(
            (torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)).float(),
            dim=-1
        ).to(q.dtype)
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

# class WaveletResidualConnection(nn.Module):
#     """Source image features injected at each encoder level."""
#     def __init__(self, in_channels, out_channels, downsample_level=1):
#         super().__init__()
#         self.dwt = DWT()
#         self.downsample_level = downsample_level
#         final_channels = in_channels * (4 ** downsample_level)
#         self.conv = nn.Conv2d(final_channels, out_channels, 1)

#     def forward(self, x):
#         for _ in range(self.downsample_level):
#             x = self.dwt(x)
#         return self.conv(x)
class SpatialResidualConnection(nn.Module):
    """
    Source image features injected at each encoder level.
    (Đã gỡ bỏ hoàn toàn Haar DWT, thay bằng Average Pooling & Conv2d)
    """
    def __init__(self, in_channels, out_channels, downsample_level=1):
        super().__init__()
        self.downsample_level = downsample_level
        
        # Dùng AvgPool2d để giảm kích thước (H, W) đi một nửa ở mỗi level 
        # (Giống hệt tính chất decimation của DWT nhưng không bị phình số kênh)
        layers = []
        for _ in range(downsample_level):
            layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            
        self.downsample = nn.Sequential(*layers)
        
        # Dùng Conv2d (kernel 3x3) để trích xuất đặc trưng và ép về đúng out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        # x: [B, 3, H, W] (Ảnh LL_source)
        x = self.downsample(x) # Kích thước giảm 2^level lần
        x = self.conv(x)       # Số kênh chuyển từ 3 -> out_channels
        x = self.norm(x)
        return self.act(x)

class SpatialDownsample(nn.Module):
    """
    Giảm kích thước không gian (H, W) đi một nửa bằng Conv2d stride 2.
    Thay thế hoàn toàn cho FreqAwareDownsample (DWT).
    """
    def __init__(self, in_channels, out_channels, time_dim, emotion_dim):
        super().__init__()
        # kernel_size=3, stride=2, padding=1 giúp giảm (H, W) xuống đúng 1 nửa
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.time_mlp = nn.Linear(time_dim, out_channels)
        self.emotion_mlp = nn.Linear(emotion_dim, out_channels)

    def forward(self, x, time_emb, emotion_emb):
        out = self.norm(self.conv(x))
        # Nhúng thêm thông tin Thời gian (t) và Cảm xúc (emotion)
        out = out + self.time_mlp(F.silu(time_emb))[:, :, None, None]
        out = out + self.emotion_mlp(F.silu(emotion_emb))[:, :, None, None]
        return F.silu(out)

class SpatialUpsample(nn.Module):
    """
    Tăng kích thước không gian (H, W) lên gấp đôi bằng Interpolate + Conv2d.
    Thay thế hoàn toàn cho FreqAwareUpsample (IWT & DGWM).
    """
    def __init__(self, in_channels, out_channels, time_dim, emotion_dim):
        super().__init__()
        # Conv2d bình thường với stride=1 để làm mượt sau khi phóng to
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.time_mlp = nn.Linear(time_dim, out_channels)
        self.emotion_mlp = nn.Linear(emotion_dim, out_channels)

    def forward(self, x, time_emb, emotion_emb):
        # 1. Phóng to ma trận lên gấp đôi bằng phép nội suy (Chống lỗi bàn cờ)
        x_up = F.interpolate(x, scale_factor=2.0, mode='nearest')
        
        # 2. Đưa qua Tích chập để làm mượt và trích xuất đặc trưng
        out = self.norm(self.conv(x_up))
        
        # 3. Nhúng Thời gian và Cảm xúc
        out = out + self.time_mlp(F.silu(time_emb))[:, :, None, None]
        out = out + self.emotion_mlp(F.silu(emotion_emb))[:, :, None, None]
        return F.silu(out)
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
        
        # Initial projection: HF is at H/2 (112x112), UNet level 0 is at H (224x224)
        # So we upsample HF spatially by 2 first
        self.init_proj = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode='nearest'),
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
            ll_size: (H, W) — spatial size of LL (= 224x224, UNet level 0)
        Returns:
            list of features: level 0 at H, level 1 at H/2, level 2 at H/4
        """
        # Project and upsample to H (224x224) to match UNet input resolution
        x = self.init_proj(hf_mag_phase)  # (B, feat[0], H, W)
        
        features = []
        for i in range(self.num_levels):
            if i == 0:
                # Level 0: UNet is at H (224) — direct projection, no resize needed
                features.append(self.level_projs[i](x))
            else:
                # Level i: UNet is at H/2^i → pool accordingly
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
        # Force fp32 — determinant division is unstable in fp16
        with torch.amp.autocast('cuda', enabled=False):
            ll_y_src = ll_y_src.float()
            ll_y_tgt = ll_y_tgt.float()
            
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
                # Use larger eps (1e-6) for numerical stability
                det = a11 * a22 - a12 * a12 + 1e-6
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
            
            # Clamp displacement to prevent extreme warps
            u = u.clamp(-50, 50)
        
        return u
    
    def _make_warp_grid(self, u):
        """
        Convert pixel displacement u to normalized grid for F.grid_sample.
        
        Args:
            u: (B, 2, H, W) — displacement in pixels (dy, dx)
        Returns:
            grid: (B, H, W, 2) — normalized grid for grid_sample (x, y order)
        """
        with torch.amp.autocast('cuda', enabled=False):
            u = u.float()
            B, _, H, W = u.shape
            # Base grid: identity mapping
            yy = torch.arange(H, device=u.device, dtype=torch.float32)
            xx = torch.arange(W, device=u.device, dtype=torch.float32)
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
        # Force fp32 — grid_sample + atan2 are unstable in fp16
        with torch.amp.autocast('cuda', enabled=False):
            src_mag = src_mag.float()
            src_phase = src_phase.float()
            u = u.float()
            
            _, _, Hm, Wm = src_mag.shape
            
            # Resize u to match HF resolution if needed
            if u.shape[2] != Hm or u.shape[3] != Wm:
                scale_h = Hm / u.shape[2]
                scale_w = Wm / u.shape[3]
                u = F.interpolate(u, size=(Hm, Wm), mode='bilinear',
                                  align_corners=False)
                # Scale displacement proportionally
                u[:, 0] = u[:, 0] * scale_h
                u[:, 1] = u[:, 1] * scale_w
            
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
        with torch.amp.autocast('cuda', enabled=False):
            ll_hf_skip = ll_hf_skip.float()
            u = u.float()
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
                 in_channels=11,    # 3 LL (Y, Cb, Cr) + 8 emotion one-hot
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
                SpatialDownsample(features[i], features[i+1], time_dim, emotion_dim)
            )
            # Source image residual: UNet level 0 is at H (224x224)
            # So level 0 needs no downsampling (H→H)
            # level i needs i downsamples: src H→H/2^i to match UNet at H/2^i
            self.res_blocks.append(
                SpatialResidualConnection(3, features[i], downsample_level=i)
            )
        
        # Bottleneck
        self.bottleneck = nn.ModuleList([
            ResBlock(features[-1], features[-1], time_dim, emotion_dim,
                     use_film=use_film, use_adagn=use_adagn),
            ResBlock(features[-1], features[-1], time_dim, emotion_dim,
                     use_film=use_film, use_adagn=use_adagn),
            ResBlock(features[-1], features[-1], time_dim, emotion_dim,
                     use_film=use_film, use_adagn=use_adagn)
        ])
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        
        for i in range(len(features) - 1, 0, -1):
            self.upsample_blocks.append(
                SpatialUpsample(features[i], features[i-1], time_dim, emotion_dim)
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
        
        # Noise/velocity prediction head (3ch: Y, Cb, Cr)
        self.output_noise = nn.Conv2d(features[0], 3, 3, padding=1)
        
        # Δmagnitude head (init to zero → identity at start)
        # UNet is at 224x224, but HF is at 112x112
        # So output_delta_mag needs to pool down, or we apply a Convolution with stride=2
        self.output_delta_mag = nn.Conv2d(features[0], 18, kernel_size=4, stride=2, padding=1)
        nn.init.zeros_(self.output_delta_mag.weight)
        nn.init.zeros_(self.output_delta_mag.bias)
        
        # NOTE: Δphase head removed — phase handled by APT
    
    def forward(self, x, t, emotion_id, hf_condition_features, src_image=None):
        """
        Args:
            x: (B, 3+E, H, W) — noisy LL (224x224) + emotion map
            t: (B,) — timesteps
            emotion_id: (B,) — emotion class indices
            hf_condition_features: list of features from HFConditionEncoder
            src_image: (B, 3, H, W) — source RGB for residual connections
        Returns:
            noise_pred:  (B, 3, H, W) — noise prediction
            delta_mag:   (B, 18, H/2, W/2) — HF magnitude delta
        """
        time_emb = self.time_embedding(t).to(dtype=x.dtype)
        emotion_emb = self.emotion_embedding(emotion_id)
        
        x = self.input_conv(x)
        skip_connections = []
        
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
            x = downsample_block(x, time_emb, emotion_emb)
        
        # Bottleneck
        for block in self.bottleneck:
            x = block(x, time_emb, emotion_emb)
        
        # Decoder
        for i, (upsample_block, decoder_block) in enumerate(
            zip(self.upsample_blocks, self.decoder_blocks)
        ):
            x = upsample_block(x, time_emb, emotion_emb)
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
    V3: Full 12-channel Wavelet Diffusion with Rectified Flow.
    
    Pipeline:
    1. DTCWT decompose → LL (112x112)
    2. Haar DWT → 12ch (56x56): LL(3) + LH(3) + HL(3) + HH(3)
    3. Per-channel normalize (balance LL vs HF variance)
    4. Rectified Flow: x_t = (1-t)*x_0 + t*noise
    5. UNet predicts velocity v = noise - x_0 on all 12ch
    6. Un-normalize → Haar IWT → LL' (112x112)
    7. APT warp DTCWT coefficients + delta_mag → IDTCWT reconstruct
    """
    def __init__(self,
                 num_emotions=8,
                 features=None,
                 use_film=True,
                 use_adagn=False,
                 wavelet_stats_path='wavelet_stats.pt'):
        super().__init__()
        if features is None:
            features = [48, 96, 192, 384]
        
        self.num_emotions = num_emotions
        
        # DTCWT decomposition / reconstruction
        self.dtcwt = DTCWTWrapper()
        self.idtcwt = IDTCWTWrapper()
        
        # Tự học tỷ lệ cho 3 kênh LL (Y, Cb, Cr)
        self.v_scale = nn.Parameter(torch.ones(1, 3, 1, 1))

        # Tự học tỷ lệ cho 18 kênh HF Magnitude
        self.dmag_scale = nn.Parameter(torch.ones(1, 18, 1, 1))
        with torch.no_grad():
            self.dmag_scale[:, 6:] = 0.3  # Khởi tạo Cb/Cr của Mag an toàn ở 0.3
        
        # Analytic Phase Transport (for DTCWT HF warp only)
        self.apt = AnalyticPhaseTransport()
        
        # Magnitude Modulator
        emotion_dim = 64
        self.mag_mod = MagnitudeModulator(emotion_dim)
        
        # Change-mask gating: suppress old HF in regions where LL changed
        # mask_temperature: higher = sharper binary mask (init=5.0 for moderate sharpness)
        # mask_threshold: change level at which suppression kicks in (init=0.3)
        self.mask_temperature = nn.Parameter(torch.tensor(5.0))
        self.mask_threshold = nn.Parameter(torch.tensor(0.3))
        
        # UNet: input = 3 channels (Y,Cb,Cr) + num_emotions one-hot
        self.unet = DirectionalWaveDiffusionUNet(
            in_channels=3 + num_emotions,
            features=features,
            emotion_dim=emotion_dim,
            num_emotions=num_emotions,
            use_film=use_film,
            use_adagn=use_adagn
        )
        
        # Per-channel normalization buffers (Computed at pixel-level on 5000 images)
        # LL range: mean=[0.01, 1.33], std=[0.46, 2.30]
        # HF range: mean=[-0.0004, 0.0001], std=[0.004, 0.142]
        ch_mean = torch.tensor([0.011998, 0.896247, 1.325454]).view(1, 3, 1, 1)
        
        ch_std = torch.tensor([2.295658, 0.460137, 0.533324]).view(1, 3, 1, 1)

        self.register_buffer('ch_mean', ch_mean)
        self.register_buffer('ch_std', ch_std)
    
    def normalize_wavelet(self, x):
        """Per-channel normalize 12ch wavelet to ~N(0,1)."""
        return (x - self.ch_mean) / (self.ch_std + 1e-6)
    
    def unnormalize_wavelet(self, x):
        """Undo per-channel normalization."""

        return x * (self.ch_std + 1e-6) + self.ch_mean
    
    def _compute_change_mask(self, src_ll, pred_ll, hf_size):
        """
        Compute retain mask from LL difference: regions that changed → suppress old HF.
        
        Args:
            src_ll:  (B, 3, H, W) — source LL (YCbCr)
            pred_ll: (B, 3, H, W) — predicted/denoised LL (YCbCr)
            hf_size: (Hh, Wh) — spatial size of HF coefficients
        Returns:
            retain_mask: (B, 1, Hh, Wh) — 1=keep old HF, 0=suppress old HF
        """
        # Change magnitude across all YCbCr channels
        ll_diff = (pred_ll - src_ll).abs().mean(dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Per-image normalization to [0, 1]
        diff_max = ll_diff.amax(dim=[2, 3], keepdim=True) + 1e-6
        change_norm = ll_diff / diff_max  # (B, 1, H, W), 0=no change, 1=max change
        
        # Smooth sigmoid thresholding: regions above threshold → suppress
        # retain = sigmoid(temperature * (threshold - change))
        # When change > threshold: retain → 0 (suppress old HF)
        # When change < threshold: retain → 1 (keep old HF)
        temperature = self.mask_temperature.abs() + 1.0  # ensure positive, min=1
        threshold = torch.sigmoid(self.mask_threshold)     # keep in (0, 1)
        retain_mask = torch.sigmoid(temperature * (threshold - change_norm))
        
        # Resize to HF resolution
        retain_mask = F.interpolate(retain_mask, size=hf_size, mode='bilinear',
                                    align_corners=False)
        
        return retain_mask
    
    def _inject_emotion(self, x_wavelet, emotion_id):
        """
        StarGAN-style injection: concatenate one-hot emotion map to 12ch wavelet.
        
        Args:
            x_wavelet: (B, 12, H/2, W/2) — normalized noisy wavelet
            emotion_id: (B,)
        Returns:
            (B, 12 + num_emotions, H/2, W/2)
        """
        B, C, H, W = x_wavelet.shape
        emotion_onehot = F.one_hot(emotion_id, num_classes=self.num_emotions).to(dtype=x_wavelet.dtype)
        emotion_map = emotion_onehot[:, :, None, None].expand(B, self.num_emotions, H, W)
        return torch.cat([x_wavelet, emotion_map], dim=1)
    
    def rf_forward(self, x0, t, noise=None):
        """
        Rectified Flow forward: x_t = (1-t)*x0 + t*noise
        
        Args:
            x0: (B, 12, H, W) — clean normalized wavelet
            t: (B,) — timestep in [0, 1]
            noise: optional pre-generated noise
        Returns:
            x_t: (B, 12, H, W) — interpolated sample
            velocity: (B, 12, H, W) — target velocity v = noise - x0
        """
        if noise is None:
            noise = torch.randn_like(x0)
        t_expand = t.view(-1, 1, 1, 1)
        x_t = (1 - t_expand) * x0 + t_expand * noise
        velocity = noise - x0
        return x_t, velocity
    
    def forward(self, x, emotion_id, src_image=None, drop_prob=0.0):
        """
        V3 Training forward with Rectified Flow on 12ch normalized wavelet.
        
        Pipeline:
        1. DTCWT decompose → LL (112x112), mag, phase
        2. Haar DWT → 12ch (56x56)
        3. Per-channel normalize → RF interpolation → UNet → velocity pred
        4. Predict x0 → un-normalize → Haar IWT → LL' (112x112)
        5. APT warp DTCWT coefficients + delta_mag
        6. IDTCWT reconstruct → pred_img for DAN/LPIPS
        """
        # 1. DTCWT decompose target
        ll_full, mag, phase = self.dtcwt(x)
        
        # Also decompose source for APT
        src_for_apt = src_image if src_image is not None else x
        src_ll_full, src_mag, src_phase = self.dtcwt(src_for_apt)
        
        # 2. Per-channel normalize directly on 3ch LL
        ll_norm = self.normalize_wavelet(ll_full)
        
        # 3. Rectified Flow: sample t, interpolate, get velocity target
        B = x.shape[0]
        t = torch.rand(B, device=x.device)
        x_t, velocity = self.rf_forward(ll_norm, t)
        
        # 4. Prepare HF conditioning (from source DTCWT)
        hf_condition = torch.cat([src_mag, src_phase], dim=1)
        ll_size = (ll_full.shape[2], ll_full.shape[3])
        hf_features = self.unet.hf_encoder(hf_condition, ll_size)
        
        # 5. Emotion injection on noisy 3ch
        unet_input = self._inject_emotion(x_t, emotion_id)
        
        # 6. UNet forward → velocity_pred (3ch) + Δmag (18ch)
        v_pred, delta_mag = self.unet(
            unet_input, t, emotion_id, hf_features, src_for_apt
        )
        
        # 7. MagnitudeModulator bias
        emotion_emb = self.unet.emotion_embedding(emotion_id)
        mod_dmag = self.mag_mod(emotion_emb)
        delta_mag = delta_mag + mod_dmag
        
        # Soft-bound v_pred và delta_mag
        v_pred = 5.0 * torch.tanh(v_pred / 5.0) * self.v_scale
        delta_mag = torch.tanh(delta_mag) * self.dmag_scale
        
        # 8. Predict clean x0 from RF: x0 = x_t - t * v_pred
        t_expand = t.view(-1, 1, 1, 1)
        pred_ll_norm = x_t - t_expand * v_pred
        pred_ll_norm = torch.clamp(pred_ll_norm, -5, 5)
        
        # 9. Un-normalize
        pred_ll_full = self.unnormalize_wavelet(pred_ll_norm)
        
        # 11. APT: estimate displacement from Y channel
        with torch.amp.autocast('cuda', enabled=False):
            src_ll_Y = src_ll_full[:, 0:1].float()
            pred_ll_Y = pred_ll_full[:, 0:1].float()
            displacement = self.apt.solve_displacement(src_ll_Y, pred_ll_Y)
        
        # 12. Warp DTCWT coefficients by displacement
        warped_mag, warped_phase = self.apt.warp_coefficients(
            src_mag, src_phase, displacement
        )
        
        # 13. Compute change mask from LL difference
        retain_mask = self._compute_change_mask(
            src_ll_full, pred_ll_full, warped_mag.shape[2:]
        )
        
        # 14. Apply Δmag with change-gated suppression
        # retain_mask ≈ 1: keep old HF (identity regions)
        # retain_mask ≈ 0: suppress old HF (expression-change regions)
        pred_mag = retain_mask * warped_mag * (1 + delta_mag)
        pred_phase = warped_phase
        
        # 14. Compute losses
        velocity_damped = velocity.clone() * self.v_scale.detach()
        rf_loss = F.l1_loss(v_pred, velocity_damped)
        
        # For logging compatibility
        rf_loss_ll = rf_loss
        rf_loss_hf = torch.tensor(0.0, device=rf_loss.device)
        
        # Magnitude loss
        mag_loss = F.l1_loss(pred_mag, mag)
        
        # Chroma loss 
        chroma_idx = [1, 2]
        pred_chroma = pred_ll_norm[:, chroma_idx]
        target_chroma = (x_t - t_expand * velocity)[:, chroma_idx]  # target x0 chroma
        chroma_loss = F.l1_loss(pred_chroma, target_chroma)
        
        # Displacement smoothness
        u_smooth_loss = (torch.mean(torch.abs(displacement[:,:,:,1:] - displacement[:,:,:,:-1])) +
                         torch.mean(torch.abs(displacement[:,:,1:,:] - displacement[:,:,:-1,:])))
        
        # 16. Reconstruct image for DAN/LPIPS loss
        pred_real = pred_mag * torch.cos(pred_phase)
        pred_imag = pred_mag * torch.sin(pred_phase)
        pred_img = self.idtcwt(pred_ll_full, pred_real, pred_imag)
        pred_img = torch.nan_to_num(pred_img, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return {
            'ddpm_loss': rf_loss,        # key kept as 'ddpm_loss' for training script compat
            'rf_loss_ll': rf_loss_ll,
            'rf_loss_hf': rf_loss_hf,
            'mag_loss': mag_loss,
            'chroma_loss': chroma_loss,
            'u_smooth_loss': u_smooth_loss,
            'pred_img': pred_img,
            'pred_ll': pred_ll_full,
            'delta_mag': delta_mag,
            'v_pred': v_pred,
            'displacement': displacement,
            'retain_mask': retain_mask,
        }
    
    @torch.no_grad()
    def sample(self, src_image, target_emotion_id, num_steps=20, denoising_strength=1.0):
        """
        V3 Inference: Euler ODE sampling with Rectified Flow on 12ch wavelet.
        
        Pipeline:
        1. DTCWT decompose source → LL, mag, phase
        2. Haar DWT → 12ch → normalize
        3. Mix with noise: x_start = (1-s)*x0_norm + s*noise
        4. Euler ODE: step from t=s to t=0
        5. Un-normalize → Haar IWT → LL' (112x112)
        6. APT warp + delta_mag → IDTCWT reconstruct
        """
        device = src_image.device
        B = src_image.shape[0]
        
        # 1. DTCWT decompose source
        src_ll_full, src_mag, src_phase = self.dtcwt(src_image)
        
        # 2. Normalize 3ch LL
        src_norm = self.normalize_wavelet(src_ll_full)
        
        # 3. Prepare HF conditioning
        hf_condition = torch.cat([src_mag, src_phase], dim=1)
        ll_size = (src_ll_full.shape[2], src_ll_full.shape[3])
        hf_features = self.unet.hf_encoder(hf_condition, ll_size)
        emotion_emb = self.unet.emotion_embedding(target_emotion_id)
        
        # 4. Start from noisy version of source
        noise = torch.randn_like(src_norm)
        start_t = denoising_strength  # how much noise to add (0=no change, 1=pure noise)
        x = (1 - start_t) * src_norm + start_t * noise
        
        # 5. Euler ODE: step from t=start_t to t=0
        timesteps = torch.linspace(start_t, 0, num_steps + 1, device=device)
        final_dmag = None
        
        for i in range(num_steps):
            t_now = timesteps[i]
            t_next = timesteps[i + 1]
            t_tensor = torch.full((B,), t_now.item(), device=device)
            
            unet_input = self._inject_emotion(x, target_emotion_id)
            v_pred, delta_mag = self.unet(
                unet_input, t_tensor, target_emotion_id, hf_features, src_image
            )
            
            # MagnitudeModulator bias
            mod_dm = self.mag_mod(emotion_emb)
            delta_mag = delta_mag + mod_dm
            
            # Soft-bound v_pred và delta_mag
            v_pred = 5.0 * torch.tanh(v_pred / 5.0) * self.v_scale
            delta_mag = torch.tanh(delta_mag) * self.dmag_scale
            final_dmag = delta_mag
            
            # Predict x0 (clean target) for stability and clamping
            # x_t = (1-t)*x0 + t*noise  => x0 = (x_t - t*noise)/(1-t) 
            # In Rectified Flow: v = noise - x0 => x0 = x_t - t*v
            x0_pred = x - t_now * v_pred
            x0_pred = torch.clamp(x0_pred, -5.0, 5.0)  # Core stability fix
            
            # Re-derive velocity from clamped x0
            v_actual = (x - x0_pred) / t_now if t_now > 0 else v_pred
            
            # Euler step: x_{t-dt} = x_t + (t_next - t_now) * v_pred
            dt = t_next - t_now  # negative
            x = x + dt * v_actual
            
            # Final safety clamp
            x = torch.clamp(x, -5.0, 5.0)
        
        # 6. Clamp and un-normalize
        x = torch.clamp(x, -5, 5)
        pred_ll_full = self.unnormalize_wavelet(x)
        
        # 7. APT: estimate displacement
        src_ll_Y = src_ll_full[:, 0:1].float()
        pred_ll_Y = pred_ll_full[:, 0:1].float()
        displacement = self.apt.solve_displacement(src_ll_Y, pred_ll_Y)
        
        # 8. Warp DTCWT coefficients
        warped_mag, warped_phase = self.apt.warp_coefficients(
            src_mag, src_phase, displacement
        )
        
        # 9. Compute change mask and apply gated Δmag + reconstruct
        retain_mask = self._compute_change_mask(
            src_ll_full, pred_ll_full, warped_mag.shape[2:]
        )
        new_mag = retain_mask * warped_mag * (1 + final_dmag)
        new_real = new_mag * torch.cos(warped_phase)
        new_imag = new_mag * torch.sin(warped_phase)
        
        output = self.idtcwt(pred_ll_full, new_real, new_imag)
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
    print(f"Max abs v_pred: {outputs['v_pred'].abs().max().item():.4f}")
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
