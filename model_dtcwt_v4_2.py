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
# Module 1: DTCWT_Encoder
# ============================================================================
class DTCWT_Encoder(nn.Module):
    """
    Nhiệm vụ: Tiền xử lý dữ liệu
    Input: x_rgb - kích thước [B, 3, 224, 224]
    Output: ll_src, hf_real_1, hf_imag_1, hf_real_2, hf_imag_2
    """
    def __init__(self, biort='near_sym_b', qshift='qshift_b'):
        super().__init__()
        self.dtcwt = DTCWTForward(J=2, biort=biort, qshift=qshift)
    
    def forward(self, x_rgb):
        # Chuyển RGB sang YCbCr
        x_ycbcr = rgb_to_ycbcr(x_rgb)  # [B, 3, 224, 224]
        
        # Chạy DTCWT với J=2
        with torch.amp.autocast('cuda', enabled=False):
            x_ycbcr = x_ycbcr.float()
            ll_src, yh = self.dtcwt(x_ycbcr)
            
            # ll_src: [B, 3, 56, 56] (LL at J=2)
            # yh[0]: [B, 3, 6, 112, 112, 2] (HF level 1)
            # yh[1]: [B, 3, 6, 56, 56, 2] (HF level 2)
            
            hf_level1 = yh[0]  # [B, 3, 6, 112, 112, 2]
            hf_level2 = yh[1]  # [B, 3, 6, 56, 56, 2]
            
            # Tách real và imag cho level 1
            hf_real_1 = hf_level1[..., 0]  # [B, 3, 6, 112, 112]
            hf_imag_1 = hf_level1[..., 1]  # [B, 3, 6, 112, 112]
            
            # Tách real và imag cho level 2
            hf_real_2 = hf_level2[..., 0]  # [B, 3, 6, 56, 56]
            hf_imag_2 = hf_level2[..., 1]  # [B, 3, 6, 56, 56]
            
            # Chuyển về magnitude & phase
            mag1 = torch.sqrt(hf_real_1**2 + hf_imag_1**2 + 1e-8)  # [B, 3, 6, 112, 112]
            phase1 = torch.atan2(hf_imag_1, hf_real_1)  # [B, 3, 6, 112, 112]
            
            mag2 = torch.sqrt(hf_real_2**2 + hf_imag_2**2 + 1e-8)  # [B, 3, 6, 56, 56]
            phase2 = torch.atan2(hf_imag_2, hf_real_2)  # [B, 3, 6, 56, 56]
            
            # Reshape để gộp channels: [B, 18, H, W]
            mag1 = mag1.reshape(mag1.shape[0], mag1.shape[1] * mag1.shape[2], mag1.shape[3], mag1.shape[4])
            phase1 = phase1.reshape(phase1.shape[0], phase1.shape[1] * phase1.shape[2], phase1.shape[3], phase1.shape[4])
            
            mag2 = mag2.reshape(mag2.shape[0], mag2.shape[1] * mag2.shape[2], mag2.shape[3], mag2.shape[4])
            phase2 = phase2.reshape(phase2.shape[0], phase2.shape[1] * phase2.shape[2], phase2.shape[3], phase2.shape[4])
        
        return ll_src, mag1, phase1, mag2, phase2


# ============================================================================
# Module 2: LL_Diffusion_UNet
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


class LL_Diffusion_UNet(nn.Module):
    """
    Nhiệm vụ: Học cách thay đổi biểu cảm (chỉ trên tần số thấp)
    Kiến trúc: UNet chuẩn với FiLM/AdaGN/CrossAttention conditioning
    Input: ll_concat [B, 3+num_emotions, H, W], timestep [B], emotion_emb [B, D]
    Output: ll_tgt_pred [B, 3, H, W]
    """
    def __init__(self, in_channels=10, out_channels=3, features=None, time_dim=256, 
                 emotion_dim=64, use_film=True, use_adagn=False):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]  # Features cho input 112x112
        
        self.time_embed = TimeEmbedding(time_dim)
        
        # Encoder
        self.input_conv = nn.Conv2d(in_channels, features[0], 3, padding=1)
        self.down_blocks = nn.ModuleList()
        
        in_feat = features[0]
        for i, feat in enumerate(features[1:]):
            # CrossAttention ở các level sâu hơn (nửa sau)
            use_attn = i >= len(features) // 2 - 1
            self.down_blocks.append(
                nn.ModuleList([
                    nn.Conv2d(in_feat, feat, 3, stride=2, padding=1),  # Downsample
                    ResBlock(feat, feat, time_dim, emotion_dim,
                             use_cross_attn=use_attn, use_film=use_film, use_adagn=use_adagn)
                ])
            )
            in_feat = feat
        
        # Middle block — always use cross attention
        self.middle_block = ResBlock(features[-1], features[-1], time_dim, emotion_dim,
                                     use_cross_attn=True, use_film=use_film, use_adagn=use_adagn)
        
        # Decoder
        self.up_blocks = nn.ModuleList()
        reversed_features = list(reversed(features[:-1]))
        for i, feat in enumerate(reversed_features):
            # CrossAttention ở các level đầu decoder (gần bottleneck)
            use_attn = i < len(features) // 2
            skip_idx = -(i+2)
            skip_feat = features[skip_idx]
            total_in = feat + skip_feat
            self.up_blocks.append(
                nn.ModuleList([
                    nn.ConvTranspose2d(in_feat, feat, 4, stride=2, padding=1),  # Upsample
                    ResBlock(total_in, feat, time_dim, emotion_dim,
                             use_cross_attn=use_attn, use_film=use_film, use_adagn=use_adagn)
                ])
            )
            in_feat = feat
        
        # Output
        self.output_conv = nn.Conv2d(features[0], out_channels, 3, padding=1)
    
    def forward(self, ll_concat, timestep, emotion_emb):
        time_emb = self.time_embed(timestep)
        
        # Encoder
        x = self.input_conv(ll_concat)
        # print(f"Input conv: {x.shape}")
        skip_connections = [x]
        
        for down_conv, res_block in self.down_blocks:
            x = down_conv(x)
            # print(f"After down_conv: {x.shape}")
            x = res_block(x, time_emb, emotion_emb)
            # print(f"After res_block: {x.shape}")
            skip_connections.append(x)
        
        # Middle
        x = self.middle_block(x, time_emb, emotion_emb)
        # print(f"After middle: {x.shape}")
        
        # Decoder
        skip_connections = skip_connections[::-1]
        for i, (up_conv, res_block) in enumerate(self.up_blocks):
            x = up_conv(x)
            # print(f"After up_conv {i}: {x.shape}")
            skip = skip_connections[i+1]
            # print(f"Skip {i}: {skip.shape}")
            
            # Resize skip to match up_conv spatial dimensions
            if skip.shape[2:] != x.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
                # print(f"Skip {i} resized: {skip.shape}")
            
            x = torch.cat([x, skip], dim=1)
            # print(f"After cat {i}: {x.shape}")
            x = res_block(x, time_emb, emotion_emb)
        
        # Output
        ll_tgt_pred = self.output_conv(x)
        return ll_tgt_pred


# ============================================================================
# Module 3: WPTL_Layer (Wavelet Phase Transport Layer)
# ============================================================================
def _wrap_phase(x):
    """Wrap angle to [-π, π]."""
    return torch.atan2(torch.sin(x), torch.cos(x))


class WPTL_Layer(nn.Module):
    """
    Nhiệm vụ: Tìm trường biến dạng u(x) và kéo (warp) mọi thứ theo nó
    Đây là module khó nhất - giải phương trình tịnh tiến Pha và Warp ảnh
    """
    def __init__(self, biort='near_sym_b', qshift='qshift_b'):
        super().__init__()
        self.dtcwt_phase = DTCWTForward(J=1, biort=biort, qshift=qshift)
        
        # DTCWT 6 orientations in radians: 15°, 45°, 75°, 105°, 135°, 165°
        self.orient_angles = [
            15 * math.pi / 180, 45 * math.pi / 180, 75 * math.pi / 180,
            105 * math.pi / 180, 135 * math.pi / 180, 165 * math.pi / 180,
        ]
        
        # Precompute wave-vector matrix K (6, 2)
        freq = math.pi / 2  # Center frequency at scale 1
        K = torch.zeros(6, 2)
        for d, theta in enumerate(self.orient_angles):
            K[d, 0] = freq * math.sin(theta)  # ky
            K[d, 1] = freq * math.cos(theta)  # kx
        self.register_buffer('K', K)
    
    def _extract_y_phase(self, ll_y):
        """Trích xuất pha cục bộ từ kênh Y. Input HxW, Output (H/2)x(W/2)"""
        with torch.amp.autocast('cuda', enabled=False):
            ll_y = ll_y.float()
            _, yh = self.dtcwt_phase(ll_y)
            hf = yh[0]  # [B, 1, 6, H/2, W/2, 2]
            real = hf[..., 0].squeeze(1)  # [B, 6, H/2, W/2]
            imag = hf[..., 1].squeeze(1)
            phase = torch.atan2(imag, real)  
            return phase
    
    def _solve_displacement_field(self, phi_s, phi_t):
        """
        Giải hệ phương trình Least-Squares TỐC ĐỘ CAO (Vectorized)
        delta_phi = k · u  (6 phương trình, 2 ẩn số)
        """
        with torch.amp.autocast('cuda', enabled=False):
            delta_phi = _wrap_phase(phi_t - phi_s)  # [B, 6, H, W]
            B, _, H, W = delta_phi.shape
            K = self.K  # [6, 2]
            
            KtK = K.T @ K  # [2, 2]
            Kt = K.T       # [2, 6]
            
            # Tính ma trận nghịch đảo KtK^-1 (chỉ [2x2] nên cực nhanh)
            KtK_inv = torch.linalg.inv(KtK)  # [2, 2]
            
            # Vector hóa: Ép delta_phi về [B*H*W, 6]
            delta_phi_flat = delta_phi.permute(0, 2, 3, 1).reshape(-1, 6)
            
            # Tính RHS (Right Hand Side) = Kt @ delta_phi
            # [2, 6] @ [6, B*H*W] -> [2, B*H*W]
            rhs = Kt @ delta_phi_flat.T
            
            # Giải u = KtK_inv @ RHS
            # [2, 2] @ [2, B*H*W] -> [2, B*H*W]
            u_flat = KtK_inv @ rhs
            
            # Đưa về lại kích thước ảnh [B, 2, H, W]
            u_field = u_flat.T.reshape(B, H, W, 2).permute(0, 3, 1, 2)
            
            return u_field.detach()  # Stop gradient
    
    def _make_warp_grid(self, u):
        """Chuyển displacement thành grid cho F.grid_sample"""
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
    
    def forward(self, ll_src, ll_tgt_pred, mag1, phase1, mag2, phase2):
        """
        Input: ll_src [B, 3, 112, 112], ll_tgt_pred [B, 3, 112, 112]
               mag1, phase1 [B, 18, 112, 112], mag2, phase2 [B, 18, 56, 56]
        Output: u_field [B, 2, 112, 112], warped mag/phase
        """
        # Trích xuất kênh Y (112x112)
        ll_src_y = ll_src[:, 0:1]  
        ll_tgt_y = ll_tgt_pred[:, 0:1]  
        
        # Upsample LL về 224x224 để khi qua DTCWT(J=1), pha sẽ có kích thước 112x112
        ll_src_y_up = F.interpolate(ll_src_y, size=(224, 224), mode='bilinear', align_corners=False)
        ll_tgt_y_up = F.interpolate(ll_tgt_y, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Trích xuất pha (Kết quả sẽ là 112x112)
        phi_s = self._extract_y_phase(ll_src_y_up)  # [B, 6, 112, 112]
        phi_t = self._extract_y_phase(ll_tgt_y_up)  # [B, 6, 112, 112]
        
        # Giải trường dịch chuyển gốc tại 112x112
        u_field_112 = self._solve_displacement_field(phi_s, phi_t)  # [B, 2, 112, 112]
        
        # Tạo u_field cho 56x56 (Nhân 0.5 vì không gian giảm một nửa)
        u_field_56 = F.interpolate(u_field_112, size=(56, 56), mode='bilinear', align_corners=False) * 0.5
        
        # Tạo warp grid
        grid_112 = self._make_warp_grid(u_field_112)  # [B, 112, 112, 2]
        grid_56 = self._make_warp_grid(u_field_56)    # [B, 56, 56, 2]
        
        # Warp magnitude và phase level 1 (112x112)
        mag1_warp = F.grid_sample(mag1, grid_112, mode='bilinear', padding_mode='border', align_corners=False)
        phase1_warp = F.grid_sample(phase1, grid_112, mode='bilinear', padding_mode='border', align_corners=False)
        
        K = self.K  # [6, 2]
        for d in range(6):
            phase_corr = K[d, 0] * u_field_112[:, 0:1] + K[d, 1] * u_field_112[:, 1:2]
            phase1_warp[:, d:d+1] += phase_corr
            phase1_warp[:, d+6:d+7] += phase_corr
            phase1_warp[:, d+12:d+13] += phase_corr
        phase1_warp = _wrap_phase(phase1_warp)
        
        # Warp magnitude và phase level 2 (56x56)
        mag2_warp = F.grid_sample(mag2, grid_56, mode='bilinear', padding_mode='border', align_corners=False)
        phase2_warp = F.grid_sample(phase2, grid_56, mode='bilinear', padding_mode='border', align_corners=False)
        
        for d in range(6):
            phase_corr = K[d, 0] * u_field_56[:, 0:1] + K[d, 1] * u_field_56[:, 1:2]
            phase2_warp[:, d:d+1] += phase_corr
            phase2_warp[:, d+6:d+7] += phase_corr
            phase2_warp[:, d+12:d+13] += phase_corr
        phase2_warp = _wrap_phase(phase2_warp)
        
        return u_field_112, mag1_warp, phase1_warp, mag2_warp, phase2_warp


# ============================================================================
# Module 4: BirthDeath_Innovation
# ============================================================================
class BirthDeath_Innovation(nn.Module):
    """
    Nhiệm vụ: Tìm vùng miệng mở/nhắm và vẽ chi tiết HF mới
    """
    def __init__(self, temperature=10.0):
        super().__init__()
        self.temperature = temperature
        
        # Mạng Innovation cho level 1 (input 112x112)
        self.innovation_net1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 36, 3, padding=1)  # 18 mag + 18 phase
        )
        
        # Mạng Innovation cho level 2 (56x56)
        self.innovation_net2 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 36, 3, padding=1)  # 18 mag + 18 phase
        )
        
        # DTCWT để tính oriented energy
        self.dtcwt_energy = DTCWTForward(J=1, biort='near_sym_b', qshift='qshift_b')
    
    def _compute_oriented_energy(self, ll):
        """Tính Oriented Energy (Fix kích thước)"""
        with torch.amp.autocast('cuda', enabled=False):
            ll = ll.float()
            # Phóng to LL lên 2 lần trước khi DTCWT để giữ nguyên kích thước gốc
            ll_up = F.interpolate(ll, scale_factor=2.0, mode='bilinear', align_corners=False)
            _, yh = self.dtcwt_energy(ll_up)
            hf = yh[0]  # [B, 3, 6, H, W, 2]
            energy = torch.sum(torch.abs(hf), dim=[2, 5])  # [B, 3, H, W]
            return torch.sum(energy, dim=1, keepdim=True)  # [B, 1, H, W]
    
    def forward(self, ll_src_warped, ll_tgt_pred, mag1_warp, phase1_warp, mag2_warp, phase2_warp):
        """
        Input: ll_src_warped [B, 3, 112, 112], ll_tgt_pred [B, 3, 112, 112]
               mag1_warp, phase1_warp [B, 18, 112, 112]
               mag2_warp, phase2_warp [B, 18, 56, 56]
        Output: mag1_final, phase1_final, mag2_final, phase2_final
        """
        # Tính Oriented Energy
        E_tgt = self._compute_oriented_energy(ll_tgt_pred)  # [B, 1, 112, 112]
        E_warp = self._compute_oriented_energy(ll_src_warped)  # [B, 1, 112, 112]
        
        # Tính Masks (112x112)
        M_birth_112 = torch.sigmoid((E_tgt - E_warp) * self.temperature)  # [B, 1, 112, 112]
        M_death_112 = torch.sigmoid((E_warp - E_tgt) * self.temperature)  # [B, 1, 112, 112]
        
        # Resize masks cho level 2
        M_birth_56 = F.interpolate(M_birth_112, size=(56, 56), mode='bilinear', align_corners=False)
        M_death_56 = F.interpolate(M_death_112, size=(56, 56), mode='bilinear', align_corners=False)
        
        # Innovation networks
        ll_tgt_56 = F.interpolate(ll_tgt_pred, size=(56, 56), mode='bilinear', align_corners=False)
        res1 = self.innovation_net1(ll_tgt_pred)  # [B, 36, 112, 112]
        res2 = self.innovation_net2(ll_tgt_56)    # [B, 36, 56, 56]
        
        # Tách res thành mag và phase
        res_mag1 = res1[:, :18]  # [B, 18, 112, 112]
        res_phase1 = res1[:, 18:]  # [B, 18, 112, 112]
        res_mag2 = res2[:, :18]    # [B, 18, 56, 56]
        res_phase2 = res2[:, 18:]  # [B, 18, 56, 56]
        
        # Tổng hợp kết quả cuối cùng
        # Level 1
        mag1_final = (1 - M_death_112) * mag1_warp + M_birth_112 * res_mag1
        phase1_final = (1 - M_death_112) * phase1_warp + M_birth_112 * res_phase1
        
        # Level 2
        mag2_final = (1 - M_death_56) * mag2_warp + M_birth_56 * res_mag2
        phase2_final = (1 - M_death_56) * phase2_warp + M_birth_56 * res_phase2
        
        return mag1_final, phase1_final, mag2_final, phase2_final


# ============================================================================
# Module 5: DTCWT_Decoder
# ============================================================================
class DTCWT_Decoder(nn.Module):
    """
    Nhiệm vụ: Gom thành ảnh cuối
    """
    def __init__(self, biort='near_sym_b', qshift='qshift_b'):
        super().__init__()
        self.idtcwt = DTCWTInverse(biort=biort, qshift=qshift)
    
    def forward(self, ll_tgt_pred, mag1_final, phase1_final, mag2_final, phase2_final):
        """
        Input: ll_tgt_pred [B, 3, 56, 56]
               mag1_final, phase1_final [B, 18, 112, 112]
               mag2_final, phase2_final [B, 18, 56, 56]
        Output: pred_rgb [B, 3, 224, 224]
        """
        with torch.amp.autocast('cuda', enabled=False):
            ll_tgt_pred = ll_tgt_pred.float()
            mag1_final = mag1_final.float()
            phase1_final = phase1_final.float()
            mag2_final = mag2_final.float()
            phase2_final = phase2_final.float()
            
            B = ll_tgt_pred.shape[0]
            
            # Ép hệ tọa độ cực về số phức cho level 1
            real1 = mag1_final * torch.cos(phase1_final)  # [B, 18, 112, 112]
            imag1 = mag1_final * torch.sin(phase1_final)  # [B, 18, 112, 112]
            
            # Reshape về [B, 3, 6, H, W]
            real1 = real1.reshape(B, 3, 6, 112, 112)
            imag1 = imag1.reshape(B, 3, 6, 112, 112)
            
            # Ép hệ tọa độ cực về số phức cho level 2
            real2 = mag2_final * torch.cos(phase2_final)  # [B, 18, 56, 56]
            imag2 = mag2_final * torch.sin(phase2_final)  # [B, 18, 56, 56]
            
            # Reshape về [B, 3, 6, H, W]
            real2 = real2.reshape(B, 3, 6, 56, 56)
            imag2 = imag2.reshape(B, 3, 6, 56, 56)
            
            # Gộp thành complex coefficients
            hf1 = torch.stack([real1, imag1], dim=-1)  # [B, 3, 6, 112, 112, 2]
            hf2 = torch.stack([real2, imag2], dim=-1)  # [B, 3, 6, 56, 56, 2]
            
            # IDTCWT reconstruction (same order as forward: yh[0]=hf1@112, yh[1]=hf2@56)
            ycbcr = self.idtcwt((ll_tgt_pred, [hf1, hf2]))  # [B, 3, 224, 224]
            
            # Chuyển YCbCr về RGB
            pred_rgb = ycbcr_to_rgb(ycbcr)
        
        return pred_rgb


# ============================================================================
# Main Model: DTCWT_Emotion_Model
# ============================================================================
class DTCWT_Emotion_Model(nn.Module):
    """
    Mô hình chính kết hợp 5 modules
    Với cơ chế điều khiển cảm xúc nâng cao: StarGAN injection + FiLM/AdaGN + CrossAttention
    """
    def __init__(self, num_emotions=7, emotion_dim=64, time_dim=256,
                 use_film=True, use_adagn=False):
        super().__init__()
        self.num_emotions = num_emotions
        
        # Module 1: Encoder
        self.encoder = DTCWT_Encoder()
        
        # Module 2: UNet — input = 3 (LL YCbCr) + num_emotions (one-hot map)
        self.unet = LL_Diffusion_UNet(
            in_channels=3 + num_emotions, out_channels=3,
            time_dim=time_dim, emotion_dim=emotion_dim,
            use_film=use_film, use_adagn=use_adagn
        )
        
        # Module 3: WPTL
        self.wptl = WPTL_Layer()
        
        # Module 4: BirthDeath Innovation
        self.birth_death = BirthDeath_Innovation()
        
        # Module 5: Decoder
        self.decoder = DTCWT_Decoder()
        
        # Emotion embedding (for conditioning inside ResBlocks)
        self.emotion_embedding = nn.Embedding(num_emotions, emotion_dim)
    
    def _inject_emotion(self, x_wavelet, emotion_id):
        """
        StarGAN-style injection: concatenate one-hot emotion map to LL input.
        
        Args:
            x_wavelet: (B, 3, H, W) — LL component (YCbCr)
            emotion_id: (B,) — emotion indices
        Returns:
            (B, 3 + num_emotions, H, W)
        """
        B, C, H, W = x_wavelet.shape
        emotion_onehot = F.one_hot(emotion_id, num_classes=self.num_emotions).to(dtype=x_wavelet.dtype)
        emotion_map = emotion_onehot[:, :, None, None].expand(B, self.num_emotions, H, W)
        return torch.cat([x_wavelet, emotion_map], dim=1)
    
    def forward(self, x_rgb, emotion_idx, timestep):
        """
        Forward pass với emotion conditioning nâng cao
        Args:
            x_rgb: [B, 3, 224, 224]
            emotion_idx: [B] (indices 0-6)
            timestep: [B] (diffusion timesteps)
        Returns:
            pred_rgb, ll_tgt_pred, u_field
        """
        # Embed emotion
        emotion_emb = self.emotion_embedding(emotion_idx)
        
        # 1. Tách cấu trúc (LL) và chi tiết (HF)
        ll_src, mag1, phase1, mag2, phase2 = self.encoder(x_rgb)
        
        # 2. StarGAN injection: concat one-hot emotion map vào LL
        ll_concat = self._inject_emotion(ll_src, emotion_idx)
        
        # 3. Diffusion chỉ dự đoán LL đích (Học ngữ nghĩa) — with emotion conditioning
        ll_tgt_pred = self.unet(ll_concat, timestep, emotion_emb)
        
        # 4. WPTL: Giải mã trường dịch chuyển và tịnh tiến HF (Toán học)
        u_field, mag1_warp, phase1_warp, mag2_warp, phase2_warp = self.wptl(
            ll_src, ll_tgt_pred, mag1, phase1, mag2, phase2
        )
        
        # 5. Quản lý vùng Sinh/Diệt (Topology)
        ll_src_warped = F.grid_sample(ll_src, self.wptl._make_warp_grid(u_field), 
                                     mode='bilinear', padding_mode='border', align_corners=False)
        
        mag1_fin, phase1_fin, mag2_fin, phase2_fin = self.birth_death(
            ll_src_warped=ll_src_warped,
            ll_tgt_pred=ll_tgt_pred,
            mag1_warp=mag1_warp, phase1_warp=phase1_warp,
            mag2_warp=mag2_warp, phase2_warp=phase2_warp
        )
        
        # 6. Tái tạo ảnh RGB cuối cùng
        pred_rgb = self.decoder(ll_tgt_pred, mag1_fin, phase1_fin, mag2_fin, phase2_fin)
        
        return pred_rgb, ll_tgt_pred, u_field

    @torch.no_grad()
    def sample(self, x_rgb, emotion_idx, num_steps: int = 20, strength: float = 0.7):
        """
        Rectified Flow ODE inference for emotion editing.

        Strategy: add noise to ll_src at level `strength`, then run Euler
        integration from t=strength → t=0 conditioned on the target emotion.
        strength=1.0  →  start from pure noise (full generation)
        strength≈0.5  →  light edit, preserves most source structure

        Args:
            x_rgb:       [B, 3, 224, 224]
            emotion_idx: [B] (0-6)
            num_steps:   Euler discretisation steps
            strength:    noise level to start from (0, 1]
        Returns:
            pred_rgb: [B, 3, 224, 224]
        """
        B, _, H, W = x_rgb.shape
        device = x_rgb.device

        emotion_emb = self.emotion_embedding(emotion_idx)
        ll_src, mag1, phase1, mag2, phase2 = self.encoder(x_rgb)

        # StarGAN injection cho sampling
        ll_concat = self._inject_emotion(ll_src, emotion_idx)

        # Start from ll_src + noise at t = strength
        eps   = torch.randn_like(ll_src)
        ll_t  = (1.0 - strength) * ll_src + strength * eps

        # Euler ODE:  dx/dt = v_theta(x_t, c, t)  integrated from t→0
        dt = -strength / num_steps
        for k in range(num_steps):
            t_cur     = strength + k * dt          # decreasing  strength→0
            t_tensor  = torch.full((B,), t_cur, device=device)
            timestep  = (t_tensor * 999).long().clamp(0, 999)
            # Concat emotion map với ll_t hiện tại cho mỗi step
            ll_t_concat = self._inject_emotion(ll_t, emotion_idx)
            v         = self.unet(ll_t_concat, timestep, emotion_emb)
            ll_t      = ll_t + dt * v              # dt < 0

        ll_tgt_pred = ll_t

        # WPTL → BirthDeath → Decoder  (same as forward())
        u_field, mag1_warp, phase1_warp, mag2_warp, phase2_warp = self.wptl(
            ll_src, ll_tgt_pred, mag1, phase1, mag2, phase2
        )
        ll_src_warped = F.grid_sample(
            ll_src, self.wptl._make_warp_grid(u_field),
            mode='bilinear', padding_mode='border', align_corners=False
        )
        mag1_fin, phase1_fin, mag2_fin, phase2_fin = self.birth_death(
            ll_src_warped, ll_tgt_pred,
            mag1_warp, phase1_warp, mag2_warp, phase2_warp
        )
        pred_rgb = self.decoder(ll_tgt_pred, mag1_fin, phase1_fin, mag2_fin, phase2_fin)
        return pred_rgb


# ============================================================================
# Utility function for testing
# ============================================================================
def test_model():
    """Test function to verify model works"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DTCWT_Emotion_Model().to(device)
    
    # Print parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DTCWT_Emotion_Model parameters:")
    print(f"  Total:      {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  Trainable:  {trainable_params:,} ({trainable_params/1e6:.1f}M)")
    
    # Test inputs
    batch_size = 2
    x_rgb = torch.randn(batch_size, 3, 224, 224).to(device)
    emotion_idx = torch.randint(0, 7, (batch_size,)).to(device)
    timestep = torch.randint(0, 1000, (batch_size,)).to(device)
    
    # Forward pass
    with torch.no_grad():
        pred_rgb, ll_tgt_pred, u_field = model(x_rgb, emotion_idx, timestep)
    
    # print(f"Input RGB: {x_rgb.shape}")
    # print(f"Output RGB: {pred_rgb.shape}")
    # print(f"LL target: {ll_tgt_pred.shape}")
    # print(f"Displacement field: {u_field.shape}")
    # print("Model test passed!")


if __name__ == "__main__":
    test_model()
