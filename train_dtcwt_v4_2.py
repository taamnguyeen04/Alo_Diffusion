"""
Training script for DTCWT_Emotion_Model (v4.2)

Losses:
- L_flow     : Rectified Flow velocity matching on LL subband
- L_mag      : HF magnitude preservation in birth regions
- L_chroma   : Cb/Cr channel constraint on LL
- L_smooth   : Total-Variation smoothness on displacement field u
- L_rec      : Spatial L1 reconstruction loss
- L_LPIPS    : Perceptual (LPIPS/VGG) loss
- L_ID       : Identity cosine loss (optional, requires ArcFace)
- L_cls      : Emotion classification loss (DAN classifier)
"""

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Resize, ToTensor, Compose, Normalize
from torchvision.utils import save_image
from tqdm import tqdm
import lpips

from model_dtcwt_v4_2 import DTCWT_Emotion_Model
from criterions import DAN
from dataset import Affectnet


# ---------------------------------------------------------------------------
# Helper: Total-Variation smoothness loss on displacement field
# ---------------------------------------------------------------------------
def tv_loss(u: torch.Tensor) -> torch.Tensor:
    """L1 TV on u [B, 2, H, W] — Eq. L_smooth."""
    return ((u[:, :, 1:, :] - u[:, :, :-1, :]).abs().mean() +
            (u[:, :, :, 1:] - u[:, :, :, :-1]).abs().mean())


class TrainingWrapper(nn.Module):
    """
    Wraps DTCWT_Emotion_Model + all loss functions for DataParallel.
    Implements Rectified Flow training: samples t, builds LL_t, predicts
    velocity v_theta, then runs the full WPTL + BirthDeath + Decoder pipeline
    on the predicted clean LL to compute all reconstruction losses.
    """
    def __init__(self, model: DTCWT_Emotion_Model, lpips_fn, arcface=None, dan_model=None):
        super().__init__()
        self.model = model
        self.lpips_fn = lpips_fn
        self.arcface = arcface   # optional frozen ArcFace for L_ID
        self.dan_model = dan_model  # optional frozen DAN for L_cls
        # ImageNet normalization for DAN input
        self.register_buffer('dan_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('dan_std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x_rgb: torch.Tensor, emotion_idx: torch.Tensor,
                drop_prob: float = 0.0):
        B, _, H, W = x_rgb.shape
        device = x_rgb.device

        # ── Emotion embedding ────────────────────────────────────────────────
        emotion_emb = self.model.emotion_embedding(emotion_idx)

        # ── 1. Encode source into LL + HF ───────────────────────────────────
        ll_src, mag1, phase1, mag2, phase2 = self.model.encoder(x_rgb)
        # ll_src: [B, 3, 112, 112]   (Y Cb Cr)
        # mag1:   [B, 18, 112, 112], mag2:  [B, 18, 56, 56]

        # ── 2. Rectified Flow on LL ──────────────────────────────────────────
        # x_t = (1-t)*x_0 + t*eps,  v_target = eps - x_0
        t = torch.rand(B, device=device)           # t ~ U[0,1]
        eps = torch.randn_like(ll_src)
        t4 = t.view(B, 1, 1, 1)
        ll_t = (1 - t4) * ll_src + t4 * eps

        timestep = (t * 999).long().clamp(0, 999)
        ll_t_concat = self.model._inject_emotion(ll_t, emotion_idx)
        v_theta = self.model.unet(ll_t_concat, timestep, emotion_emb)
        v_target = eps - ll_src

        # L_flow — Eq. 1.1
        L_flow = F.mse_loss(v_theta, v_target)

        # Predicted clean LL from velocity:  x_0_hat = x_t - t * v_theta
        ll_tgt_pred = ll_t - t4 * v_theta          # [B, 3, 112, 112]

        # ── 3. L_chroma — keep Cb/Cr of LL unchanged ── Eq. 1.3 ──────────────
        L_chroma = F.l1_loss(ll_tgt_pred[:, 1:3], ll_src[:, 1:3])

        # ── 4. WPTL: warp HF using phase transport ───────────────────────────
        ll_tgt_det = ll_tgt_pred.detach()
        u_field, mag1_warp, phase1_warp, mag2_warp, phase2_warp = self.model.wptl(
            ll_src, ll_tgt_det, mag1, phase1, mag2, phase2
        )

        # ── 5. L_smooth — TV on displacement field ── Eq. 2.1 ─────────────────
        L_smooth = tv_loss(u_field)

        # Warp ll_src for BirthDeath energy computation
        ll_src_warped = F.grid_sample(
            ll_src,
            self.model.wptl._make_warp_grid(u_field),
            mode='bilinear', padding_mode='border', align_corners=False
        )

        # ── 6. BirthDeath Innovation ─────────────────────────────────────────
        mag1_fin, phase1_fin, mag2_fin, phase2_fin = self.model.birth_death(
            ll_src_warped, ll_tgt_det, mag1_warp, phase1_warp, mag2_warp, phase2_warp
        )

        # ── 7. L_mag — HF magnitude in birth regions ── Eq. 1.2 ───────────────
        # Approximation: recompute M_birth from energy difference
        with torch.no_grad():
            ll_up_tgt = F.interpolate(ll_tgt_det, scale_factor=2.0,
                                      mode='bilinear', align_corners=False)
            ll_up_warp = F.interpolate(ll_src_warped, scale_factor=2.0,
                                       mode='bilinear', align_corners=False)
            _, yh_tgt  = self.model.birth_death.dtcwt_energy(ll_up_tgt.float())
            _, yh_warp = self.model.birth_death.dtcwt_energy(ll_up_warp.float())
            def _energy(yh):
                hf = yh[0]  # [B, 3, 6, H, W, 2]
                return torch.sum(torch.abs(hf), dim=[2, 5]).sum(1, keepdim=True)
            E_tgt_  = _energy(yh_tgt)
            E_warp_ = _energy(yh_warp)
            M_birth_112 = torch.sigmoid((E_tgt_ - E_warp_) * 10.0)  # [B,1,112,112]
            M_birth_56  = F.interpolate(M_birth_112, size=(56, 56),
                                        mode='bilinear', align_corners=False)

        L_mag = (F.l1_loss(M_birth_112 * mag1_fin, M_birth_112 * mag1) +
                 F.l1_loss(M_birth_56  * mag2_fin, M_birth_56  * mag2))

        # ── 8. Decode to RGB ─────────────────────────────────────────────────
        pred_rgb = self.model.decoder(
            ll_tgt_pred, mag1_fin, phase1_fin, mag2_fin, phase2_fin
        )
        pred_clamped = pred_rgb.clamp(-1, 1)
        x_clamped    = x_rgb.clamp(-1, 1)

        # ── 9. L_rec — spatial L1 ── Eq. 3.1 ─────────────────────────────────
        L_rec = F.l1_loss(pred_rgb, x_rgb)

        # ── 10. L_LPIPS — perceptual ── Eq. 3.2 ──────────────────────────────
        L_lpips = self.lpips_fn(pred_clamped, x_clamped).mean()

        # ── 11. L_ID — identity cosine ── Eq. 3.3 ────────────────────────────
        if self.arcface is not None:
            # Resize to 160x160 (InceptionResnetV1 / ArcFace standard input)
            x_160    = F.interpolate(x_clamped,    size=(160, 160), mode='bilinear', align_corners=False)
            pred_160 = F.interpolate(pred_clamped, size=(160, 160), mode='bilinear', align_corners=False)
            with torch.no_grad():
                f_src  = self.arcface(x_160)
            f_pred = self.arcface(pred_160)
            L_id = (1 - F.cosine_similarity(f_pred, f_src, dim=1)).mean()
        else:
            L_id = pred_rgb.new_zeros(1).squeeze()

        # ── 12. L_cls — DAN emotion classification ── Eq. 4.1 ────────────────
        if self.dan_model is not None:
            # Resize pred to 224×224, unnormalize from [-1,1] to [0,1], then apply ImageNet norm
            pred_224 = F.interpolate(pred_clamped, size=(224, 224), mode='bilinear', align_corners=False)
            pred_01  = (pred_224 + 1) / 2  # [-1,1] -> [0,1]
            pred_dan = (pred_01 - self.dan_mean) / self.dan_std
            with torch.amp.autocast('cuda', enabled=False):
                dan_logits, _, _ = self.dan_model(pred_dan.float())
            L_cls = F.cross_entropy(dan_logits, emotion_idx)
        else:
            L_cls = pred_rgb.new_zeros(1).squeeze()

        # ── Metrics (no grad) ────────────────────────────────────────────────
        with torch.no_grad():
            mse  = F.mse_loss(pred_clamped, x_clamped)
            psnr = 10 * torch.log10(torch.tensor(4.0, device=device) / (mse + 1e-8))
            u_mean = u_field.abs().mean()
            u_max  = u_field.abs().max()

        return {
            'L_flow':   L_flow,
            'L_mag':    L_mag,
            'L_chroma': L_chroma,
            'L_smooth': L_smooth,
            'L_rec':    L_rec,
            'L_lpips':  L_lpips,
            'L_id':     L_id,
            'L_cls':    L_cls,
            'pred_rgb': pred_clamped,
            'psnr':     psnr,
            'u_mean':   u_mean,
            'u_max':    u_max,
        }


def save_checkpoint(filepath, epoch, step, model, optimizer, loss):
    print(f"Saving checkpoint: epoch {epoch}, step {step}, loss {loss:.4f}")
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'loss': loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    last_path = os.path.join(filepath, "last_model.pt")
    torch.save(checkpoint, last_path)

    best_path = os.path.join(filepath, "best_model.pt")
    if not os.path.exists(best_path):
        torch.save(checkpoint, best_path)
        print("Saved best_model.pt (first time)")
    else:
        try:
            best_loss = torch.load(best_path, map_location='cpu', weights_only=True)['loss']
            if loss < best_loss:
                torch.save(checkpoint, best_path)
                print(f"Updated best_model.pt: {best_loss:.4f} → {loss:.4f}")
        except Exception as e:
            print(f"Error reading best_model.pt: {e} → overwriting")
            torch.save(checkpoint, best_path)


def load_checkpoint(filepath, model, optimizer, device):
    last_path = os.path.join(filepath, "last_model.pt")
    best_path = os.path.join(filepath, "best_model.pt")

    start_epoch = 0
    step = 0
    best_loss = float('inf')

    for path_to_try in [last_path, best_path]:
        if os.path.isfile(path_to_try):
            try:
                checkpoint = torch.load(path_to_try, map_location=device, weights_only=True)
                # Use strict=False to allow adding new modules like DGWM to existing checkpoints
                missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                if len(missing) > 0:
                    print(f"  Missing keys (initialized fresh): {len(missing)}")
                if len(unexpected) > 0:
                    print(f"  Unexpected keys (ignored): {len(unexpected)}")
                
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch']
                step = checkpoint['step']
                print(f"Loaded checkpoint from {os.path.basename(path_to_try)}, epoch {start_epoch}, step {step}")
                break
            except Exception as e:
                print(f"Error loading {os.path.basename(path_to_try)}: {e}")

    if os.path.isfile(best_path):
        try:
            best_loss = torch.load(best_path, map_location='cpu', weights_only=True)['loss']
        except:
            best_loss = float('inf')

    return start_epoch, best_loss, step


def train():
    # ============================
    # Setup
    # ============================
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"Device: {device}, GPUs: {num_gpus}")

    # ============================
    # Hyperparameters
    # ============================
    batch_size = 4
    lr = 4e-4
    num_epochs = 10
    image_size = 224
    labels = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"]
    num_emotions = len(labels)

    # Loss weights  (see paper Sec. 4 for rationale)
    lambda_flow   = 1.0    # RF velocity matching — train expression first
    lambda_mag    = 0.5    # HF magnitude in birth regions
    lambda_chroma = 0.1    # Cb/Cr constraint on LL
    lambda_smooth = 0.5    # TV smoothness on u-field — stabilise early
    lambda_rec    = 0.5    # Spatial L1 reconstruction
    lambda_lpips  = 1.5    # Perceptual quality
    lambda_id     = 0.3    # Identity cosine (0 when no ArcFace)
    lambda_cls    = 0.3   # DAN emotion classifier — start small, increase in Stage 2

    print(f"Batch size: {batch_size}")

    # ============================
    # Directories
    # ============================
    exp_name = "DTCWT_V4_2"
    log_dir = f"{exp_name}/runs/exp"
    model_path = f"{exp_name}/model"
    out_path = f"{exp_name}/out"

    for dir_path in [log_dir, model_path, out_path]:
        os.makedirs(dir_path, exist_ok=True)

    writer = SummaryWriter(log_dir)
    best_loss = float('inf')

    # ============================
    # Data
    # ============================
    transform = Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading datasets...")
    train_dataset = Affectnet(is_train=True, transform=transform)
    val_dataset = Affectnet(is_train=False, transform=transform)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=min(os.cpu_count(), 8),
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=min(4, os.cpu_count()),
        shuffle=False,
        drop_last=True
    )
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # ============================
    # Models
    # ============================
    print("Building models...")
    diff_model = DTCWT_Emotion_Model(num_emotions=num_emotions).to(device)

    total_params     = sum(p.numel() for p in diff_model.parameters())
    trainable_params = sum(p.numel() for p in diff_model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"Trainable params: {trainable_params:,} ({trainable_params/1e6:.1f}M)")

    # LPIPS (frozen)
    lpips_fn = lpips.LPIPS(net='vgg').to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad = False

    # ArcFace identity model (InceptionResnetV1 from facenet-pytorch)
    from facenet_pytorch import InceptionResnetV1
    print("Loading ArcFace model...")
    arcface = InceptionResnetV1(pretrained='vggface2').to(device).eval()
    for p in arcface.parameters():
        p.requires_grad = False

    # DAN emotion classifier (frozen)
    print("Loading DAN emotion classifier...")
    dan_model = DAN(num_class=7, num_head=4, pretrained=False)
    dan_ckpt = torch.load('affecnet7_epoch6_acc0.6569.pth', map_location=device)
    dan_model.load_state_dict(dan_ckpt['model_state_dict'], strict=True)
    dan_model.to(device).eval()
    for p in dan_model.parameters():
        p.requires_grad = False
    print("DAN loaded OK")

    # Training wrapper
    model = TrainingWrapper(diff_model, lpips_fn, arcface=arcface, dan_model=dan_model).to(device)

    if num_gpus > 1:
        print(f"Using DataParallel with {num_gpus} GPUs")
        model = torch.nn.DataParallel(model)

    base_model = model.module.model if num_gpus > 1 else diff_model

    # ============================
    # Optimizer
    # ============================
    # UNet learns faster; WPTL + BirthDeath + Decoder learn slightly slower
    unet_params       = list(diff_model.unet.parameters())
    other_params      = (list(diff_model.encoder.parameters()) +
                         list(diff_model.wptl.parameters()) +
                         list(diff_model.birth_death.parameters()) +
                         list(diff_model.decoder.parameters()) +
                         list(diff_model.emotion_embedding.parameters()))

    print(f"UNet params: {sum(p.numel() for p in unet_params):,}, "
          f"Other params: {sum(p.numel() for p in other_params):,}")

    optimizer = torch.optim.AdamW([
        {'params': unet_params,  'lr': lr,        'weight_decay': 1e-4},
        {'params': other_params, 'lr': lr * 0.5,  'weight_decay': 1e-4},
    ], betas=(0.9, 0.999))

    # AMP mixed precision for VRAM savings (~50%)
    # scaler = torch.amp.GradScaler('cuda')
    scaler = torch.cuda.amp.GradScaler()
    print("AMP fp16 enabled")

    # Load checkpoint
    print("Loading checkpoint...")
    start_epoch, best_loss, step = load_checkpoint(model_path, base_model, optimizer, device)

    # (Fresh RF training — no mag_scale reset needed)

    # Fixed validation samples
    x_fixed, expr_fixed, _, _ = next(iter(val_dataloader))
    x_fixed = x_fixed.to(device)
    expr_fixed = expr_fixed.to(device)

    # ============================
    # Training Loop
    # ============================
    try:
        print("Starting training...")
        for epoch in range(start_epoch, num_epochs):
            model.train()

            # If resuming mid-epoch, skip already-processed batches
            skip_batches = step % len(train_dataloader) if epoch == start_epoch and step > 0 else 0

            epoch_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                             desc=f"Epoch {epoch}/{num_epochs}")

            total_iters = len(train_dataloader)
            log_every = max(1, total_iters // 40)    # Log ~20 times per epoch
            ckpt_every = max(1, total_iters // 5)    # Checkpoint ~3 times per epoch
            img_every = max(1, total_iters // 2)     # Images ~2 times per epoch (mid + end)

            for i, (img_real, expr_org, _, _) in epoch_bar:
                # Skip batches already processed (when resuming mid-epoch)
                if skip_batches > 0:
                    skip_batches -= 1
                    continue

                img_real = img_real.to(device)
                expr_org = expr_org.to(device)

                # ===== FORWARD PASS (AMP + DataParallel) =====
                # with torch.amp.autocast('cuda'):
                with torch.cuda.amp.autocast():
                    outputs = model(img_real, expr_org)

                    # DataParallel returns (num_gpus,) tensors — reduce with .mean()
                    L_flow   = outputs['L_flow'].mean()
                    L_mag    = outputs['L_mag'].mean()
                    L_chroma = outputs['L_chroma'].mean()
                    L_smooth = outputs['L_smooth'].mean()
                    L_rec    = outputs['L_rec'].mean()
                    L_lpips  = outputs['L_lpips'].mean()
                    L_id     = outputs['L_id'].mean()
                    L_cls    = outputs['L_cls'].mean()
                    psnr     = outputs['psnr'].mean()

                    # ===== TOTAL LOSS — Eq. 4 =====
                    total_loss = (
                        lambda_flow   * L_flow   +
                        lambda_mag    * L_mag    +
                        lambda_chroma * L_chroma +
                        lambda_smooth * L_smooth +
                        lambda_rec    * L_rec    +
                        lambda_lpips  * L_lpips  +
                        lambda_id     * L_id     +
                        lambda_cls    * L_cls
                    )

                # ===== NaN GUARD =====
                if not torch.isfinite(total_loss):
                    nan_info = [name for name, val in [
                        ('Flow', L_flow), ('Mag', L_mag), ('Chroma', L_chroma),
                        ('Smooth', L_smooth), ('Rec', L_rec),
                        ('LPIPS', L_lpips), ('ID', L_id), ('CLS', L_cls)
                    ] if not torch.isfinite(val)]
                    print(f"\n⚠ NaN/Inf at epoch {epoch}, step {i} — sources: {nan_info}")
                    optimizer.zero_grad()
                    continue

                # ===== BACKWARD (AMP) =====
                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                # ===== PROGRESS BAR =====
                epoch_bar.set_postfix({
                    "T":  f"{total_loss.item():.3f}",
                    "RF": f"{L_flow.item():.3f}",
                    "Rc": f"{L_rec.item():.3f}",
                    "LP": f"{L_lpips.item():.3f}",
                    "CL": f"{L_cls.item():.3f}",
                    "P":  f"{psnr.item():.1f}",
                })

                step += 1

                if i % log_every == 0:
                    writer.add_scalar("Loss/Total",   total_loss.item(), step)
                    writer.add_scalar("Loss/Flow",    L_flow.item(),     step)
                    writer.add_scalar("Loss/Mag",     L_mag.item(),      step)
                    writer.add_scalar("Loss/Chroma",  L_chroma.item(),   step)
                    writer.add_scalar("Loss/Smooth",  L_smooth.item(),   step)
                    writer.add_scalar("Loss/Rec",     L_rec.item(),      step)
                    writer.add_scalar("Loss/LPIPS",   L_lpips.item(),    step)
                    writer.add_scalar("Loss/ID",      L_id.item(),       step)
                    writer.add_scalar("Loss/CLS",     L_cls.item(),      step)
                    writer.add_scalar("Metric/PSNR",  psnr.item(),       step)
                    writer.add_scalar("Grad/Norm",    grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm, step)
                    writer.add_scalar("APT/U_mean",   outputs['u_mean'].mean().item(), step)
                    writer.add_scalar("APT/U_max",    outputs['u_max'].max().item(),   step)

                # ===== CHECKPOINT (N times per epoch) =====
                if i > 0 and i % ckpt_every == 0:
                    save_checkpoint(model_path, epoch, i, base_model, optimizer, total_loss.item())

                # ===== MID-EPOCH VALIDATION IMAGES =====
                if i > 0 and i % img_every == 0:
                    print(f"Generating mid-epoch validation images (iter {i}/{total_iters})...")
                    model.eval()
                    with torch.no_grad():
                        tensor_images = []
                        n_samples = min(4, x_fixed.shape[0])
                        for sample_idx in range(n_samples):
                            sample = x_fixed[sample_idx:sample_idx+1]
                            orig_tensor = (sample[0].clamp(-1, 1) + 1) / 2
                            tensor_images.append(orig_tensor)
                            for emotion_id in range(num_emotions):
                                emotion_tensor = torch.full((1,), emotion_id, dtype=torch.long, device=device)
                                gen = base_model.sample(sample, emotion_tensor, num_steps=20)
                                gen_tensor = (gen[0].clamp(-1, 1) + 1) / 2
                                tensor_images.append(gen_tensor)
                        if tensor_images:
                            all_imgs = torch.stack(tensor_images, dim=0)
                            n_cols = num_emotions + 1
                            save_image(all_imgs, f"{out_path}/epoch{epoch}_mid_emotions.png",
                                       nrow=n_cols, normalize=False)
                            print(f"Saved: {out_path}/epoch{epoch}_mid_emotions.png")
                            writer.add_images("Validation/Grid", all_imgs[:n_cols], step)
                    model.train()

            # ===== END OF EPOCH: checkpoint + validation images =====
            save_checkpoint(model_path, epoch, total_iters, base_model, optimizer, total_loss.item())
            print(f"Generating validation images for epoch {epoch}...")
            model.eval()
            with torch.no_grad():
                tensor_images = []
                n_samples = min(4, x_fixed.shape[0])

                for sample_idx in range(n_samples):
                    sample = x_fixed[sample_idx:sample_idx+1]

                    # Original
                    orig_tensor = (sample[0].clamp(-1, 1) + 1) / 2
                    tensor_images.append(orig_tensor)

                    # Generate for each emotion
                    for emotion_id in range(num_emotions):
                        emotion_tensor = torch.full((1,), emotion_id, dtype=torch.long, device=device)
                        generated = base_model.sample(sample, emotion_tensor, num_steps=20)
                        gen_tensor = (generated[0].clamp(-1, 1) + 1) / 2
                        tensor_images.append(gen_tensor)

                # Save grid
                if tensor_images:
                    all_imgs = torch.stack(tensor_images, dim=0)
                    n_cols = num_emotions + 1  # +1 for original
                    save_image(
                        all_imgs, 
                        f"{out_path}/epoch{epoch}_emotions.png",
                        nrow=n_cols,
                        normalize=False
                    )
                    print(f"Saved: {out_path}/epoch{epoch}_emotions.png")

                    # TensorBoard
                    step = (epoch + 1) * total_iters
                    writer.add_images("Validation/Grid", all_imgs[:n_cols], step)

            model.train()

    except KeyboardInterrupt:
        print("Training interrupted by user")
        if 'total_loss' in locals():
            save_checkpoint(model_path, epoch, i, base_model, optimizer, total_loss.item())

    print("Training completed!")
    writer.close()


if __name__ == '__main__':
    train()
