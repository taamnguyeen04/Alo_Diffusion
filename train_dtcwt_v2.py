"""
Training script for DirectionalWaveDiffusion (DTCWT v2)

Key differences from train.py:
- Uses DirectionalWaveDiffusionModel (LL-only diffusion + Δmag/Δphase)
- model.forward() returns dict of losses + pred_img
- Adds magnitude preservation loss
- Tracks DTCWT-specific metrics (direction attention, Δmag/Δphase stats)
"""

import os
import sys
import shutil
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Resize, ToTensor, Compose, Normalize
from torchvision.utils import save_image
from tqdm import tqdm
import lpips

from model_dtcwt_v2 import DirectionalWaveDiffusionModel, DTCWTWrapper, IDTCWTWrapper, AnalyticPhaseTransport
from dataset import Affectnet
from criterions import Emotion_model, AdaptiveLossWeighter, PerceptualWaveletLoss_DTCWT, CoarseStructureLoss_DTCWT


class TrainingWrapper(nn.Module):
    """
    Wraps model + loss functions for DataParallel.
    All loss computation happens on each GPU independently.
    Only scalar losses are returned and gathered across GPUs.
    """
    def __init__(self, model, dan_model, lpips_fn, coarse_loss_fn):
        super().__init__()
        self.model = model
        self.dan = dan_model
        self.lpips_fn = lpips_fn
        self.coarse_loss_fn = coarse_loss_fn
        # DAN normalization buffers (auto-replicated to each GPU)
        self.register_buffer('dan_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('dan_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, img_real, expr_org, drop_prob=0.0):
        # Main model forward
        outputs = self.model(img_real, expr_org, src_image=img_real, drop_prob=drop_prob)

        ddpm_loss = outputs['ddpm_loss']
        mag_loss = outputs['mag_loss']
        chroma_loss = outputs['chroma_loss']
        u_smooth_loss = outputs['u_smooth_loss']
        pred_img = outputs['pred_img']

        # DAN loss (computed on each GPU independently)
        pred_img_clamp = torch.clamp(pred_img, -1, 1)
        pred_img_norm = (pred_img_clamp + 1) / 2
        dan_input = F.interpolate(pred_img_norm, size=(224, 224), mode='bilinear', align_corners=False)
        dan_input = (dan_input - self.dan_mean) / self.dan_std
        dan_out, _, _ = self.dan(dan_input)
        dan_expr_loss = F.cross_entropy(dan_out, expr_org, label_smoothing=0.1)

        # Coarse structure loss
        coarse_loss = self.coarse_loss_fn(pred_img_clamp, img_real)

        # LPIPS perceptual loss
        lpips_loss = self.lpips_fn(pred_img_clamp, img_real).mean()

        # PSNR + logging stats (no gradient needed)
        with torch.no_grad():
            mse = F.mse_loss(pred_img_clamp, img_real)
            psnr = 10 * torch.log10(4 / mse)
            dmag = outputs['delta_mag']
            disp = outputs['displacement']
            dmag_mean = dmag.abs().mean()
            dmag_max = dmag.abs().max()
            disp_mean = disp.abs().mean()
            disp_max = disp.abs().max()
            mask = outputs['retain_mask']
            mask_mean = mask.mean()
            mask_min = mask.min()

        return {
            'ddpm_loss': ddpm_loss,
            'mag_loss': mag_loss,
            'chroma_loss': chroma_loss,
            'u_smooth_loss': u_smooth_loss,
            'dan_expr_loss': dan_expr_loss,
            'coarse_loss': coarse_loss,
            'lpips_loss': lpips_loss,
            'psnr': psnr,
            'dmag_mean': dmag_mean,
            'dmag_max': dmag_max,
            'disp_mean': disp_mean,
            'disp_max': disp_max,
            'rf_loss_ll': outputs['rf_loss_ll'].mean() if 'rf_loss_ll' in outputs else ddpm_loss,
            'rf_loss_hf': outputs['rf_loss_hf'].mean() if 'rf_loss_hf' in outputs else ddpm_loss,
            'mask_mean': mask_mean,
            'mask_min': mask_min,
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
    batch_size = 8          # Smaller default since DTCWT uses more memory
    lr = 4e-4
    num_epochs = 10
    image_size = 224
    labels = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"]
    num_emotions = len(labels)

    # Conditioning method
    use_film = True
    use_adagn = False

    # Loss weights
    lambda_ddpm = 1.0
    lambda_mag = 0.1         # Magnitude matching (reduced: was 0.5 when L1(dmag,0); now L1(pred,target) is larger scale)
    lambda_dan_expr = 0.5    # DAN emotion loss (reduced: was 1.6, prevent adversarial optimization)
    lambda_coarse = 2.0      # Coarse structure preservation (boosted: was 1.0)
    lambda_chroma = 2.0      # Chrominance preservation (Cb/Cr LL)
    lambda_lpips = 1.0       # LPIPS perceptual quality
    lambda_smooth = 0.5      # Displacement field smoothness (TV reg)

    # DTCWT model features (lightweight)
    features = [48, 96, 192, 384]

    print(f"Batch size: {batch_size}")
    print(f"Features: {features}")
    print(f"FiLM: {use_film}, AdaGN: {use_adagn}")

    # ============================
    # Directories
    # ============================
    exp_name = "DTCWT_V3_RF"
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
    diff_model = DirectionalWaveDiffusionModel(
        num_emotions=num_emotions,
        features=features,
        use_film=use_film,
        use_adagn=use_adagn
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in diff_model.parameters())
    trainable_params = sum(p.numel() for p in diff_model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"Trainable params: {trainable_params:,} ({trainable_params/1e6:.1f}M)")

    # DAN emotion model (frozen)
    emotion_model = Emotion_model()
    for param in emotion_model.model.parameters():
        param.requires_grad = False

    # Coarse structure loss (DTCWT-based)
    coarse_loss_fn = CoarseStructureLoss_DTCWT().to(device)

    # LPIPS perceptual loss (prevents adversarial optimization against DAN)
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    lpips_fn.eval()
    for param in lpips_fn.parameters():
        param.requires_grad = False

    # Adaptive loss weighting
    initial_weights = {'lambda_dan_expr': lambda_dan_expr}
    loss_weighter = AdaptiveLossWeighter(initial_weights, warmup_steps=5000)

    # Training wrapper: bundles model + all loss functions
    # DataParallel wraps THIS so each GPU computes its own losses
    model = TrainingWrapper(diff_model, emotion_model.model, lpips_fn, coarse_loss_fn).to(device)

    if num_gpus > 1:
        print(f"Using DataParallel with {num_gpus} GPUs")
        model = torch.nn.DataParallel(model)

    # base_model = raw diffusion model (for checkpoints, sampling, attribute access)
    base_model = model.module.model if num_gpus > 1 else diff_model

    # ============================
    # DTCWT Roundtrip Test
    # ============================
    dtcwt_wrapper = DTCWTWrapper().to(device)
    idtcwt_wrapper = IDTCWTWrapper().to(device)
    with torch.no_grad():
        x_test = next(iter(val_dataloader))[0].to(device)[:4]
        ll, mag, phase = dtcwt_wrapper(x_test)
        real = mag * torch.cos(phase)
        imag = mag * torch.sin(phase)
        x_rec = idtcwt_wrapper(ll, real, imag)
        mse = ((x_test - x_rec) ** 2).mean().item()
        print(f"DTCWT roundtrip MSE: {mse:.8f} {'✓' if mse < 1e-4 else '⚠ TOO HIGH!'}")
    del dtcwt_wrapper, idtcwt_wrapper, x_test, x_rec

    # ============================
    # Optimizer
    # ============================
    # Separate learning rate for MagnitudePhaseModulator (learn slower)
    mod_params = []
    mask_params = []
    other_params = []
    for name, param in diff_model.named_parameters():
        if 'mag_mod' in name:
            mod_params.append(param)
        elif 'mask_temperature' in name or 'mask_threshold' in name:
            mask_params.append(param)
        else:
            other_params.append(param)

    print(f"MagModulator params: {len(mod_params)}, Mask params: {len(mask_params)}, Other params: {len(other_params)}")

    optimizer = torch.optim.AdamW([
        {'params': other_params, 'lr': lr, 'weight_decay': 1e-4},
        {'params': mod_params, 'lr': lr * 0.5, 'weight_decay': 1e-6},  # 2x slower
        {'params': mask_params, 'lr': lr * 0.1, 'weight_decay': 0.0}   # 10x slower, no decay
    ], betas=(0.9, 0.999))

    # AMP mixed precision for VRAM savings (~50%)
    scaler = torch.amp.GradScaler('cuda')
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
                with torch.amp.autocast('cuda'):
                    outputs = model(img_real, expr_org, drop_prob=0.1)

                    # DataParallel gathers scalars as (num_gpus,) tensors → mean them
                    ddpm_loss = outputs['ddpm_loss'].mean()
                    mag_loss = outputs['mag_loss'].mean()
                    chroma_loss = outputs['chroma_loss'].mean()
                    u_smooth_loss = outputs['u_smooth_loss'].mean()
                    dan_expr_loss = outputs['dan_expr_loss'].mean()
                    coarse_loss = outputs['coarse_loss'].mean()
                    lpips_loss = outputs['lpips_loss'].mean()
                    psnr = outputs['psnr'].mean()

                    # ===== ADAPTIVE WEIGHTS =====
                    current_losses = {
                        'dan_expr': dan_expr_loss,
                        'ddpm': ddpm_loss
                    }
                    weight_updates = loss_weighter.update_weights(current_losses)
                    current_lambda_dan = weight_updates.get('lambda_dan_expr', lambda_dan_expr)

                    # ===== TOTAL LOSS =====
                    total_loss = (
                        lambda_ddpm * ddpm_loss +
                        lambda_mag * mag_loss +
                        lambda_chroma * chroma_loss +
                        lambda_smooth * u_smooth_loss +
                        current_lambda_dan * dan_expr_loss +
                        lambda_coarse * coarse_loss +
                        lambda_lpips * lpips_loss
                    )

                # ===== NaN GUARD =====
                if not torch.isfinite(total_loss):
                    nan_info = []
                    for name, val in [('DDPM', ddpm_loss), ('Mag', mag_loss), 
                                       ('Chroma', chroma_loss), ('Smooth', u_smooth_loss),
                                       ('DAN', dan_expr_loss), ('Coarse', coarse_loss),
                                       ('LPIPS', lpips_loss)]:
                        if not torch.isfinite(val):
                            nan_info.append(name)
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

                # ===== LOGGING =====
                epoch_bar.set_postfix({
                    "T": f"{total_loss.item():.3f}",
                    "RF": f"{ddpm_loss.item():.3f}",
                    "HF": f"{outputs['rf_loss_hf'].mean().item():.3f}",
                    "D": f"{dan_expr_loss.item():.3f}",
                    "LP": f"{lpips_loss.item():.3f}",
                    "P": f"{psnr.item():.1f}",
                })

                step += 1

                if i % log_every == 0:
                    writer.add_scalar("Loss/Total", total_loss.item(), step)
                    writer.add_scalar("Loss/Flow", ddpm_loss.item(), step)
                    writer.add_scalar("Loss/DAN", dan_expr_loss.item(), step)
                    writer.add_scalar("Loss/Mag_Preservation", mag_loss.item(), step)
                    writer.add_scalar("Loss/Chroma_Preservation", chroma_loss.item(), step)
                    writer.add_scalar("Loss/Coarse_Structure", coarse_loss.item(), step)
                    writer.add_scalar("Loss/LPIPS", lpips_loss.item(), step)
                    writer.add_scalar("Metric/PSNR", psnr.item(), step)
                    writer.add_scalar("Weight/Lambda_DAN", current_lambda_dan, step)
                    writer.add_scalar("Grad/Norm", grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm, step)

                    # Log DTCWT-specific metrics (pre-computed in wrapper)
                    writer.add_scalar("DTCWT/DeltaMag_mean", outputs['dmag_mean'].mean().item(), step)
                    writer.add_scalar("DTCWT/DeltaMag_max", outputs['dmag_max'].max().item(), step)
                    writer.add_scalar("APT/Displacement_mean", outputs['disp_mean'].mean().item(), step)
                    writer.add_scalar("APT/Displacement_max", outputs['disp_max'].max().item(), step)
                    writer.add_scalar("APT/U_smooth_loss", u_smooth_loss.item(), step)
                    writer.add_scalar("Loss/RF_LL", outputs['rf_loss_ll'].mean().item(), step)
                    writer.add_scalar("Loss/RF_HF", outputs['rf_loss_hf'].mean().item(), step)
                    writer.add_scalar("Mask/RetainMean", outputs['mask_mean'].mean().item(), step)
                    writer.add_scalar("Mask/RetainMin", outputs['mask_min'].min().item(), step)

                    # Log MagnitudeModulator direction attention
                    with torch.no_grad():
                        emotion_emb = base_model.unet.emotion_embedding(expr_org[:1])
                        dir_weights = base_model.mag_mod.dir_attn(emotion_emb)
                        for d in range(6):
                            writer.add_scalar(f"DTCWT/DirAttn_orient{d}", dir_weights[0, d].item(), step)

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
                                emotion_tensor = torch.full((1,), emotion_id, device=device)
                                generated = base_model.sample(
                                    sample, emotion_tensor, 
                                    num_steps=20, denoising_strength=1.0
                                )
                                gen_tensor = (generated[0].clamp(-1, 1) + 1) / 2
                                tensor_images.append(gen_tensor)
                        if tensor_images:
                            all_imgs = torch.stack(tensor_images, dim=0)
                            n_cols = num_emotions + 1
                            save_image(all_imgs, f"{out_path}/epoch{epoch}_mid_emotions.png", nrow=n_cols, normalize=False)
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
                        emotion_tensor = torch.full((1,), emotion_id, device=device)
                        generated = base_model.sample(
                            sample,
                            emotion_tensor,
                            num_steps=20,
                            denoising_strength=0.7
                        )
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
