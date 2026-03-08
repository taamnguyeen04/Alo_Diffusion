"""
Debug Ghosting Artifacts — Comprehensive Diagnostic Script
===========================================================
Loads best_model.pt and systematically tests each pipeline component
to identify the root cause of ghosting/overlapping in generated images.

Tests:
  1. Checkpoint Integrity: Load and inspect model weights
  2. DTCWT Roundtrip: Verify decomposition/reconstruction fidelity
  3. LL Diffusion Only: Generate images using ONLY denoised LL (no APT)
  4. APT Displacement Analysis: Measure displacement field statistics
  5. APT Warp Quality: Visual comparison of warped vs unwarped HF
  6. Magnitude Modulation Analysis: Check delta_mag scale/distribution
  7. Full Pipeline vs Ablated: Compare outputs with/without each component
  8. Denoising Strength Sweep: Check how strength affects ghosting

Usage:
  python debug_ghosting.py
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import Resize, ToTensor, Compose, Normalize
from torchvision.utils import save_image

from model_dtcwt_v2 import (
    DirectionalWaveDiffusionModel, DTCWTWrapper, IDTCWTWrapper,
    AnalyticPhaseTransport, rgb_to_ycbcr, ycbcr_to_rgb
)


# ============================================================================
# Config
# ============================================================================
CHECKPOINT_PATH = "DTCWT_apt/model/best_model.pt"
OUTPUT_DIR = "debug_ghosting_output"
IMAGE_SIZE = 224
NUM_EMOTIONS = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"]


def setup():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def load_model():
    """Load model from checkpoint."""
    print("\n" + "=" * 60)
    print("  Loading Model from Checkpoint")
    print("=" * 60)

    model = DirectionalWaveDiffusionModel(
        num_emotions=NUM_EMOTIONS,
        features=[48, 96, 192, 384],
        use_film=True,
        use_adagn=False
    ).to(DEVICE)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"  Checkpoint epoch: {checkpoint.get('epoch', '?')}")
    print(f"  Checkpoint step: {checkpoint.get('step', '?')}")
    print(f"  Checkpoint loss: {checkpoint.get('loss', '?')}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params:,} ({total_params / 1e6:.1f}M)")

    return model


def load_test_images():
    """Load test images from validation set or create synthetic ones."""
    print("\n  Loading test images...")

    # Try to load real images from dataset
    try:
        from dataset import Affectnet
        transform = Compose([
            Resize((IMAGE_SIZE, IMAGE_SIZE)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = Affectnet(is_train=False, transform=transform)
        # Get a few samples
        images = []
        emotions = []
        for i in range(min(4, len(dataset))):
            img, expr, _, _ = dataset[i * 100]  # spread out samples
            images.append(img)
            emotions.append(expr)
        images = torch.stack(images).to(DEVICE)
        emotions = torch.stack(emotions).to(DEVICE)
        print(f"  Loaded {images.shape[0]} images from Affectnet validation set")
        return images, emotions
    except Exception as e:
        print(f"  Could not load Affectnet: {e}")
        print("  Using synthetic test images")
        images = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE) * 0.5
        emotions = torch.tensor([0, 1, 2, 3], device=DEVICE)
        return images, emotions


def tensor_to_image(t):
    """Convert normalized tensor [-1,1] or ImageNet-normalized to [0,1] for saving."""
    # Undo ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406], device=t.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=t.device).view(3, 1, 1)
    t = t * std + mean
    return t.clamp(0, 1)


# ============================================================================
# Test 1: Checkpoint Integrity
# ============================================================================
def test_checkpoint_integrity(model):
    print("\n" + "=" * 60)
    print("  Test 1: Checkpoint Integrity & Weight Statistics")
    print("=" * 60)

    key_modules = {
        "UNet input_conv": model.unet.input_conv,
        "UNet noise_head": model.unet.output_noise,
        "UNet dmag_head": model.unet.output_delta_mag,
        "MagnitudeModulator dir_attn": model.mag_mod.dir_attn,
        "MagnitudeModulator delta_pred": model.mag_mod.delta_pred,
        "MagnitudeModulator mag_scale": None,  # scalar
        "HF Encoder init_proj": model.unet.hf_encoder.init_proj,
    }

    for name, module in key_modules.items():
        if name == "MagnitudeModulator mag_scale":
            val = model.mag_mod.mag_scale.item()
            print(f"  {name}: value = {val:.6f} (init=0.1)")
            continue

        params = list(module.parameters())
        for i, p in enumerate(params):
            mean = p.data.mean().item()
            std = p.data.std().item()
            abs_max = p.data.abs().max().item()
            has_nan = torch.isnan(p.data).any().item()
            has_inf = torch.isinf(p.data).any().item()
            print(f"  {name}[{i}]: shape={tuple(p.shape)}, "
                  f"mean={mean:.6f}, std={std:.6f}, max={abs_max:.6f}, "
                  f"NaN={has_nan}, Inf={has_inf}")

    # Check emotion embedding learned differences
    print("\n  Emotion Embedding Analysis:")
    emb_weight = model.unet.emotion_embedding.weight.data
    for i in range(NUM_EMOTIONS):
        norm = emb_weight[i].norm().item()
        print(f"    {LABELS[i]:10s}: L2 norm = {norm:.4f}")

    # Pairwise cosine similarity
    print("\n  Emotion Embedding Pairwise Cosine Similarity:")
    norms = emb_weight.norm(dim=1, keepdim=True)
    cos_sim = (emb_weight @ emb_weight.T) / (norms * norms.T + 1e-8)
    for i in range(NUM_EMOTIONS):
        sims = [f"{cos_sim[i, j].item():.2f}" for j in range(NUM_EMOTIONS)]
        print(f"    {LABELS[i]:10s}: {' '.join(sims)}")


# ============================================================================
# Test 2: DTCWT Roundtrip
# ============================================================================
def test_dtcwt_roundtrip(images):
    print("\n" + "=" * 60)
    print("  Test 2: DTCWT Roundtrip Fidelity")
    print("=" * 60)

    dtcwt = DTCWTWrapper().to(DEVICE)
    idtcwt = IDTCWTWrapper().to(DEVICE)

    with torch.no_grad():
        ll, mag, phase = dtcwt(images)
        real = mag * torch.cos(phase)
        imag = mag * torch.sin(phase)
        recon = idtcwt(ll, real, imag)

    mse = ((images - recon) ** 2).mean().item()
    psnr = 10 * math.log10(4 / (mse + 1e-8))

    print(f"  MSE:  {mse:.8f}")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  LL shape:    {ll.shape}")
    print(f"  Mag shape:   {mag.shape}")
    print(f"  Phase shape: {phase.shape}")
    print(f"  Mag range:   [{mag.min():.4f}, {mag.max():.4f}]")
    print(f"  Phase range: [{phase.min():.4f}, {phase.max():.4f}]")

    # Save comparison
    orig = tensor_to_image(images[0])
    rec = tensor_to_image(recon[0])
    diff = (orig - rec).abs() * 10  # amplified diff
    save_image(torch.stack([orig, rec, diff]), f"{OUTPUT_DIR}/02_dtcwt_roundtrip.png", nrow=3)
    print(f"  Saved: {OUTPUT_DIR}/02_dtcwt_roundtrip.png")

    return mse < 1e-4


# ============================================================================
# Test 3: LL-Only Reconstruction (No APT, No Δmag)
# ============================================================================
def test_ll_only(model, images, emotions):
    print("\n" + "=" * 60)
    print("  Test 3: LL-Only Reconstruction (bypass APT)")
    print("=" * 60)
    print("  If ghosting appears here → problem is in LL diffusion")
    print("  If clean here → problem is in APT/HF pipeline")

    with torch.no_grad():
        # Decompose source
        src_ll_full, src_mag, src_phase = model.dtcwt(images[:1])
        src_ll, ll_hf_skip = model.ll_downsample(src_ll_full)

        results = []
        for strength in [0.1, 0.15, 0.2, 0.3, 0.5]:
            # DDIM denoise LL
            start_t = max(1, int(strength * model.num_timesteps))
            timesteps = torch.linspace(start_t - 1, 0,
                                       min(50, start_t),
                                       dtype=torch.long, device=DEVICE)

            noise = torch.randn_like(src_ll)
            alpha_start = model.sqrt_alphas_cumprod[start_t]
            sigma_start = model.sqrt_one_minus_alphas_cumprod[start_t]
            x = alpha_start * src_ll + sigma_start * noise

            emotion_id = torch.tensor([1], device=DEVICE)  # Happy
            hf_condition = torch.cat([src_mag, src_phase], dim=1)
            ll_size = (src_ll.shape[2], src_ll.shape[3])
            hf_features = model.unet.hf_encoder(hf_condition, ll_size)

            for i, t in enumerate(timesteps):
                t_tensor = torch.full((1,), t.item(), device=DEVICE, dtype=torch.long)
                unet_input = model._inject_emotion(x, emotion_id)
                noise_pred, _ = model.unet(unet_input, t_tensor, emotion_id,
                                          hf_features, images[:1])

                alpha_t = model.alphas_cumprod[t.item()]
                alpha_prev = (model.alphas_cumprod[timesteps[i+1].item()]
                              if i < len(timesteps) - 1
                              else torch.tensor(1.0, device=DEVICE))

                pred_x0 = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
                pred_x0 = torch.clamp(pred_x0, -3, 3)

                if i < len(timesteps) - 1:
                    x = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1 - alpha_prev) * noise_pred
                else:
                    x = pred_x0

            # Reconstruct using ORIGINAL source HF (no APT warp)
            ll_full = model.ll_upsample(x, ll_hf_skip)
            src_real = src_mag * torch.cos(src_phase)
            src_imag = src_mag * torch.sin(src_phase)
            output_no_apt = model.idtcwt(ll_full, src_real, src_imag)

            mse = ((images[:1] - output_no_apt) ** 2).mean().item()
            print(f"  strength={strength:.2f}: MSE={mse:.4f}, "
                  f"output range=[{output_no_apt.min():.3f}, {output_no_apt.max():.3f}]")

            results.append(tensor_to_image(output_no_apt[0]))

        # Save grid
        orig = tensor_to_image(images[0])
        all_imgs = [orig] + results
        save_image(torch.stack(all_imgs), f"{OUTPUT_DIR}/03_ll_only_no_apt.png", nrow=6)
        print(f"  Saved: {OUTPUT_DIR}/03_ll_only_no_apt.png")
        print("         [Original, s=0.1, s=0.15, s=0.2, s=0.3, s=0.5]")


# ============================================================================
# Test 4: APT Displacement Analysis
# ============================================================================
def test_apt_displacement(model, images, emotions):
    print("\n" + "=" * 60)
    print("  Test 4: APT Displacement Field Analysis")
    print("=" * 60)

    with torch.no_grad():
        src_ll_full, src_mag, src_phase = model.dtcwt(images[:1])
        src_ll, ll_hf_skip = model.ll_downsample(src_ll_full)

        for strength in [0.15, 0.3, 0.5]:
            for emo_id in [0, 1, 2]:
                emotion_id = torch.tensor([emo_id], device=DEVICE)

                # Denoise
                start_t = max(1, int(strength * model.num_timesteps))
                timesteps = torch.linspace(start_t - 1, 0,
                                           min(50, start_t),
                                           dtype=torch.long, device=DEVICE)

                noise = torch.randn_like(src_ll)
                x = (model.sqrt_alphas_cumprod[start_t] * src_ll +
                     model.sqrt_one_minus_alphas_cumprod[start_t] * noise)

                hf_condition = torch.cat([src_mag, src_phase], dim=1)
                ll_size = (src_ll.shape[2], src_ll.shape[3])
                hf_features = model.unet.hf_encoder(hf_condition, ll_size)

                for i, t in enumerate(timesteps):
                    t_tensor = torch.full((1,), t.item(), device=DEVICE, dtype=torch.long)
                    unet_input = model._inject_emotion(x, emotion_id)
                    noise_pred, _ = model.unet(unet_input, t_tensor, emotion_id,
                                              hf_features, images[:1])

                    alpha_t = model.alphas_cumprod[t.item()]
                    alpha_prev = (model.alphas_cumprod[timesteps[i+1].item()]
                                  if i < len(timesteps) - 1
                                  else torch.tensor(1.0, device=DEVICE))

                    pred_x0 = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
                    pred_x0 = torch.clamp(pred_x0, -3, 3)

                    if i < len(timesteps) - 1:
                        x = (torch.sqrt(alpha_prev) * pred_x0 +
                             torch.sqrt(1 - alpha_prev) * noise_pred)
                    else:
                        x = pred_x0

                # APT displacement
                denoised_ll_full = model.ll_upsample(x, ll_hf_skip)
                src_ll_Y = src_ll_full[:, 0:1].float()
                pred_ll_Y = denoised_ll_full[:, 0:1].float()

                displacement = model.apt.solve_displacement(src_ll_Y, pred_ll_Y)

                dy = displacement[:, 0]
                dx = displacement[:, 1]

                print(f"  s={strength:.2f}, {LABELS[emo_id]:10s}: "
                      f"dy=[{dy.min():.2f}, {dy.max():.2f}], mean={dy.abs().mean():.3f} | "
                      f"dx=[{dx.min():.2f}, {dx.max():.2f}], mean={dx.abs().mean():.3f} | "
                      f"|u|_max={displacement.abs().max():.2f}")

        # Save displacement field visualization for strength=0.3, Happy
        print("\n  Generating displacement field visualization...")
        emotion_id = torch.tensor([1], device=DEVICE)
        start_t = max(1, int(0.3 * model.num_timesteps))
        timesteps = torch.linspace(start_t - 1, 0, min(50, start_t),
                                   dtype=torch.long, device=DEVICE)

        noise = torch.randn_like(src_ll)
        x = (model.sqrt_alphas_cumprod[start_t] * src_ll +
             model.sqrt_one_minus_alphas_cumprod[start_t] * noise)

        hf_condition = torch.cat([src_mag, src_phase], dim=1)
        ll_size = (src_ll.shape[2], src_ll.shape[3])
        hf_features = model.unet.hf_encoder(hf_condition, ll_size)

        for i, t in enumerate(timesteps):
            t_tensor = torch.full((1,), t.item(), device=DEVICE, dtype=torch.long)
            unet_input = model._inject_emotion(x, emotion_id)
            noise_pred, _ = model.unet(unet_input, t_tensor, emotion_id,
                                      hf_features, images[:1])
            alpha_t = model.alphas_cumprod[t.item()]
            alpha_prev = (model.alphas_cumprod[timesteps[i+1].item()]
                          if i < len(timesteps) - 1
                          else torch.tensor(1.0, device=DEVICE))
            pred_x0 = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            pred_x0 = torch.clamp(pred_x0, -3, 3)
            if i < len(timesteps) - 1:
                x = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1 - alpha_prev) * noise_pred
            else:
                x = pred_x0

        denoised_ll_full = model.ll_upsample(x, ll_hf_skip)
        displacement = model.apt.solve_displacement(
            src_ll_full[:, 0:1].float(), denoised_ll_full[:, 0:1].float()
        )

        # Visualize displacement as color map
        dy = displacement[0, 0].cpu().numpy()
        dx = displacement[0, 1].cpu().numpy()
        mag_u = np.sqrt(dy**2 + dx**2)

        # Normalize for visualization
        dy_vis = torch.from_numpy((dy - dy.min()) / (dy.max() - dy.min() + 1e-8)).unsqueeze(0)
        dx_vis = torch.from_numpy((dx - dx.min()) / (dx.max() - dx.min() + 1e-8)).unsqueeze(0)
        mag_vis = torch.from_numpy((mag_u - mag_u.min()) / (mag_u.max() - mag_u.min() + 1e-8)).unsqueeze(0)

        disp_rgb = torch.stack([dy_vis, dx_vis, mag_vis], dim=0).squeeze(1)  # (3, H, W)
        save_image(disp_rgb, f"{OUTPUT_DIR}/04_displacement_field.png")
        print(f"  Saved: {OUTPUT_DIR}/04_displacement_field.png (R=dy, G=dx, B=magnitude)")


# ============================================================================
# Test 5: Δmag Analysis
# ============================================================================
def test_delta_mag(model, images, emotions):
    print("\n" + "=" * 60)
    print("  Test 5: Delta Magnitude Analysis")
    print("=" * 60)

    with torch.no_grad():
        src_ll_full, src_mag, src_phase = model.dtcwt(images[:1])
        src_ll, ll_hf_skip = model.ll_downsample(src_ll_full)

        hf_condition = torch.cat([src_mag, src_phase], dim=1)
        ll_size = (src_ll.shape[2], src_ll.shape[3])
        hf_features = model.unet.hf_encoder(hf_condition, ll_size)

        for emo_id in range(NUM_EMOTIONS):
            emotion_id = torch.tensor([emo_id], device=DEVICE)

            # Single-step UNet prediction (t=0, clean input → direct delta_mag)
            t_tensor = torch.full((1,), 0, device=DEVICE, dtype=torch.long)
            unet_input = model._inject_emotion(src_ll, emotion_id)
            _, delta_mag_raw = model.unet(unet_input, t_tensor, emotion_id,
                                         hf_features, images[:1])

            # Apply same processing as in forward()
            emotion_emb = model.unet.emotion_embedding(emotion_id)
            mod_dmag = model.mag_mod(emotion_emb)
            delta_mag = delta_mag_raw + mod_dmag
            delta_mag = 0.1 * torch.tanh(delta_mag)
            delta_mag[:, 6:, :, :] = delta_mag[:, 6:, :, :] * 0.1

            # Stats per orientation group
            y_dmag = delta_mag[0, :6].abs()   # Y orientations
            cb_dmag = delta_mag[0, 6:12].abs()  # Cb
            cr_dmag = delta_mag[0, 12:18].abs()  # Cr

            print(f"  {LABELS[emo_id]:10s}: "
                  f"Y_dmag={y_dmag.mean():.5f} (max={y_dmag.max():.5f}), "
                  f"Cb_dmag={cb_dmag.mean():.6f}, Cr_dmag={cr_dmag.mean():.6f}")

        # Direction attention analysis
        print("\n  MagnitudeModulator Direction Attention per Emotion:")
        for emo_id in range(NUM_EMOTIONS):
            emotion_id = torch.tensor([emo_id], device=DEVICE)
            emotion_emb = model.unet.emotion_embedding(emotion_id)
            dir_w = model.mag_mod.dir_attn(emotion_emb)
            weights = [f"{dir_w[0, d].item():.3f}" for d in range(6)]
            print(f"    {LABELS[emo_id]:10s}: {' '.join(weights)}  "
                  f"(orientations: 15° 45° 75° 105° 135° 165°)")


# ============================================================================
# Test 6: Full Pipeline vs Ablated Comparisons
# ============================================================================
def test_full_vs_ablated(model, images, emotions):
    print("\n" + "=" * 60)
    print("  Test 6: Full Pipeline vs Ablated Comparisons")
    print("=" * 60)

    strength = 0.3
    emotion_id = torch.tensor([1], device=DEVICE)  # Happy

    with torch.no_grad():
        # === Full pipeline (normal sample) ===
        full_output = model.sample(images[:1], emotion_id, num_steps=50,
                                   denoising_strength=strength)

        # === LL-only (no APT, original HF) ===
        src_ll_full, src_mag, src_phase = model.dtcwt(images[:1])
        src_ll, ll_hf_skip = model.ll_downsample(src_ll_full)

        start_t = max(1, int(strength * model.num_timesteps))
        timesteps = torch.linspace(start_t - 1, 0, min(50, start_t),
                                   dtype=torch.long, device=DEVICE)

        noise = torch.randn_like(src_ll)
        x = (model.sqrt_alphas_cumprod[start_t] * src_ll +
             model.sqrt_one_minus_alphas_cumprod[start_t] * noise)

        hf_condition = torch.cat([src_mag, src_phase], dim=1)
        ll_size = (src_ll.shape[2], src_ll.shape[3])
        hf_features = model.unet.hf_encoder(hf_condition, ll_size)
        emotion_emb = model.unet.emotion_embedding(emotion_id)
        final_dmag = None

        for i, t in enumerate(timesteps):
            t_tensor = torch.full((1,), t.item(), device=DEVICE, dtype=torch.long)
            unet_input = model._inject_emotion(x, emotion_id)
            noise_pred, delta_mag = model.unet(unet_input, t_tensor, emotion_id,
                                              hf_features, images[:1])
            mod_dm = model.mag_mod(emotion_emb)
            delta_mag = delta_mag + mod_dm
            delta_mag = 0.1 * torch.tanh(delta_mag)
            delta_mag[:, 6:, :, :] = delta_mag[:, 6:, :, :] * 0.1
            final_dmag = delta_mag

            alpha_t = model.alphas_cumprod[t.item()]
            alpha_prev = (model.alphas_cumprod[timesteps[i+1].item()]
                          if i < len(timesteps) - 1
                          else torch.tensor(1.0, device=DEVICE))
            pred_x0 = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            pred_x0 = torch.clamp(pred_x0, -3, 3)
            if i < len(timesteps) - 1:
                x = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1 - alpha_prev) * noise_pred
            else:
                x = pred_x0

        denoised_ll = x  # denoised LL at H/2

        # --- Variant A: LL-only, original HF (no APT, no dmag) ---
        ll_A = model.ll_upsample(denoised_ll, ll_hf_skip)
        src_real = src_mag * torch.cos(src_phase)
        src_imag = src_mag * torch.sin(src_phase)
        output_A = model.idtcwt(ll_A, src_real, src_imag)

        # --- Variant B: LL + dmag only (no APT warp) ---
        new_mag_B = src_mag * (1 + final_dmag)
        new_real_B = new_mag_B * torch.cos(src_phase)
        new_imag_B = new_mag_B * torch.sin(src_phase)
        ll_B = model.ll_upsample(denoised_ll, ll_hf_skip)
        output_B = model.idtcwt(ll_B, new_real_B, new_imag_B)

        # --- Variant C: LL + APT warp only (no dmag) ---
        denoised_ll_full_C = model.ll_upsample(denoised_ll, ll_hf_skip)
        src_ll_Y = src_ll_full[:, 0:1].float()
        pred_ll_Y_C = denoised_ll_full_C[:, 0:1].float()
        displacement_C = model.apt.solve_displacement(src_ll_Y, pred_ll_Y_C)

        warped_mag_C, warped_phase_C = model.apt.warp_coefficients(
            src_mag, src_phase, displacement_C)
        warped_hf_skip_C = model.apt.warp_haar_detail(ll_hf_skip, displacement_C)

        new_real_C = warped_mag_C * torch.cos(warped_phase_C)
        new_imag_C = warped_mag_C * torch.sin(warped_phase_C)
        ll_C = model.ll_upsample(denoised_ll, warped_hf_skip_C)
        output_C = model.idtcwt(ll_C, new_real_C, new_imag_C)

        # Compute metrics
        orig = images[:1]
        for name, output in [("A: LL only", output_A),
                              ("B: LL+dmag", output_B),
                              ("C: LL+APT", output_C),
                              ("D: Full", full_output)]:
            mse = ((orig - output) ** 2).mean().item()
            psnr = 10 * math.log10(4 / (mse + 1e-8))
            print(f"  {name}: MSE={mse:.4f}, PSNR={psnr:.2f} dB, "
                  f"range=[{output.min():.3f}, {output.max():.3f}]")

        # Save comparison grid
        grid_imgs = [
            tensor_to_image(orig[0]),        # Original
            tensor_to_image(output_A[0]),     # LL only
            tensor_to_image(output_B[0]),     # LL + dmag
            tensor_to_image(output_C[0]),     # LL + APT
            tensor_to_image(full_output[0]),  # Full pipeline
        ]

        save_image(torch.stack(grid_imgs),
                   f"{OUTPUT_DIR}/06_ablation_comparison.png", nrow=5)
        print(f"\n  Saved: {OUTPUT_DIR}/06_ablation_comparison.png")
        print("         [Original, LL-only, LL+dmag, LL+APT, Full]")
        print("  → If ghosting appears in LL+APT (col 4) but not LL-only (col 2): APT causes ghosting")
        print("  → If ghosting appears in LL-only too: problem is in LL diffusion")


# ============================================================================
# Test 7: Denoising Strength Sweep
# ============================================================================
def test_strength_sweep(model, images):
    print("\n" + "=" * 60)
    print("  Test 7: Denoising Strength Sweep")
    print("=" * 60)

    emotion_id = torch.tensor([1], device=DEVICE)  # Happy
    strengths = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7]

    grid_imgs = [tensor_to_image(images[0])]  # Original first

    with torch.no_grad():
        for strength in strengths:
            output = model.sample(images[:1], emotion_id, num_steps=50,
                                  denoising_strength=strength)
            mse = ((images[:1] - output) ** 2).mean().item()
            print(f"  strength={strength:.2f}: MSE={mse:.4f}, "
                  f"range=[{output.min():.3f}, {output.max():.3f}]")
            grid_imgs.append(tensor_to_image(output[0]))

    save_image(torch.stack(grid_imgs),
               f"{OUTPUT_DIR}/07_strength_sweep.png", nrow=len(grid_imgs))
    print(f"\n  Saved: {OUTPUT_DIR}/07_strength_sweep.png")
    print(f"         [Original, {', '.join(f's={s}' for s in strengths)}]")
    print("  → Check at which strength ghosting appears")


# ============================================================================
# Test 8: Multi-Emotion Grid (same as training validation)
# ============================================================================
def test_emotion_grid(model, images):
    print("\n" + "=" * 60)
    print("  Test 8: Multi-Emotion Grid (replicates validation)")
    print("=" * 60)

    n_samples = min(2, images.shape[0])
    grid_imgs = []

    with torch.no_grad():
        for idx in range(n_samples):
            sample = images[idx:idx+1]
            grid_imgs.append(tensor_to_image(sample[0]))

            for emo_id in range(NUM_EMOTIONS):
                emotion_id = torch.tensor([emo_id], device=DEVICE)
                output = model.sample(sample, emotion_id, num_steps=50,
                                      denoising_strength=0.3)
                grid_imgs.append(tensor_to_image(output[0]))

    save_image(torch.stack(grid_imgs),
               f"{OUTPUT_DIR}/08_emotion_grid.png",
               nrow=NUM_EMOTIONS + 1)
    print(f"  Saved: {OUTPUT_DIR}/08_emotion_grid.png")
    print(f"         Columns: [Original, {', '.join(LABELS)}]")


# ============================================================================
# Test 9: Consecutive sampling stability (same input, same noise seed)
# ============================================================================
def test_sampling_stability(model, images):
    print("\n" + "=" * 60)
    print("  Test 9: Sampling Stability (deterministic noise)")
    print("=" * 60)

    emotion_id = torch.tensor([1], device=DEVICE)
    grid_imgs = [tensor_to_image(images[0])]

    with torch.no_grad():
        for run in range(4):
            torch.manual_seed(42)  # Same seed each time
            output = model.sample(images[:1], emotion_id, num_steps=50,
                                  denoising_strength=0.3)
            grid_imgs.append(tensor_to_image(output[0]))

            if run > 0:
                diff = (grid_imgs[-1] - grid_imgs[1]).abs().max().item()
                print(f"  Run {run+1} vs Run 1: max pixel diff = {diff:.6f}")

    save_image(torch.stack(grid_imgs),
               f"{OUTPUT_DIR}/09_sampling_stability.png", nrow=5)
    print(f"  Saved: {OUTPUT_DIR}/09_sampling_stability.png")
    print("         [Original, Run1, Run2, Run3, Run4]")
    print("  → All runs should look identical (deterministic)")


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    setup()
    model = load_model()
    images, emotions = load_test_images()

    print("\n" + "#" * 60)
    print("  RUNNING ALL DIAGNOSTIC TESTS")
    print("#" * 60)

    test_checkpoint_integrity(model)
    test_dtcwt_roundtrip(images)
    test_ll_only(model, images, emotions)
    test_apt_displacement(model, images, emotions)
    test_delta_mag(model, images, emotions)
    test_full_vs_ablated(model, images, emotions)
    test_strength_sweep(model, images)
    test_emotion_grid(model, images)
    test_sampling_stability(model, images)

    print("\n" + "=" * 60)
    print("  ALL TESTS COMPLETE")
    print("=" * 60)
    print(f"\n  Output images saved to: {OUTPUT_DIR}/")
    print(f"  Key images to check:")
    print(f"    03_ll_only_no_apt.png  — If clean: APT is the problem")
    print(f"    06_ablation_comparison.png — Compare each component")
    print(f"    07_strength_sweep.png  — Find ghosting threshold")
    print(f"    08_emotion_grid.png    — Full emotion comparison")
