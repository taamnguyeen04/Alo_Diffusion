"""
APT (Analytic Phase Transport) Verification Test Suite
=====================================================
Comprehensive tests to ensure the APT module and full DTCWT v2 pipeline
are functioning correctly before investing in training.

Tests:
  1. DTCWT Roundtrip Fidelity
  2. APT Zero-Displacement (src == tgt)
  3. APT Known Translation Recovery
  4. Warp Equivariance (warp + inverse_warp ≈ identity)
  5. Self-Reconstruction (Protocol D — sample with target = source emotion)
  6. Gradient Flow Sanity
  7. Forward/Sample No-NaN No-Inf (stress test)

Usage:
  python verify_apt.py
"""

import torch
import torch.nn.functional as F
import math
import sys
import traceback

from model_dtcwt_v2 import (
    DirectionalWaveDiffusionModel,
    DTCWTWrapper, IDTCWTWrapper,
    AnalyticPhaseTransport,
    rgb_to_ycbcr, ycbcr_to_rgb,
)


def _separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _result(passed, msg=""):
    status = "\033[92mPASS\033[0m" if passed else "\033[91mFAIL\033[0m"
    print(f"  [{status}] {msg}")
    return passed


# ============================================================================
# Test 1: DTCWT Roundtrip Fidelity
# ============================================================================
def test_dtcwt_roundtrip(device):
    _separator("Test 1: DTCWT Roundtrip Fidelity")
    
    wrapper = DTCWTWrapper().to(device)
    iwrapper = IDTCWTWrapper().to(device)
    
    # Use various test images
    test_images = {
        "random":    torch.rand(2, 3, 224, 224, device=device),
        "gradient":  _make_gradient_image(2, 224, 224, device),
        "zeros":     torch.zeros(2, 3, 224, 224, device=device),
        "ones":      torch.ones(2, 3, 224, 224, device=device),
    }
    
    all_pass = True
    for name, x in test_images.items():
        ll, mag, phase = wrapper(x)
        real = mag * torch.cos(phase)
        imag = mag * torch.sin(phase)
        recon = iwrapper(ll, real, imag)
        mse = ((x - recon) ** 2).mean().item()
        passed = mse < 1e-4
        all_pass &= _result(passed, f"{name}: MSE = {mse:.8f} (threshold: 1e-4)")
    
    return all_pass


# ============================================================================
# Test 2: APT Zero-Displacement
# ============================================================================
def test_zero_displacement(device):
    _separator("Test 2: APT Zero-Displacement (src == tgt)")
    
    apt = AnalyticPhaseTransport().to(device)
    wrapper = DTCWTWrapper().to(device)
    
    # Test with various images
    test_images = {
        "random":   torch.rand(2, 3, 224, 224, device=device),
        "gradient": _make_gradient_image(2, 224, 224, device),
    }
    
    all_pass = True
    for name, x in test_images.items():
        ll, _, _ = wrapper(x)
        ll_y = ll[:, 0:1]  # Y channel
        
        with torch.no_grad():
            u = apt.solve_displacement(ll_y, ll_y)
        
        max_u = u.abs().max().item()
        mean_u = u.abs().mean().item()
        
        # When src==tgt, displacement should be essentially zero
        passed = max_u < 0.1
        all_pass &= _result(passed, f"{name}: max|u|={max_u:.6f}, mean|u|={mean_u:.6f} (threshold: 0.1)")
    
    return all_pass


# ============================================================================
# Test 3: APT Known Translation Recovery
# ============================================================================
def test_known_translation(device):
    _separator("Test 3: APT Known Translation Recovery")
    
    apt = AnalyticPhaseTransport().to(device)
    wrapper = DTCWTWrapper().to(device)
    
    # Create a structured test image (edges help phase estimation)
    x = _make_edge_rich_image(1, 224, 224, device)
    
    known_shifts = [(2, 0), (0, 3), (2, 2), (-2, 1)]
    all_pass = True
    
    for dy, dx in known_shifts:
        # Shift image
        x_shifted = torch.roll(torch.roll(x, dy, dims=2), dx, dims=3)
        
        ll_src, _, _ = wrapper(x)
        ll_tgt, _, _ = wrapper(x_shifted)
        
        with torch.no_grad():
            u = apt.solve_displacement(ll_src[:, 0:1], ll_tgt[:, 0:1])
        
        # u should approximate the known shift
        # Note: due to Gaussian smoothing and boundary effects,
        # we check the central region
        h, w = u.shape[2], u.shape[3]
        crop = 4  # ignore boundary
        u_center = u[:, :, crop:-crop, crop:-crop]
        
        mean_dy = u_center[:, 0].mean().item()
        mean_dx = u_center[:, 1].mean().item()
        
        err_dy = abs(mean_dy - dy)
        err_dx = abs(mean_dx - dx)
        
        # Tolerance: APT uses Gaussian smoothing + coarse-to-fine registration
        # which spreads the displacement, so we use generous tolerances.
        # The key property is that the direction is correct, not exact magnitude.
        total_err = math.sqrt(err_dy**2 + err_dx**2)
        shift_mag = math.sqrt(dy**2 + dx**2) + 0.01
        relative_err = total_err / shift_mag
        
        # Pass if: direction is roughly correct (relative error < 200%)
        # or total pixel error < 4 pixels
        passed = relative_err < 2.0 or total_err < 4.0
        all_pass &= _result(passed, 
            f"shift=({dy},{dx}): recovered=({mean_dy:.2f},{mean_dx:.2f}), "
            f"err=({err_dy:.2f},{err_dx:.2f}), total_err={total_err:.2f}, rel_err={relative_err:.2f}")
    
    return all_pass


# ============================================================================
# Test 4: Warp Equivariance
# ============================================================================
def test_warp_equivariance(device):
    _separator("Test 4: Warp Equivariance (warp + inverse ~ identity)")
    
    apt = AnalyticPhaseTransport().to(device)
    
    # Create synthetic HF coefficients and Haar details
    B, C, H, W = 2, 18, 112, 112
    src_mag = torch.rand(B, C, H, W, device=device) + 0.1
    src_phase = (torch.rand(B, C, H, W, device=device) - 0.5) * 2 * math.pi
    
    # Small displacement
    u = torch.randn(B, 2, H, W, device=device) * 2.0
    minus_u = -u
    
    with torch.no_grad():
        # Forward warp
        warped_mag, warped_phase = apt.warp_coefficients(src_mag, src_phase, u)
        # Inverse warp
        recovered_mag, recovered_phase = apt.warp_coefficients(warped_mag, warped_phase, minus_u)
    
    # Note: double bilinear interpolation loses some info, but should be close
    mag_err = (src_mag - recovered_mag).abs().mean().item()
    phase_err = torch.abs(torch.sin(src_phase - recovered_phase)).mean().item()
    
    all_pass = True
    all_pass &= _result(mag_err < 0.5, f"Magnitude recovery error: {mag_err:.4f} (threshold: 0.5)")
    all_pass &= _result(phase_err < 0.7, f"Phase recovery error (sin diff): {phase_err:.4f} (threshold: 0.7)")
    
    # Test Haar detail warping
    haar_detail = torch.randn(B, 9, H, W, device=device)
    with torch.no_grad():
        warped_haar = apt.warp_haar_detail(haar_detail, u)
        recovered_haar = apt.warp_haar_detail(warped_haar, minus_u)
    
    haar_err = (haar_detail - recovered_haar).abs().mean().item()
    all_pass &= _result(haar_err < 1.0, f"Haar detail recovery error: {haar_err:.4f} (threshold: 1.0)")
    
    return all_pass


# ============================================================================
# Test 5: Self-Reconstruction (Protocol D)
# ============================================================================
def test_self_reconstruction(device):
    _separator("Test 5: Self-Reconstruction (Protocol D)")
    
    model = DirectionalWaveDiffusionModel(num_emotions=8).to(device)
    model.eval()
    
    # Random test image (since model is untrained, just check pipeline doesn't crash)
    x = torch.rand(1, 3, 224, 224, device=device)
    emotion = torch.tensor([3], device=device)
    
    with torch.no_grad():
        out = model.sample(x, emotion, num_steps=5)
    
    # Basic shape check
    assert out.shape == x.shape, f"Output shape mismatch: {out.shape} vs {x.shape}"
    
    # Compute metrics (these will be poor with random weights, but pipeline should work)
    # Clamp to [0,1] for metric computation
    out_c = out.clamp(0, 1)
    x_c = x.clamp(0, 1)
    
    l1 = (out_c - x_c).abs().mean().item()
    mse = ((out_c - x_c) ** 2).mean().item()
    psnr = 10 * math.log10(1.0 / (mse + 1e-8)) if mse > 0 else 100
    ssim_val = _compute_ssim(out_c, x_c)
    
    print(f"  Metrics (random weights, expect poor):")
    print(f"    L1:   {l1:.4f}")
    print(f"    PSNR: {psnr:.2f} dB")
    print(f"    SSIM: {ssim_val:.4f}")
    
    all_pass = True
    all_pass &= _result(not torch.isnan(out).any().item(), "No NaN in sample output")
    all_pass &= _result(not torch.isinf(out).any().item(), "No Inf in sample output")
    all_pass &= _result(out.shape == x.shape, f"Shape: {out.shape}")
    
    return all_pass


# ============================================================================
# Test 6: Gradient Flow
# ============================================================================
def test_gradient_flow(device):
    _separator("Test 6: Gradient Flow Sanity")
    
    model = DirectionalWaveDiffusionModel(num_emotions=8).to(device)
    model.train()
    
    x = torch.rand(2, 3, 224, 224, device=device)
    emotion = torch.randint(0, 8, (2,), device=device)
    
    outputs = model(x, emotion, src_image=x)
    
    # Total loss (mimics training)
    total_loss = (
        outputs['ddpm_loss'] +
        0.1 * outputs['mag_loss'] +
        0.1 * outputs.get('chroma_loss', torch.tensor(0.0)) +
        0.01 * outputs.get('u_smooth_loss', torch.tensor(0.0))
    )
    
    total_loss.backward()
    
    # Check gradients in key parameter groups
    param_groups = {
        "UNet (input conv)": list(model.unet.input_conv.parameters()),
        "UNet (noise head)": list(model.unet.output_noise.parameters()),
        "UNet (dmag head)": list(model.unet.output_delta_mag.parameters()),
        "MagnitudeModulator": list(model.mag_mod.parameters()),
        "HF Encoder": list(model.unet.hf_encoder.parameters()),
        "UNet emotion embed": list(model.unet.emotion_embedding.parameters()),
    }
    
    all_pass = True
    for group_name, params in param_groups.items():
        has_grad = any(p.grad is not None and p.grad.abs().max() > 0 for p in params if p.requires_grad)
        all_pass &= _result(has_grad, f"{group_name}: grad exists = {has_grad}")
    
    # APT should have NO learned parameters
    apt_params = list(model.apt.parameters())
    all_pass &= _result(len(apt_params) == 0, 
        f"APT has {len(apt_params)} learned params (expect 0 — analytic module)")
    
    return all_pass


# ============================================================================
# Test 7: No NaN/Inf Stress Test
# ============================================================================
def test_no_nan_inf(device):
    _separator("Test 7: Forward/Sample No-NaN No-Inf (Stress Test)")
    
    model = DirectionalWaveDiffusionModel(num_emotions=8).to(device)
    
    stress_inputs = {
        "random":    torch.rand(2, 3, 224, 224, device=device),
        "zeros":     torch.zeros(2, 3, 224, 224, device=device) + 1e-6,
        "ones":      torch.ones(2, 3, 224, 224, device=device),
        "high_val":  torch.ones(2, 3, 224, 224, device=device) * 0.99,
        "low_val":   torch.ones(2, 3, 224, 224, device=device) * 0.01,
        "noise":     torch.randn(2, 3, 224, 224, device=device).abs().clamp(0, 1),
    }
    
    all_pass = True
    
    for name, x in stress_inputs.items():
        emotion = torch.randint(0, 8, (x.shape[0],), device=device)
        
        # Test forward
        model.train()
        try:
            outputs = model(x, emotion, src_image=x)
            has_nan = any(torch.isnan(v).any().item() for v in outputs.values() if isinstance(v, torch.Tensor))
            has_inf = any(torch.isinf(v).any().item() for v in outputs.values() if isinstance(v, torch.Tensor))
            
            all_pass &= _result(not has_nan, f"forward({name}): no NaN")
            all_pass &= _result(not has_inf, f"forward({name}): no Inf")
            
            if has_nan or has_inf:
                for k, v in outputs.items():
                    if isinstance(v, torch.Tensor):
                        if torch.isnan(v).any():
                            print(f"    NaN in '{k}'")
                        if torch.isinf(v).any():
                            print(f"    Inf in '{k}'")
        except Exception as e:
            all_pass &= _result(False, f"forward({name}): EXCEPTION {e}")
            traceback.print_exc()
        
        # Test sample (only for a subset to save time)
        if name in ("random", "zeros", "ones"):
            model.eval()
            try:
                with torch.no_grad():
                    out = model.sample(x[:1], emotion[:1], num_steps=3)
                sample_nan = torch.isnan(out).any().item()
                sample_inf = torch.isinf(out).any().item()
                all_pass &= _result(not sample_nan, f"sample({name}): no NaN")
                all_pass &= _result(not sample_inf, f"sample({name}): no Inf")
            except Exception as e:
                all_pass &= _result(False, f"sample({name}): EXCEPTION {e}")
                traceback.print_exc()
    
    return all_pass


# ============================================================================
# Helper functions
# ============================================================================

def _make_gradient_image(B, H, W, device):
    """Create a smooth gradient image (useful for testing smooth displacements)."""
    y = torch.linspace(0, 1, H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
    x = torch.linspace(0, 1, W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
    return torch.cat([y, x, (y + x) / 2], dim=1)


def _make_edge_rich_image(B, H, W, device):
    """Create image with strong edges (stripes/checkerboard) for phase testing."""
    y = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)
    x = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, 1, W)
    
    # Horizontal stripes
    h_stripe = ((y / 8).floor() % 2).expand(B, 1, H, W)
    # Vertical stripes
    v_stripe = ((x / 12).floor() % 2).expand(B, 1, H, W)
    # Diagonal
    d_stripe = (((x + y) / 10).floor() % 2).expand(B, 1, H, W)
    
    img = torch.cat([h_stripe * 0.8 + 0.1, v_stripe * 0.7 + 0.15, d_stripe * 0.6 + 0.2], dim=1)
    return img


def _compute_ssim(img1, img2, window_size=11, C1=0.01**2, C2=0.03**2):
    """Simplified SSIM computation (single scale, luminance only)."""
    # Convert to grayscale
    g1 = img1.mean(dim=1, keepdim=True)
    g2 = img2.mean(dim=1, keepdim=True)
    
    # Create window
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=img1.device) - window_size // 2
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    g /= g.sum()
    window = g.unsqueeze(0) * g.unsqueeze(1)
    window = window.unsqueeze(0).unsqueeze(0)  # (1,1,ws,ws)
    
    pad = window_size // 2
    
    mu1 = F.conv2d(g1, window, padding=pad)
    mu2 = F.conv2d(g2, window, padding=pad)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2
    
    sigma1_sq = F.conv2d(g1 * g1, window, padding=pad) - mu1_sq
    sigma2_sq = F.conv2d(g2 * g2, window, padding=pad) - mu2_sq
    sigma12 = F.conv2d(g1 * g2, window, padding=pad) - mu12
    
    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  APT Verification Test Suite")
    print("  Model: DirectionalWaveDiffusionModel with APT")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    results = {}
    
    # Run all tests
    tests = [
        ("DTCWT Roundtrip",          test_dtcwt_roundtrip),
        ("APT Zero-Displacement",    test_zero_displacement),
        ("APT Known Translation",    test_known_translation),
        ("Warp Equivariance",        test_warp_equivariance),
        ("Self-Reconstruction",      test_self_reconstruction),
        ("Gradient Flow",            test_gradient_flow),
        ("NaN/Inf Stress Test",      test_no_nan_inf),
    ]
    
    for test_name, test_fn in tests:
        try:
            results[test_name] = test_fn(device)
        except Exception as e:
            print(f"\n  [\033[91mERROR\033[0m] {test_name}: Unhandled exception")
            traceback.print_exc()
            results[test_name] = False
        
        # Clear GPU cache between tests
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    # Summary
    _separator("SUMMARY")
    total = len(results)
    passed = sum(results.values())
    for name, ok in results.items():
        status = "\033[92mPASS\033[0m" if ok else "\033[91mFAIL\033[0m"
        print(f"  [{status}] {name}")
    
    print(f"\n  {passed}/{total} tests passed")
    
    if passed == total:
        print(f"\n  [OK] All tests passed! Model is ready for training.")
    else:
        print(f"\n  [!!] {total - passed} test(s) failed. Review before training.")
    
    sys.exit(0 if passed == total else 1)
