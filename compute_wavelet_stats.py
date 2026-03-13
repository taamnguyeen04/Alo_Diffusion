"""
Compute PIXEL-LEVEL per-channel mean and std for 12-channel Haar DWT.
Uses Welford's algorithm on EVERY PIXEL, not per-image averages.

Output: wavelet_stats.pt containing {'ch_mean': (12,), 'ch_std': (12,)}
"""

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, ToTensor, Compose, Normalize
from tqdm import tqdm

from model_dtcwt_v2 import DTCWTWrapper, DWT
from dataset import Affectnet

NUM_SAMPLES = 5000
BATCH_SIZE = 32
IMAGE_SIZE = 224


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = Compose([
        Resize((IMAGE_SIZE, IMAGE_SIZE)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = Affectnet(is_train=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=4, drop_last=False)

    dtcwt = DTCWTWrapper().to(device)
    haar_dwt = DWT().to(device)

    # Accumulate sum and sum_sq for PIXEL-LEVEL stats
    ch_sum = torch.zeros(12, device=device, dtype=torch.float64)
    ch_sum_sq = torch.zeros(12, device=device, dtype=torch.float64)
    total_pixels = 0  # Total number of pixels per channel

    total_batches = min(NUM_SAMPLES // BATCH_SIZE + 1, len(dataloader))
    processed = 0

    with torch.no_grad():
        for batch_idx, (images, _, _, _) in enumerate(tqdm(dataloader, total=total_batches,
                                                           desc="Computing wavelet stats")):
            if processed >= NUM_SAMPLES:
                break

            images = images.to(device)
            B = images.shape[0]

            # DTCWT → Haar DWT
            ll_full, _, _ = dtcwt(images)
            wavelet_12ch = haar_dwt(ll_full)  # (B, 12, H, W)

            # Flatten spatial dims: (B, 12, H*W)
            flat = wavelet_12ch.view(B, 12, -1).to(torch.float64)
            num_pixels_per_sample = flat.shape[2]  # H*W

            # Accumulate sum and sum_sq per channel
            ch_sum += flat.sum(dim=(0, 2))       # sum over batch and spatial
            ch_sum_sq += (flat ** 2).sum(dim=(0, 2))
            total_pixels += B * num_pixels_per_sample

            processed += B

    # Compute mean and std
    ch_mean = (ch_sum / total_pixels).float()
    ch_var = (ch_sum_sq / total_pixels - ch_mean.double() ** 2).float()
    ch_std = torch.sqrt(ch_var.clamp(min=1e-8))

    print(f"\nComputed from {processed} images ({total_pixels:,} pixels per channel):")
    ch_names = ['LL_Y', 'LL_Cb', 'LL_Cr', 'LH_Y', 'LH_Cb', 'LH_Cr',
                'HL_Y', 'HL_Cb', 'HL_Cr', 'HH_Y', 'HH_Cb', 'HH_Cr']
    print(f"{'Channel':<10} {'Mean':>12} {'Std':>12}")
    print("-" * 36)
    for i in range(12):
        print(f"{ch_names[i]:<10} {ch_mean[i].item():>12.6f} {ch_std[i].item():>12.6f}")

    print(f"\nLL mean range:  [{ch_mean[:3].min():.4f}, {ch_mean[:3].max():.4f}]")
    print(f"HF mean range:  [{ch_mean[3:].min():.6f}, {ch_mean[3:].max():.6f}]")
    print(f"LL std range:   [{ch_std[:3].min():.4f}, {ch_std[:3].max():.4f}]")
    print(f"HF std range:   [{ch_std[3:].min():.6f}, {ch_std[3:].max():.6f}]")
    print(f"\nLL/HF std ratio: {ch_std[:3].mean() / ch_std[3:].mean():.1f}x")

    stats = {
        'ch_mean': ch_mean.cpu(),
        'ch_std': ch_std.cpu(),
        'num_samples': processed,
        'total_pixels': total_pixels,
    }
    save_path = "wavelet_stats.pt"
    torch.save(stats, save_path)
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
