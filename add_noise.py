from model import DDPMSampler
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt

generator = torch.Generator().manual_seed(0)
sampler = DDPMSampler(generator=generator)

img = Image.open("1.jpg").convert("RGB")
img_np = np.array(img)
img_tensor = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1)  # (C, H, W)
img_tensor = ((img_tensor / 255.0) * 2.0) - 1.0  # Chuẩn hóa [-1, 1]
img_tensor = img_tensor.unsqueeze(0)  # (1, C, H, W)

noise_levels = [0, 10, 50, 100, 250, 500, 750]
timesteps = torch.tensor(noise_levels, dtype=torch.long)

batch = img_tensor.repeat(len(noise_levels), 1, 2, 2)  # (B, C, H, W)

noised_imgs = sampler.add_noise(batch, timesteps)  # (B, C, H, W)

noised_imgs = (noised_imgs.clamp(-1, 1) + 1) / 2
noised_imgs = (noised_imgs * 255).type(torch.uint8).permute(0, 2, 3, 1)  # (B, H, W, C)

plt.figure(figsize=(20, 4))
for i, t in enumerate(noise_levels):
    plt.subplot(1, len(noise_levels), i + 1)
    plt.imshow(noised_imgs[i].cpu().numpy())
    plt.title(f"t = {t}")
    plt.axis("off")
plt.tight_layout()
plt.show()
