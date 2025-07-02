import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Generator
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize, ToTensor, Compose, Normalize
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import VAE_Encoder, VAE_Decoder, CLIP, Diffusion, DDPMSampler
from dataset import Affectnet, AffectnetPt
from icecream import ic
import time

def save_checkpoint(filepath, epoch, step, diffusion, optimizer, loss):
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'loss': loss.item(),
        'diffusion_state_dict': diffusion.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, os.path.join(filepath, "last_model.pt"))
    if not os.path.exists(os.path.join(filepath, "best_model.pt")) or loss.item() < checkpoint['loss']:
        torch.save(checkpoint, os.path.join(filepath, "best_model.pt"))

def load_checkpoint(filepath, diffusion, optimizer, best_loss, device):
    last_path = os.path.join(filepath, "last_model.pt")
    best_path = os.path.join(filepath, "best_model.pt")
    if os.path.isfile(last_path):
        print(f"Loading checkpoint '{last_path}'")
        checkpoint = torch.load(last_path, map_location=device)
        diffusion.load_state_dict(checkpoint['diffusion_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Loaded epoch {start_epoch}, step {checkpoint['step']}")
    else:
        print("No checkpoint found. Start from scratch.")
        start_epoch = 0
    if os.path.isfile(best_path):
        best_loss = torch.load(best_path, map_location=device)['loss']
    else:
        best_loss = float('inf')
    return start_epoch, best_loss

def gradient_penalty(device, y, x):
    weight = torch.ones(y.size()).to(device)
    dydx = torch.autograd.grad(outputs=y,
                               inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm-1)**2)

def label2onehot(labels, dim):
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out

def create_labels(c_org, c_dim=5):
    c_trg_list = []
    for i in range(c_dim):
        c_trg = label2onehot(torch.ones(c_org.size(0)) * i, c_dim)
        c_trg_list.append(c_trg)
    return c_trg_list

def compute_kl_loss(mean, logvar):
    return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / mean.size(0)

def cout(a):
    print("****************************")
    print(a)
    print(type(a))
    print("****************************")

def train():
    batch_size = 8
    lr = 1e-4
    num_epochs = 100
    image_channels = 3
    c_dim = 11
    image_size = 224
    accumulation_steps = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = ["angry", "disgust", "fear","happy", "neutral", "sad", "surprise"]

    log_dir = "SD/runs/exp"
    model_path = "SD/model"
    out_path = "SD/out"

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)
    else:
        os.makedirs(log_dir, exist_ok=True)
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
        os.makedirs(out_path, exist_ok=True)
    else:
        os.makedirs(out_path, exist_ok=True)

    print(device)
    best_loss = float('inf')
    writer = SummaryWriter(log_dir)


    transform = Compose([
        # Resize((image_size, image_size)),
        # ToTensor(),
        Normalize(mean=[0.5402, 0.4410, 0.3938], std=[0.2914, 0.2657, 0.2609]),#tìm mean std
    ])
    # Data loader
    print("train_dataloader")
    # train_dataset = ImageFolder(root='/home/tam/Desktop/pythonProject1/archive/AffectNet/data', transform=transform)
    # train_dataset = Affectnet(root="C:/Users/tam/Documents/data/Affectnet", is_train=True, transform=transform)
    train_dataset = AffectnetPt(root="C:/Users/tam/Documents/data/Affectnet", is_train=False, transform=transform)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True,
        drop_last=True
    )
    print("val_dataloader")
    # val_dataset = Affectnet(root="C:/Users/tam/Documents/data/Affectnet", is_train=False, transform=transform)
    val_dataset = AffectnetPt(root="C:/Users/tam/Documents/data/Affectnet", is_train=False, transform=transform)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True,
        drop_last=True
    )
    print("Số lượng ảnh trong val_dataset:", len(val_dataset))

    # Models
    # Load checkpoint VAE
    print("models")
    encoder = VAE_Encoder().to(device)
    decoder = VAE_Decoder().to(device)
    vae_ckpt_path = r"C:\Users\tam\Downloads\VAE\model\best_model.pt"
    vae_ckpt = torch.load(vae_ckpt_path, map_location=device)
    encoder.load_state_dict(vae_ckpt['encoder_state_dict'])
    decoder.load_state_dict(vae_ckpt['decoder_state_dict'])
    encoder.eval()
    decoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    for param in decoder.parameters():
        param.requires_grad = False

    diffusion = Diffusion().to(device)
    sampler = DDPMSampler(torch.Generator(device = device).manual_seed(0))#.to(device)

    # Optimizer
    diffusion_optimizer = torch.optim.Adam(diffusion.parameters(), lr=lr, betas=(0.5, 0.999))

    # Load the best model if it exists
    print("load model")
    start_epoch, best_loss = load_checkpoint(model_path, diffusion, diffusion_optimizer, best_loss, device)

    x_fixed, c_org, valence_org, arousal_org = next(iter(val_dataloader))
    x_fixed = x_fixed.to(device)
    c_fixed_list = create_labels(c_org, c_dim)
    c_fixed_list = torch.stack(c_fixed_list).to(device)

    try:
        alpha = 0.1  # hệ số loss phụ
        print("train")
        for epoch in range(start_epoch, num_epochs):
            for i, (img_real, expr_org, valence_org, arousal_org) in enumerate(train_dataloader):
                # Trộn nhãn (random target)
                rand_idx = torch.randperm(expr_org.size(0))
                label_trg = expr_org[rand_idx]
                valence_trg = valence_org[rand_idx]
                arousal_trg = arousal_org[rand_idx]

                actual_batch_size = img_real.size(0)  # = 8
                ic(img_real.size(0))
                mini_batch_size = actual_batch_size // accumulation_steps  # = 2

                loss_accum = 0.0
                valid_steps = 0  # Đếm số lần backward thực sự
                last_loss = None
                last_mse_loss = None

                for acc_i in range(accumulation_steps):
                    start_idx = acc_i * mini_batch_size
                    end_idx = (acc_i + 1) * mini_batch_size
                    ic(start_idx, end_idx, acc_i, mini_batch_size)
                    mini_img = img_real[start_idx:end_idx].to(device)
                    mini_label = label_trg[start_idx:end_idx].to(device)
                    mini_val = valence_trg[start_idx:end_idx].to(device)
                    mini_aro = arousal_trg[start_idx:end_idx].to(device)

                    if mini_img.size(0) == 0:
                        continue

                    noise = torch.randn(mini_img.size(0), 4, 28, 28).to(torch.float32).to(device)
                    with torch.no_grad():
                        latent, _, _ = encoder(mini_img, noise)

                    latent = latent.to(device)
                    B, C, H, W = latent.shape
                    pad_h = (8 - H % 8) % 8
                    pad_w = (8 - W % 8) % 8
                    latent = F.pad(latent, (0, pad_w, 0, pad_h), mode='reflect')

                    timestep = torch.randint(0, sampler.num_train_timesteps, (mini_img.size(0),), device=device).long()
                    noisy_latent = sampler.add_noise(latent, timestep)

                    pred_noise, expr_pred, val_pred, aro_pred = diffusion(
                        latent=noisy_latent,
                        expr_label=mini_label,
                        valence=mini_val,
                        arousal=mini_aro,
                        time=timestep.unsqueeze(-1).float()
                    )

                    mse_loss = nn.MSELoss()(pred_noise, torch.randn_like(pred_noise))
                    expr_loss = nn.CrossEntropyLoss()(expr_pred, mini_label)
                    val_loss = nn.MSELoss()(val_pred, mini_val)
                    aro_loss = nn.MSELoss()(aro_pred, mini_aro)

                    loss = mse_loss + alpha * (expr_loss + val_loss + aro_loss)
                    loss_accum += loss.item()

                    last_loss = loss  # lưu lại loss cuối để log
                    last_mse_loss = mse_loss

                    loss = loss / accumulation_steps
                    loss.backward()
                    valid_steps += 1
                    ic(valid_steps)
                if valid_steps > 0:
                    diffusion_optimizer.step()
                    diffusion_optimizer.zero_grad()

                    print(f"[Epoch {epoch} Iter {i}] Avg Loss: {loss_accum:.4f}")

                    if i % 10 == 0:
                        writer.add_scalar("Loss/Total", last_loss.item(), epoch * len(train_dataloader) + i)
                        writer.add_scalar("Loss/MSE", last_mse_loss.item(), epoch * len(train_dataloader) + i)

                # Save ảnh
                if i % 500 == 0:
                    print("➡️ [SAVE IMAGE] Bắt đầu sinh ảnh val...")
                    with torch.no_grad():
                        all_imgs = []
                        num_samples = x_fixed.size(0)
                        mini_bs = 2  # tránh OOM

                        for idx in range(0, num_samples, mini_bs):
                            print(f"🔁 Sinh batch ảnh val nhỏ: từ {idx} đến {idx + mini_bs}")
                            x_part = x_fixed[idx:idx + mini_bs]
                            noise = torch.randn(x_part.size(0), 4, 28, 28).to(torch.float32).to(device)
                            latent_fixed, _, _ = encoder(x_part, noise)

                            B, C, H, W = latent_fixed.shape
                            pad_h = (8 - H % 8) % 8
                            pad_w = (8 - W % 8) % 8
                            latent_fixed = F.pad(latent_fixed, (0, pad_w, 0, pad_h), mode='reflect')

                            print(f"✅ [Encode OK] latent shape: {latent_fixed.shape}")

                            timestep = torch.zeros(x_part.size(0), dtype=torch.long, device=device)
                            expr_sample = torch.ones_like(timestep) * labels.index("happy")
                            val_sample = torch.ones_like(timestep, dtype=torch.float32) * 0.9
                            aro_sample = torch.ones_like(timestep, dtype=torch.float32) * 0.8

                            expr_embed = diffusion.expr_embedding(expr_sample)
                            va_embed = diffusion.va_proj(torch.stack([val_sample, aro_sample], dim=1))
                            context = torch.cat([expr_embed, va_embed], dim=1)
                            context = diffusion.context_proj(context)

                            sampler.set_inference_timesteps(50)
                            z = latent_fixed
                            print("🚀 [Start Sampling]")
                            for t in sampler.timesteps:
                                pred, _, _, _ = diffusion(z, expr_sample, val_sample, aro_sample,
                                                          t.to(device).expand(x_part.size(0), 1).float())
                                z = sampler.step(t.item(), z, pred)
                            print("✅ [Sampling done]")

                            img_gen = decoder(z)
                            print("🖼️ [Decoded images]")
                            img_gen = (img_gen.clamp(-1, 1) + 1) / 2
                            all_imgs.append(img_gen)

                        all_imgs = torch.cat(all_imgs, dim=0)
                        print("📦 [Tổng hợp ảnh xong] -> Lưu ảnh")
                        save_image(all_imgs, f"{out_path}/epoch{epoch}_iter{i}.png", nrow=4)
                        print(f"✅ [Ảnh đã lưu]: {out_path}/epoch{epoch}_iter{i}.png")

            if last_loss is not None:
                save_checkpoint(model_path, epoch, i, diffusion, diffusion_optimizer, last_loss)

    except KeyboardInterrupt:
        if last_loss is not None:
            save_checkpoint(model_path, epoch, i, diffusion, diffusion_optimizer, last_loss)
        pass
if __name__ == '__main__':
    train()

