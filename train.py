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
from dataset import Affectnet

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
    batch_size = 4
    lr = 1e-4
    num_epochs = 100
    image_channels = 3
    c_dim = 11
    image_size = 224
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
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.5402, 0.4410, 0.3938], std=[0.2914, 0.2657, 0.2609]),#tìm mean std
    ])
    # Data loader
    # train_dataset = ImageFolder(root='/home/tam/Desktop/pythonProject1/archive/AffectNet/data', transform=transform)
    train_dataset = Affectnet(root="C:/Users/tam/Documents/data/Affectnet", is_train=True, transform=transform)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True,
        drop_last=True
    )

    val_dataset = Affectnet(root="C:/Users/tam/Documents/data/Affectnet", is_train=False, transform=transform)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True,
        drop_last=True
    )

    # Models
    # Load checkpoint VAE
    encoder = VAE_Encoder().to(device)
    decoder = VAE_Decoder().to(device)
    vae_ckpt_path = "VAE/model/best_model1.pt"
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
    start_epoch, best_loss = load_checkpoint(model_path, diffusion, diffusion_optimizer, best_loss, device)

    x_fixed, c_org, valence_org, arousal_org = next(iter(val_dataloader))
    x_fixed = x_fixed.to(device)
    c_fixed_list = create_labels(c_org, c_dim)
    c_fixed_list = torch.stack(c_fixed_list).to(device)

    try:
        alpha = 0.1  # hệ số loss phụ
        for epoch in range(start_epoch, num_epochs):
            for i, (img_real, expr_org, valence_org, arousal_org) in enumerate(train_dataloader):
                # if expr_org.max() >= 8 or expr_org.min() < 0:
                #     print(f"❌ Lỗi nhãn: max = {expr_org.max()}, min = {expr_org.min()}")
                #     # print(f"expr_pred shape: {expr_pred.shape}")
                #     continue

                rand_idx = torch.randperm(expr_org.size(0))
                label_trg = expr_org[rand_idx]
                valence_trg = valence_org[rand_idx]
                arousal_trg = arousal_org[rand_idx]

                img_real = img_real.to(device)
                label_trg = label_trg.to(device)
                valence_trg = valence_trg.to(device)
                arousal_trg = arousal_trg.to(device)

                # Encode ảnh → latent
                noise = torch.randn(1, 4, 28, 28).to(torch.float32).to(device)

                with torch.no_grad():
                    latent, _, _ = encoder(img_real, noise)

                B, C, H, W = latent.shape
                pad_h = (8 - H % 8) % 8
                pad_w = (8 - W % 8) % 8
                latent = F.pad(latent, (0, pad_w, 0, pad_h), mode='reflect')

                # Add noise
                timestep = torch.randint(0, sampler.num_train_timesteps, (img_real.size(0),), device=device).long()
                noisy_latent = sampler.add_noise(latent, timestep)

                pred_noise, expr_pred, val_pred, aro_pred = diffusion(
                    latent=noisy_latent,
                    expr_label=label_trg,
                    valence=valence_trg,
                    arousal=arousal_trg,
                    time=timestep.unsqueeze(-1).float()
                )

                # Loss chính
                mse_loss = nn.MSELoss()(pred_noise, torch.randn_like(pred_noise))

                # Loss phụ (dự đoán expr, valence, arousal)
                expr_loss = nn.CrossEntropyLoss()(expr_pred, label_trg)
                val_loss = nn.MSELoss()(val_pred, valence_trg)
                aro_loss = nn.MSELoss()(aro_pred, arousal_trg)

                # Tổng loss
                loss = mse_loss + alpha * (expr_loss + val_loss + aro_loss)

                # Backprop
                diffusion_optimizer.zero_grad()
                loss.backward()
                diffusion_optimizer.step()

                print(f"[Epoch {epoch} Iter {i}] Loss: {loss.item():.4f} | MSE: {mse_loss.item():.4f} | expr: {expr_loss.item():.4f} | val: {val_loss.item():.4f} | aro: {aro_loss.item():.4f}")
                if i % 10 == 0:
                    writer.add_scalar("Loss/Total", loss.item(), epoch * len(train_dataloader) + i)
                    writer.add_scalar("Loss/MSE", mse_loss.item(), epoch * len(train_dataloader) + i)

                # Save ảnh
                if i % 500 == 0:
                    print(500)
                    with torch.no_grad():
                        latent_fixed, _, _ = encoder(x_fixed, torch.randn(1, 4, 28, 28).to(torch.float32).to(device))
                        B, C, H, W = latent_fixed.shape
                        pad_h = (8 - H % 8) % 8
                        pad_w = (8 - W % 8) % 8
                        latent_fixed = F.pad(latent_fixed, (0, pad_w, 0, pad_h), mode='reflect')
                        timestep = torch.zeros(x_fixed.size(0), dtype=torch.long, device=device)
                        expr_sample = torch.ones_like(timestep) * labels.index("happy")
                        val_sample = torch.ones_like(timestep, dtype=torch.float32) * 0.9
                        aro_sample = torch.ones_like(timestep, dtype=torch.float32) * 0.8

                        expr_embed = diffusion.expr_embedding(expr_sample)
                        va_embed = diffusion.va_proj(torch.stack([val_sample, aro_sample], dim=1))
                        context = torch.cat([expr_embed, va_embed], dim=1)
                        context = diffusion.context_proj(context)
                        print(501)
                        sampler.set_inference_timesteps(50)
                        z = latent_fixed
                        for t in sampler.timesteps:
                            pred, _, _, _ = diffusion(z, expr_sample, val_sample, aro_sample, t.to(device).expand(x_fixed.size(0), 1).float())
                            z = sampler.step(t.item(), z, pred)
                        print(502)
                        img_gen = decoder(z)
                        print(503)
                        img_gen = (img_gen.clamp(-1, 1) + 1) / 2
                        print(504)
                        save_image(img_gen, f"{out_path}/epoch{epoch}_iter{i}.png", nrow=4)
            save_checkpoint(model_path, epoch, i, diffusion, diffusion_optimizer, loss)

    except KeyboardInterrupt:
        save_checkpoint(model_path, epoch, i, diffusion, diffusion_optimizer, loss)
        pass
if __name__ == '__main__':
    train()

