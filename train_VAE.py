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
from pprint import pprint
from icecream import ic
import time


def save_checkpoint(filepath, epoch, step, encoder, decoder, optimizer, loss):
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'loss': loss.item(),
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, os.path.join(filepath, "last_model.pt"))
    # Lưu best model nếu cần
    if not os.path.exists(os.path.join(filepath, "best_model.pt")) or loss.item() < checkpoint['loss']:
        torch.save(checkpoint, os.path.join(filepath, "best_model.pt"))

def load_checkpoint(filepath, encoder, decoder, optimizer, best_loss, device):
    last_path = os.path.join(filepath, "last_model.pt")
    best_path = os.path.join(filepath, "best_model.pt")

    if os.path.isfile(last_path):
        print(f"Loading checkpoint '{last_path}'")
        checkpoint = torch.load(last_path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from epoch {start_epoch}, step {checkpoint['step']}")
    else:
        print("No last checkpoint found. Starting from scratch.")
        start_epoch = 0

    # Cập nhật best_loss
    if os.path.isfile(best_path):
        best_checkpoint = torch.load(best_path, map_location=device)
        best_loss = best_checkpoint['loss']
    else:
        best_loss = float('inf')

    return start_epoch, best_loss


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
    start_time = time.time()
    max_duration = 11.5 * 60 * 60
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = ["angry", "disgust", "fear","happy", "neutral", "sad", "surprise"]

    log_dir = "VAE/runs/exp"
    model_path = "VAE/model"
    out_path = "VAE/out"

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
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
        os.makedirs(model_path, exist_ok=True)
    else:
        os.makedirs(model_path, exist_ok=True)

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
    encoder = VAE_Encoder().to(device)
    decoder = VAE_Decoder().to(device)

    # Optimizer
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)

    # Load the best model if it exists
    start_epoch, best_loss = load_checkpoint(model_path, encoder, decoder, optimizer, best_loss, device)

    x_fixed, c_org, _, _ = next(iter(val_dataloader))
    x_fixed = x_fixed.to(device)
    c_fixed_list = create_labels(c_org, c_dim)
    c_fixed_list = torch.stack(c_fixed_list).to(device)

    try:
        for epoch in range(start_epoch, num_epochs):
            for i, (img_real, expr_org, valence_org, arousal_org) in enumerate(train_dataloader):
                if time.time() - start_time > max_duration:
                    save_checkpoint(model_path, epoch, i, encoder, decoder, optimizer, loss)
                    return  # hoặc dùng break nếu bạn muốn thoát chỉ khỏi vòng lặp hiện tại

                img_real = img_real.to(device)
                noise = torch.randn(1, 4, 28, 28).to(torch.float32).to(device)
                latent, mean, log_variance = encoder(img_real, noise)
                recon_img = decoder(latent)
                # Loss = MSE + KL
                recon_loss = nn.MSELoss()(recon_img, img_real)
                kl_loss = compute_kl_loss(mean, log_variance)
                loss = recon_loss + 0.05 * kl_loss  # lambda = 0.001

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(f"[Epoch {epoch} Iter {i}] Recon: {recon_loss.item():.4f} | KL: {kl_loss.item():.4f}")
                # Log
                if i % 10 == 0:
                    writer.add_scalar("Loss/Reconstruction", recon_loss.item(), epoch * len(train_dataloader) + i)
                    writer.add_scalar("Loss/KL", kl_loss.item(), epoch * len(train_dataloader) + i)

                # # Lưu ảnh ví dụ
                # if i % 200 == 0:
                    with torch.no_grad():
                        vis = (recon_img.clamp(-1, 1) + 1) / 2
                        save_image(vis, f"{out_path}/epoch{epoch}_iter{i}.png", nrow=4)
                save_checkpoint(model_path, epoch, i, encoder, decoder, optimizer, loss)
    except KeyboardInterrupt:
        save_checkpoint(model_path, epoch, i, encoder, decoder, optimizer, loss)

if __name__ == "__main__":
    train()
