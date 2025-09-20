import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.transforms import Resize, ToTensor, Compose, Normalize
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import WaveletDiffusionModel, DWT, IWT
from dataset import Affectnet
from tqdm import tqdm
import lpips
from torchvision.models import resnet50


class ArcFaceIdentityLoss(nn.Module):
    """ArcFace-based identity preservation loss"""
    def __init__(self, device):
        super().__init__()
        self.device = device
        # In practice, you would load pre-trained ArcFace model
        # For simplicity, using cosine similarity on features
        self.feature_extractor = resnet50(pretrained=True)
        self.feature_extractor.fc = nn.Identity()
        self.feature_extractor.eval()

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, img1, img2):
        """
        Compute identity loss between two images
        Args:
            img1, img2: (B, 3, H, W) images in range [-1, 1]
        """
        # Normalize to [0, 1] for feature extraction
        img1_norm = (img1 + 1) / 2
        img2_norm = (img2 + 1) / 2

        # Extract features
        feat1 = self.feature_extractor(img1_norm)
        feat2 = self.feature_extractor(img2_norm)

        # Normalize features
        feat1 = F.normalize(feat1, p=2, dim=1)
        feat2 = F.normalize(feat2, p=2, dim=1)

        # Cosine similarity
        cosine_sim = F.cosine_similarity(feat1, feat2, dim=1)

        # Identity loss = 1 - cosine_similarity
        return 1 - cosine_sim.mean()


def save_checkpoint(filepath, epoch, step, model, optimizer, loss):
    print(f"💾 Đang lưu checkpoint: epoch {epoch}, step {step}, loss {loss.item():.4f}")

    checkpoint = {
        'epoch': epoch,
        'step': step,
        'loss': loss.item(),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    # Lưu last_model.pt
    last_path = os.path.join(filepath, "last_model.pt")
    torch.save(checkpoint, last_path)

    # Lưu best_model.pt nếu tốt hơn
    best_path = os.path.join(filepath, "best_model.pt")

    if not os.path.exists(best_path):
        torch.save(checkpoint, best_path)
        print("🏆 Đã lưu best_model.pt (lần đầu)")
    else:
        try:
            best_loss = torch.load(best_path, map_location='cpu')['loss']
            if loss.item() < best_loss:
                torch.save(checkpoint, best_path)
                print(f"🏆 Đã cập nhật best_model.pt: {best_loss:.4f} → {loss.item():.4f}")
        except Exception as e:
            print(f"⚠️ Lỗi khi đọc best_model.pt: {e} → lưu đè")
            torch.save(checkpoint, best_path)


def load_checkpoint(filepath, model, optimizer, best_loss, device):
    last_path = os.path.join(filepath, "last_model.pt")
    best_path = os.path.join(filepath, "best_model.pt")

    start_epoch = 0
    loaded = False

    if os.path.isfile(last_path):
        try:
            checkpoint = torch.load(last_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            loaded = True
            print(f"✅ Loaded từ last_model.pt epoch {start_epoch}")
        except Exception as e:
            print(f"⚠️ Lỗi khi load last_model.pt: {e}")

    if not loaded and os.path.isfile(best_path):
        try:
            checkpoint = torch.load(best_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"✅ Loaded từ best_model.pt epoch {start_epoch}")
            loaded = True
        except Exception as e:
            print(f"⚠️ Lỗi khi load best_model.pt: {e}")

    if not loaded:
        print("🚀 Không tìm thấy hoặc không load được checkpoint, bắt đầu từ đầu.")
        start_epoch = 0

    if os.path.isfile(best_path):
        try:
            best_loss = torch.load(best_path, map_location=device)['loss']
        except:
            best_loss = float('inf')
    else:
        best_loss = float('inf')

    return start_epoch, best_loss


def train():
    # Setup
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    batch_size = 8
    lr = 1e-4
    num_epochs = 100
    image_size = 224
    labels = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]

    print(f"📦 Batch size: {batch_size}")
    print(f"Device: {device}")

    # Loss weights
    lambda_ddpm = 1.0
    lambda_wav_ll = 0.1
    lambda_wav_hi = 0.2
    lambda_aux_expr = 0.02
    lambda_aux_va = 0.01
    lambda_id = 0.5
    lambda_lpips = 0.05

    # Directories
    log_dir = "WaveletDiffusion/runs/exp"
    model_path = "WaveletDiffusion/model"
    out_path = "WaveletDiffusion/out"

    for dir_path in [log_dir, model_path, out_path]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=True)

    writer = SummaryWriter(log_dir)
    best_loss = float('inf')

    # Data transforms
    transform = Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
    ])

    # Datasets
    print("📚 Loading datasets...")
    train_dataset = Affectnet(root="C:/Users/tam/Documents/data/FEG", is_train=True, transform=transform)
    val_dataset = Affectnet(root="C:/Users/tam/Documents/data/FEG", is_train=False, transform=transform)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        drop_last=True
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        drop_last=True
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Models
    print("🏗️ Building models...")

    # Main wavelet diffusion model
    model = WaveletDiffusionModel(num_emotions=len(labels)).to(device)

    # Auxiliary models
    identity_loss_fn = ArcFaceIdentityLoss(device).to(device)
    lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)
    for param in lpips_loss_fn.parameters():
        param.requires_grad = False

    # DWT/IWT transforms
    dwt = DWT().to(device)
    iwt = IWT().to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4
    )

    # Load checkpoint
    print("📁 Loading checkpoint...")
    start_epoch, best_loss = load_checkpoint(model_path, model, optimizer, best_loss, device)

    # Fixed validation samples
    x_fixed, expr_fixed, _, _ = next(iter(val_dataloader))
    x_fixed = x_fixed.to(device)
    expr_fixed = expr_fixed.to(device)

    try:
        print("🚀 Starting training...")
        for epoch in range(start_epoch, num_epochs):
            model.train()

            epoch_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                           desc=f"Epoch {epoch}/{num_epochs}")

            for i, (img_real, expr_org, valence_org, arousal_org) in epoch_bar:
                img_real = img_real.to(device)
                expr_org = expr_org.to(device)
                valence_org = valence_org.to(device)
                arousal_org = arousal_org.to(device)

                # Create target emotion (random shuffle for data augmentation)
                rand_idx = torch.randperm(expr_org.size(0))
                expr_trg = expr_org[rand_idx]
                valence_trg = valence_org[rand_idx]
                arousal_trg = arousal_org[rand_idx]

                # ===== FORWARD PASS =====

                # 1. DDPM Loss (main loss) - with auxiliary predictions
                x_wavelet = dwt(img_real)

                t = torch.randint(0, model.num_timesteps, (img_real.shape[0],), device=device)
                x_noisy, noise = model.forward_process(x_wavelet, t)

                # Get main prediction and auxiliary predictions
                noise_pred, expr_pred, va_pred = model.unet(
                    x_noisy, t, expr_trg, img_real, return_aux=True
                )

                # Main DDPM loss
                ddpm_loss = F.l1_loss(noise_pred, noise)

                # 2. Auxiliary losses from built-in heads
                aux_expr_loss = F.cross_entropy(expr_pred, expr_trg)
                aux_va_loss = F.mse_loss(va_pred, torch.stack([valence_trg, arousal_trg], dim=1))

                # 3. Generate samples for additional losses (less frequent to save computation)
                if i % 10 == 0:  # Only every 10 iterations
                    with torch.no_grad():
                        generated_img = model.sample(img_real, expr_trg, num_steps=50)

                    # 4. Wavelet Reconstruction Consistency Loss
                    img_wavelet_full = dwt(img_real)
                    gen_wavelet = dwt(generated_img)

                    # Split into subbands
                    C = img_real.shape[1]  # 3 for RGB
                    img_ll = img_wavelet_full[:, :C, :, :]      # Low-freq
                    img_hi = img_wavelet_full[:, C:, :, :]      # High-freq (LH, HL, HH)
                    gen_ll = gen_wavelet[:, :C, :, :]
                    gen_hi = gen_wavelet[:, C:, :, :]

                    wav_loss_ll = F.l1_loss(gen_ll, img_ll)
                    wav_loss_hi = F.l1_loss(gen_hi, img_hi)
                    wav_loss = lambda_wav_ll * wav_loss_ll + lambda_wav_hi * wav_loss_hi

                    # 5. Identity Preservation Loss
                    id_loss = identity_loss_fn(img_real, generated_img)

                    # 6. Perceptual Loss (LPIPS)
                    lpips_loss = lpips_loss_fn(img_real, generated_img).mean()

                    # ===== TOTAL LOSS =====
                    total_loss = (
                        lambda_ddpm * ddpm_loss +
                        lambda_aux_expr * aux_expr_loss +
                        lambda_aux_va * aux_va_loss +
                        wav_loss +
                        lambda_id * id_loss +
                        lambda_lpips * lpips_loss
                    )
                else:
                    # Only use main losses when not generating samples
                    total_loss = (
                        lambda_ddpm * ddpm_loss +
                        lambda_aux_expr * aux_expr_loss +
                        lambda_aux_va * aux_va_loss
                    )
                    wav_loss = torch.tensor(0.0, device=device)
                    id_loss = torch.tensor(0.0, device=device)
                    lpips_loss = torch.tensor(0.0, device=device)

                # ===== BACKWARD PASS =====
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                # ===== LOGGING =====
                epoch_bar.set_postfix({
                    "Total": f"{total_loss.item():.4f}",
                    "DDPM": f"{ddpm_loss.item():.4f}",
                    "Aux_Expr": f"{aux_expr_loss.item():.4f}",
                    "Aux_VA": f"{aux_va_loss.item():.4f}",
                    "Wav": f"{wav_loss.item():.4f}",
                    "ID": f"{id_loss.item():.4f}"
                })

                # TensorBoard logging
                if i % 10 == 0:
                    step = epoch * len(train_dataloader) + i
                    writer.add_scalar("Loss/Total", total_loss.item(), step)
                    writer.add_scalar("Loss/DDPM", ddpm_loss.item(), step)
                    writer.add_scalar("Loss/Aux_Expression", aux_expr_loss.item(), step)
                    writer.add_scalar("Loss/Aux_VA", aux_va_loss.item(), step)
                    writer.add_scalar("Loss/Wavelet", wav_loss.item(), step)
                    writer.add_scalar("Loss/Identity", id_loss.item(), step)
                    writer.add_scalar("Loss/LPIPS", lpips_loss.item(), step)

                # Save checkpoint
                if i % 100 == 0:
                    save_checkpoint(model_path, epoch, i, model, optimizer, total_loss)

                # Generate validation images
                if i % 200 == 0:
                    print("🖼️ Generating validation images...")
                    model.eval()
                    with torch.no_grad():
                        # Generate different emotions for fixed images
                        all_imgs = [x_fixed[:4]]  # Original images

                        for emotion_id in range(len(labels)):
                            emotion_tensor = torch.full((4,), emotion_id, device=device)
                            generated = model.sample(x_fixed[:4], emotion_tensor, num_steps=50)
                            all_imgs.append(generated)

                        # Combine all images
                        all_imgs = torch.cat(all_imgs, dim=0)
                        all_imgs = (all_imgs.clamp(-1, 1) + 1) / 2  # [-1,1] -> [0,1]

                        save_image(
                            all_imgs,
                            f"{out_path}/epoch{epoch}_iter{i}_emotions.png",
                            nrow=4,
                            normalize=False
                        )
                        print(f"✅ Saved validation images: {out_path}/epoch{epoch}_iter{i}_emotions.png")

                    model.train()

    except KeyboardInterrupt:
        print("⚠️ Training interrupted by user")
        if 'total_loss' in locals():
            save_checkpoint(model_path, epoch, i, model, optimizer, total_loss)

    print("✅ Training completed!")
    writer.close()


if __name__ == '__main__':
    train()
