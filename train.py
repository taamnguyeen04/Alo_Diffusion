import shutil
from torchvision.utils import save_image
from torchvision.transforms import Resize, ToTensor, Compose, Normalize
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import WaveletDiffusionModel, DWT, IWT
from dataset import Affectnet
from tqdm import tqdm
import lpips
from criterions import *
from icecream import ic
import math
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# pip install insightface

def save_checkpoint(filepath, epoch, step, model, optimizer, loss):
    print(f"Đang lưu checkpoint: epoch {epoch}, step {step}, loss {loss.item():.4f}")

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
        print("Đã lưu best_model.pt (lần đầu)")
    else:
        try:
            best_loss = torch.load(best_path, map_location='cpu')['loss']
            if loss.item() < best_loss:
                torch.save(checkpoint, best_path)
                print(f"Đã cập nhật best_model.pt: {best_loss:.4f} → {loss.item():.4f}")
        except Exception as e:
            print(f"Lỗi khi đọc best_model.pt: {e} → lưu đè")
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
            print(f"Loaded từ last_model.pt epoch {start_epoch}")
        except Exception as e:
            print(f"Lỗi khi load last_model.pt: {e}")

    if not loaded and os.path.isfile(best_path):
        try:
            checkpoint = torch.load(best_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"Loaded từ best_model.pt epoch {start_epoch}")
            loaded = True
        except Exception as e:
            print(f"Lỗi khi load best_model.pt: {e}")

    if not loaded:
        print("Không tìm thấy hoặc không load được checkpoint, bắt đầu từ đầu.")
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

    # Hyperparameters - Reduced batch size for memory
    batch_size = 32
    lr = 1e-4
    num_epochs = 1
    image_size = 224
    labels = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"]

    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")

    # Loss weights - sử dụng DAN emotion model
    lambda_ddpm = 2.0
    lambda_wav_ll = 0.6
    lambda_wav_hi = 0.5
    lambda_dan_expr = 1.6
    lambda_id = 0.8
    lambda_lpips = 0.5


    # Directories
    log_dir = "WaveletDiffusion5/runs/exp"
    model_path = "WaveletDiffusion5/model"
    out_path = "WaveletDiffusion5/out"

    for dir_path in [log_dir, model_path, out_path]:
        if dir_path == model_path:
            os.makedirs(dir_path, exist_ok=True)
            continue

        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=True)

    writer = SummaryWriter(log_dir)
    best_loss = float('inf')

    # Data transforms
    transform = Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Datasets
    print("Loading datasets...")
    train_dataset = Affectnet(root="C:/Users/tam/Documents/data/FEG", is_train=True, transform=transform)
    val_dataset = Affectnet(root="C:/Users/tam/Documents/data/FEG", is_train=False, transform=transform)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=os.cpu_count(),  # dùng hết CPU core logic
        shuffle=True,
        drop_last=True,
        pin_memory=True,  # copy CPU→GPU nhanh hơn
        persistent_workers=True,  # tránh spawn lại worker mỗi epoch
        prefetch_factor=4  # mỗi worker load trước nhiều batch
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=min(4, os.cpu_count()),
        shuffle=False,
        drop_last=True
    )
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    print("Building models...")

    model = WaveletDiffusionModel(num_emotions=len(labels)).to(device)
    emotion_model = Emotion_model()

    # Freeze DAN emotion model parameters
    for param in emotion_model.model.parameters():
        param.requires_grad = False

    import torchvision.models as models
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
    resnet50.eval()
    for param in resnet50.parameters():
        param.requires_grad = False
    try:
        lpips_loss_fn = lpips.LPIPS(net='vgg', pretrained=True).to(device)
    except:
        lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)
    for param in lpips_loss_fn.parameters():
        param.requires_grad = False

    initial_weights = {
        'lambda_id': lambda_id,
        'lambda_dan_expr': lambda_dan_expr
    }
    loss_weighter = AdaptiveLossWeighter(initial_weights, warmup_steps=5000)

    dwt = DWT().to(device)
    iwt = IWT().to(device)
    with torch.no_grad():
        x = next(iter(val_dataloader))[0].to(device)[:4]
        x_wave = dwt(x)
        x_rec = iwt(x_wave)
        mse = ((x - x_rec) ** 2).mean().item()
        print("DWT/IWT roundtrip MSE:", mse)

    # PSNR
    rmse = math.sqrt(mse)
    psnr = 20 * math.log10(2.0 / (rmse + 1e-12))  # dynamic range 2 for [-1,1]
    print('MSE', mse, 'PSNR', psnr)

    # # SSIM per image (với batch x và x_rec dạng torch tensor)
    # x_np = x.cpu().numpy().transpose(0, 2, 3, 1)  # (B, H, W, C)
    # xrec_np = x_rec.cpu().numpy().transpose(0, 2, 3, 1)
    #
    # for i in range(x_np.shape[0]):
    #     # Chuyển ảnh về [0,1] và cắt biên an toàn
    #     img1 = np.clip((x_np[i] + 1) / 2, 0, 1)
    #     img2 = np.clip((xrec_np[i] + 1) / 2, 0, 1)
    #
    #     # Xác định kích thước cửa sổ hợp lệ
    #     win_size = min(7, img1.shape[0], img1.shape[1])
    #     if win_size % 2 == 0:
    #         win_size -= 1
    #     win_size = max(win_size, 3)  # luôn >=3 và lẻ
    #
    #     try:
    #         # Dùng channel_axis=-1 (chuẩn cho scikit-image >= 0.19)
    #         s = ssim(img1, img2, channel_axis=-1, win_size=win_size, data_range=1.0)
    #     except TypeError:
    #         # fallback cho bản cũ (<0.19) vẫn còn multichannel
    #         s = ssim(img1, img2, multichannel=True, win_size=win_size, data_range=1.0)
    #
    #     print(f"SSIM image {i}: {s:.6f}")
    #
    # s = ssim(img1, img2, channel_axis=-1, win_size=win_size, data_range=1.0)
    # print("SSIM image", i, s)
    #
    # plt.subplot(1, 2, 1)
    # plt.imshow((x_np[2] + 1) / 2)
    # plt.title("Original")
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow((xrec_np[2] + 1) / 2)
    # plt.title("Reconstructed")
    # plt.show()

    # # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4
    )

    # Load checkpoint
    print("Loading checkpoint...")
    start_epoch, best_loss = load_checkpoint(model_path, model, optimizer, best_loss, device)

    # Fixed validation samples
    x_fixed, expr_fixed, _, _ = next(iter(val_dataloader))
    x_fixed = x_fixed.to(device)
    expr_fixed = expr_fixed.to(device)

    try:
        print("Starting training...")
        for epoch in range(start_epoch, num_epochs):
            model.train()

            epoch_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                             desc=f"Epoch {epoch}/{num_epochs}")

            for i, (img_real, expr_org, _, _) in epoch_bar:  # Bỏ valence, arousal
                img_real = img_real.to(device)
                expr_org = expr_org.to(device)

                rand_idx = torch.randperm(expr_org.size(0))
                expr_trg = expr_org[rand_idx]

                # ===== FORWARD PASS =====

                x_wavelet = dwt(img_real)
                t = torch.randint(0, model.num_timesteps, (img_real.shape[0],), device=device)
                x_noisy, noise = model.forward_process(x_wavelet, t)
                noise_pred = model.unet(x_noisy, t, expr_trg, img_real)
                ddpm_loss = F.l1_loss(noise_pred, noise)

                # Calculate DAN loss directly from pred_x0 (differentiable) with time-based weighting
                alpha_t = model.alphas_cumprod[t][:, None, None, None]
                sqrt_alpha_t = torch.sqrt(alpha_t)
                sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)

                # Predict x0 from noise prediction
                pred_x0_wavelet = (x_noisy - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t

                # Convert wavelet back to image space
                pred_x0_img = iwt(pred_x0_wavelet)
                pred_x0_img = torch.clamp(pred_x0_img, -1, 1)

                # Normalize to [0,1] and resize for DAN
                pred_x0_norm = (pred_x0_img + 1) / 2
                dan_input = F.interpolate(pred_x0_norm, size=(224, 224), mode='bilinear')

                # ImageNet normalization for DAN
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(dan_input.device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(dan_input.device)
                dan_input = (dan_input - mean) / std

                # Get DAN predictions (differentiable)
                dan_out, _, _ = emotion_model.model(dan_input)

                # Time-based weighting: only apply DAN loss for low noise timesteps
                t_thresh = int(0.3 * model.num_timesteps)  # Only for t < 30% of total timesteps
                time_weights = (t < t_thresh).float()  # (B,)

                # Calculate per-sample loss and apply time weighting
                per_sample_dan_loss = F.cross_entropy(dan_out, expr_trg, reduction='none')  # (B,)
                dan_expr_loss = (per_sample_dan_loss * time_weights).sum() / (time_weights.sum().clamp_min(1.0))

                # If no valid timesteps, set loss to 0
                if time_weights.sum() == 0:
                    dan_expr_loss = torch.tensor(0.0, device=device, requires_grad=True)

                current_losses = {
                    'dan_expr': dan_expr_loss,
                    'ddpm': ddpm_loss
                }
                weight_updates = loss_weighter.update_weights(current_losses)
                current_lambda_id = weight_updates.get('lambda_id', lambda_id)
                current_lambda_dan_expr = weight_updates.get('lambda_dan_expr', lambda_dan_expr)

                if i % 50 == 0:
                    # ic(dan_predictions, expr_trg, expr_org)
                    with torch.no_grad():
                        # Sử dụng denoising strength rất cao để ép model học thay đổi cảm xúc mạnh
                        generated_img = model.sample(img_real, expr_trg, num_steps=75, denoising_strength=0.2)

                    img_wavelet_full = dwt(img_real)
                    gen_wavelet = dwt(generated_img)

                    C = img_real.shape[1]
                    img_ll = img_wavelet_full[:, :C, :, :]
                    img_hi = img_wavelet_full[:, C:, :, :]
                    gen_ll = gen_wavelet[:, :C, :, :]
                    gen_hi = gen_wavelet[:, C:, :, :]

                    wav_loss_ll = F.l1_loss(gen_ll, img_ll)
                    wav_loss_hi = F.l1_loss(gen_hi, img_hi)

                    with torch.no_grad():
                        same_emotion_img = model.sample(img_real, expr_org, num_steps=75)
                        same_emotion_wavelet = dwt(same_emotion_img)
                        same_ll = same_emotion_wavelet[:, :C, :, :]
                        same_hi = same_emotion_wavelet[:, C:, :, :]

                    emotion_wav_hi_loss = F.l1_loss(gen_hi, same_hi) * 0.5
                    emotion_wav_ll_loss = F.l1_loss(gen_ll, same_ll) * 0.1

                    wav_loss = (lambda_wav_ll * wav_loss_ll +
                                lambda_wav_hi * wav_loss_hi +
                                emotion_wav_hi_loss +
                                emotion_wav_ll_loss)

                    real_norm = (img_real + 1) / 2  # [-1,1] -> [0,1]
                    gen_norm = (generated_img + 1) / 2

                    real_resized = F.interpolate(real_norm, size=(224, 224), mode='bilinear')
                    gen_resized = F.interpolate(gen_norm, size=(224, 224), mode='bilinear')

                    # ImageNet normalization
                    mean_resnet = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
                    std_resnet = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
                    real_resized = (real_resized - mean_resnet) / std_resnet
                    gen_resized = (gen_resized - mean_resnet) / std_resnet

                    with torch.no_grad():
                        real_features = resnet50(real_resized)
                        gen_features = resnet50(gen_resized)

                    id_loss = 1 - F.cosine_similarity(real_features, gen_features).mean()

                    lpips_loss = lpips_loss_fn(img_real, generated_img).mean()

                    # ===== TOTAL LOSS (with adaptive weights) =====
                    total_loss = (
                            lambda_ddpm * ddpm_loss +
                            current_lambda_dan_expr * dan_expr_loss +
                            wav_loss +
                            current_lambda_id * id_loss +
                            lambda_lpips * lpips_loss
                    )
                else:
                    total_loss = (
                            lambda_ddpm * ddpm_loss +
                            current_lambda_dan_expr * dan_expr_loss
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
                    "DAN_Expr": f"{dan_expr_loss.item():.4f}",
                    "Wav": f"{wav_loss.item():.4f}",
                    "ID": f"{id_loss.item():.4f}"
                })

                # TensorBoard logging
                if i % 50 == 0:
                    step = epoch * len(train_dataloader) + i
                    writer.add_scalar("Loss/Total", total_loss.item(), step)
                    writer.add_scalar("Loss/DDPM", ddpm_loss.item(), step)
                    writer.add_scalar("Loss/DAN_Expression", dan_expr_loss.item(), step)
                    writer.add_scalar("Loss/Wavelet", wav_loss.item(), step)
                    writer.add_scalar("Loss/Identity", id_loss.item(), step)
                    writer.add_scalar("Loss/LPIPS", lpips_loss.item(), step)

                if i % 500 == 0:
                    save_checkpoint(model_path, epoch, i, model, optimizer, total_loss)
                    print("Generating validation images...")
                    model.eval()
                    with torch.no_grad():
                        all_imgs = [x_fixed[:4]]  # Original images

                        for emotion_id in range(len(labels)):
                            emotion_tensor = torch.full((4,), emotion_id, device=device)
                            # Sử dụng denoising strength cao để tạo sự khác biệt cảm xúc rõ ràng
                            generated = model.sample(x_fixed[:4], emotion_tensor, num_steps=75, denoising_strength=0.1)
                            all_imgs.append(generated)

                        all_imgs = torch.cat(all_imgs, dim=0)
                        all_imgs = (all_imgs.clamp(-1, 1) + 1) / 2  # [-1,1] -> [0,1]

                        save_image(
                            all_imgs,
                            f"{out_path}/epoch{epoch}_iter{i}_emotions.png",
                            nrow=4,
                            normalize=False
                        )
                        print(f"Saved validation images: {out_path}/epoch{epoch}_iter{i}_emotions.png")

                    model.train()

    except KeyboardInterrupt:
        print("Training interrupted by user")
        if 'total_loss' in locals():
            save_checkpoint(model_path, epoch, i, model, optimizer, total_loss)

    print("Training completed!")
    writer.close()

#DWT/IWT roundtrip MSE: 4.485536833458148e-14

if __name__ == '__main__':
    train()
