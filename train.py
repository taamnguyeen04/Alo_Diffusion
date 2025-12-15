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
from loss_conflict_analyzer import LossConflictAnalyzer
from icecream import ic
import math
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# pip install insightface

def add_film_params_to_image(image_tensor, film_params, emotion_label, conditioning_type="FiLM"):
    """
    Add FiLM/AdaGN gamma/beta statistics as text overlay on image

    Args:
        image_tensor: (C, H, W) tensor in [0, 1]
        film_params: list of (gamma, beta) tuples from model
        emotion_label: string label for the emotion
        conditioning_type: "FiLM" or "AdaGN" for display purposes

    Returns:
        PIL Image with text overlay
    """
    # Convert tensor to PIL Image
    img_np = (image_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    draw = ImageDraw.Draw(img)

    # Calculate statistics from film_params
    if film_params:
        # Aggregate all gamma and beta values
        all_gammas = []
        all_betas = []
        for gamma, beta in film_params:
            if gamma is not None and beta is not None:
                # gamma, beta shape: (B, C, 1, 1)
                all_gammas.append(gamma.squeeze().flatten())
                all_betas.append(beta.squeeze().flatten())

        if all_gammas and all_betas:
            # Concatenate and compute statistics
            all_gammas = torch.cat(all_gammas)
            all_betas = torch.cat(all_betas)

            gamma_mean = all_gammas.mean().item()
            gamma_std = all_gammas.std().item()
            beta_mean = all_betas.mean().item()
            beta_std = all_betas.std().item()

            # Create text
            text = f"{emotion_label} ({conditioning_type})\nγ: {gamma_mean:.3f}±{gamma_std:.3f}\nβ: {beta_mean:.3f}±{beta_std:.3f}"
        else:
            text = f"{emotion_label}\nNo params"
    else:
        text = f"{emotion_label}\nNo conditioning"

    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()

    # Draw text with background for better visibility
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Position at bottom left
    x, y = 5, img.height - text_height - 5

    # Draw semi-transparent background
    draw.rectangle([x-2, y-2, x+text_width+2, y+text_height+2], fill=(0, 0, 0, 180))

    # Draw text in white
    draw.text((x, y), text, fill=(255, 255, 255), font=font)

    return img

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
            best_loss = torch.load(best_path, map_location='cpu', weights_only=True)['loss']
            if loss.item() < best_loss:
                torch.save(checkpoint, best_path)
                print(f"Đã cập nhật best_model.pt: {best_loss:.4f} → {loss.item():.4f}")
        except Exception as e:
            print(f"Lỗi khi đọc best_model.pt: {e} → lưu đè")
            torch.save(checkpoint, best_path)


def load_checkpoint(filepath, model, optimizer, best_loss, device):
    last_path = os.path.join("WaveletDiffusion_film04/WaveletDiffusion_film04/model", "best_model.pt")
    best_path = os.path.join("WaveletDiffusion_film04/WaveletDiffusion_film04/model", "best_model.pt")

    start_epoch = 0
    loaded = False

    if os.path.isfile(last_path):
        try:
            checkpoint = torch.load(last_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            loaded = True
            print(f"Loaded từ last_model.pt epoch {start_epoch}")
        except Exception as e:
            print(f"Lỗi khi load last_model.pt: {e}")

    if not loaded and os.path.isfile(best_path):
        try:
            checkpoint = torch.load(best_path, map_location=device, weights_only=True)
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
            best_loss = torch.load(best_path, map_location=device, weights_only=True)['loss']
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
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    # Hyperparameters - Reduced batch size for memory
    batch_size = 128
    lr = 1e-5
    num_epochs = 5
    image_size = 224
    labels = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"]

    # Emotion conditioning method toggle
    use_film = True  # FiLM: Feature-wise Linear Modulation
    use_adagn = False   # AdaGN: Adaptive Group Normalization

    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print(f"Using FiLM conditioning: {use_film}")
    print(f"Using AdaGN conditioning: {use_adagn}")

    # Loss weights - sử dụng DAN emotion model
    lambda_ddpm = 2.0
    lambda_wav_ll = 0.6
    lambda_wav_hi = 0.5
    lambda_dan_expr = 1.6


    # Directories
    log_dir = "WaveletDiffusion_emotion/runs/exp"
    model_path = "WaveletDiffusion_emotion/model"
    out_path = "WaveletDiffusion_emotion/out"

    for dir_path in [log_dir, model_path, out_path]:
        if dir_path == model_path:
            os.makedirs(dir_path, exist_ok=True)
            continue

        if os.path.exists(dir_path):
            # shutil.rmtree(dir_path)
            pass
        os.makedirs(dir_path, exist_ok=True)

    writer = SummaryWriter(log_dir)
    best_loss = float('inf')

    # Data transforms
    transform = Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets
    print("Loading datasets...")
    train_dataset = Affectnet(is_train=True, transform=transform)
    val_dataset = Affectnet(is_train=False, transform=transform)

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

    model = WaveletDiffusionModel(num_emotions=len(labels), use_film=use_film, use_adagn=use_adagn).to(device)

    # Wrap model with DataParallel if multiple GPUs available
    if num_gpus > 1:
        print(f"Using DataParallel with {num_gpus} GPUs")
        model = torch.nn.DataParallel(model)

    emotion_model = Emotion_model()

    # Freeze DAN emotion model parameters
    for param in emotion_model.model.parameters():
        param.requires_grad = False


    initial_weights = {
        'lambda_dan_expr': lambda_dan_expr
    }
    loss_weighter = AdaptiveLossWeighter(initial_weights, warmup_steps=5000)

    # Initialize PerceptualWaveletLoss
    perceptual_wavelet_loss = PerceptualWaveletLoss().to(device)

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

    # Optimizer with separate learning rates for AdaGN parameters
    # AdaGN (emotion_modulation) learns slower to prevent artifacts from abrupt changes
    base_model = model.module if num_gpus > 1 else model

    adagn_params = []
    other_params = []

    for name, param in base_model.named_parameters():
        if 'emotion_modulation' in name:
            adagn_params.append(param)
        else:
            other_params.append(param)

    print(f"AdaGN parameters: {len(adagn_params)}, Other parameters: {len(other_params)}")

    optimizer = torch.optim.AdamW([
        {'params': other_params, 'lr': lr, 'weight_decay': 1e-4},
        {'params': adagn_params, 'lr': lr * 0.1, 'weight_decay': 1e-6}  # 10x slower learning
    ], betas=(0.9, 0.999))

    # Load checkpoint
    print("Loading checkpoint...")
    # When using DataParallel, need to load to module (reuse base_model variable)
    start_epoch, best_loss = load_checkpoint(model_path, base_model, optimizer, best_loss, device)

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

                # ===== FORWARD PASS =====
                x_wavelet = dwt(img_real)
                t = torch.randint(0, base_model.num_timesteps, (img_real.shape[0],), device=device)
                x_noisy, noise = base_model.forward_process(x_wavelet, t)
                noise_pred = base_model.unet(x_noisy, t, expr_org, img_real)
                ddpm_loss = F.l1_loss(noise_pred, noise)

                # Calculate DAN loss directly from pred_x0 (differentiable) with time-based weighting
                alpha_t = base_model.alphas_cumprod[t][:, None, None, None]
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
                t_thresh = int(0.3 * base_model.num_timesteps)  # Only for t < 30% of total timesteps
                time_weights = (t < t_thresh).float()  # (B,)

                # Calculate per-sample loss and apply time weighting
                per_sample_dan_loss = F.cross_entropy(dan_out, expr_org, reduction='none')  # (B,)
                dan_expr_loss = (per_sample_dan_loss * time_weights).sum() / (time_weights.sum().clamp_min(1.0))

                # If no valid timesteps, set loss to 0
                if time_weights.sum() == 0:
                    dan_expr_loss = torch.tensor(0.0, device=device, requires_grad=True)

                current_losses = {
                    'dan_expr': dan_expr_loss,
                    'ddpm': ddpm_loss
                }
                weight_updates = loss_weighter.update_weights(current_losses)
                current_lambda_dan_expr = weight_updates.get('lambda_dan_expr', lambda_dan_expr)

                if i % 30 == 0:
                    # ic(dan_predictions, expr_org, expr_org)
                    with torch.no_grad():
                        # Sử dụng denoising strength rất cao để ép model học thay đổi cảm xúc mạnh
                        generated_img = base_model.sample(img_real, expr_org, num_steps=100, denoising_strength=0.4)

                    # Use PerceptualWaveletLoss instead of manual wavelet loss calculation
                    wav_loss = perceptual_wavelet_loss(
                        generated_img,
                        img_real,
                        lambda_ll=lambda_wav_ll,  # Weight for perceptual loss on LL band
                        lambda_hi=lambda_wav_hi   # Weight for L1 loss on HF bands
                    )

                    # ===== TOTAL LOSS (with adaptive weights) =====
                    total_loss = (
                            lambda_ddpm * ddpm_loss +
                            current_lambda_dan_expr * dan_expr_loss +
                            wav_loss
                    )
                else:
                    total_loss = (
                            lambda_ddpm * ddpm_loss +
                            current_lambda_dan_expr * dan_expr_loss
                    )
                    wav_loss = torch.tensor(0.0, device=device)

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
                    "Wav": f"{wav_loss.item():.4f}"
                })

                # TensorBoard logging
                if i % 50 == 0:
                    step = epoch * len(train_dataloader) + i
                    writer.add_scalar("Loss/Total", total_loss.item(), step)
                    writer.add_scalar("Loss/DDPM", ddpm_loss.item(), step)
                    writer.add_scalar("Loss/DAN_Expression", dan_expr_loss.item(), step)
                    writer.add_scalar("Loss/Wavelet", wav_loss.item(), step)

                    # Log FiLM parameters statistics during training
                    with torch.no_grad():
                        # Get a single sample to check FiLM params
                        sample_img = img_real[:1]
                        sample_emotion = expr_org[:1]
                        _, sample_film_params = base_model.sample(
                            sample_img,
                            sample_emotion,
                            num_steps=20,  # Use fewer steps for speed
                            denoising_strength=0.1,
                            return_film_params=True
                        )

                        if sample_film_params:
                            # Aggregate statistics
                            all_gammas = []
                            all_betas = []
                            for gamma, beta in sample_film_params:
                                all_gammas.append(gamma.flatten())
                                all_betas.append(beta.flatten())

                            all_gammas = torch.cat(all_gammas)
                            all_betas = torch.cat(all_betas)

                            # Log statistics
                            writer.add_scalar("FiLM/Gamma_Mean", all_gammas.mean().item(), step)
                            writer.add_scalar("FiLM/Gamma_Std", all_gammas.std().item(), step)
                            writer.add_scalar("FiLM/Gamma_Min", all_gammas.min().item(), step)
                            writer.add_scalar("FiLM/Gamma_Max", all_gammas.max().item(), step)

                            writer.add_scalar("FiLM/Beta_Mean", all_betas.mean().item(), step)
                            writer.add_scalar("FiLM/Beta_Std", all_betas.std().item(), step)
                            writer.add_scalar("FiLM/Beta_Min", all_betas.min().item(), step)
                            writer.add_scalar("FiLM/Beta_Max", all_betas.max().item(), step)

                            # Log histogram every 200 steps
                            if i % 200 == 0:
                                writer.add_histogram("FiLM/Gamma_Distribution", all_gammas, step)
                                writer.add_histogram("FiLM/Beta_Distribution", all_betas, step)

                if i % 800 == 0:
                    save_checkpoint(model_path, epoch, i, base_model, optimizer, total_loss)
                    print("Generating validation images...")
                    model.eval()
                    with torch.no_grad():
                        # Determine conditioning type for display
                        if use_adagn:
                            cond_type = "AdaGN"
                        elif use_film:
                            cond_type = "FiLM"
                        else:
                            cond_type = "None"

                        # Generate images with conditioning parameters
                        pil_images = []
                        tensor_images = []
                        # Process each sample separately to get individual params
                        for sample_idx in range(min(4, x_fixed.shape[0])):
                            sample = x_fixed[sample_idx:sample_idx+1]

                            # Add original image
                            orig_img_tensor = (sample[0].clamp(-1, 1) + 1) / 2  # [0,1]
                            orig_pil = add_film_params_to_image(orig_img_tensor, None, "Original", cond_type)
                            pil_images.append(orig_pil)
                            tensor_images.append(orig_img_tensor)

                            # Generate for each emotion
                            for emotion_id in range(len(labels)):
                                emotion_tensor = torch.full((1,), emotion_id, device=device)
                                generated, cond_params = base_model.sample(
                                    sample,
                                    emotion_tensor,
                                    num_steps=250,
                                    denoising_strength=0.4,
                                    return_film_params=True
                                )

                                gen_img_tensor = (generated[0].clamp(-1, 1) + 1) / 2  # [0,1]
                                gen_pil = add_film_params_to_image(
                                    gen_img_tensor,
                                    cond_params,
                                    labels[emotion_id],
                                    cond_type
                                )
                                pil_images.append(gen_pil)
                                tensor_images.append(gen_img_tensor)

                        # Create grid from PIL images
                        if pil_images:
                            # Calculate grid dimensions
                            n_emotions = len(labels) + 1  # +1 for original
                            n_samples = min(4, x_fixed.shape[0])
                            img_width, img_height = pil_images[0].size

                            grid_width = n_emotions * img_width
                            grid_height = n_samples * img_height

                            grid_img = Image.new('RGB', (grid_width, grid_height))

                            for idx, pil_img in enumerate(pil_images):
                                row = idx // n_emotions
                                col = idx % n_emotions
                                x_pos = col * img_width
                                y_pos = row * img_height
                                grid_img.paste(pil_img, (x_pos, y_pos))

                            grid_img.save(f"{out_path}/epoch{epoch}_iter{i}_emotions_film.png")
                            print(f"Saved validation images with FiLM params: {out_path}/epoch{epoch}_iter{i}_emotions_film.png")
                        
                        if tensor_images:
                            all_imgs = torch.stack(tensor_images, dim=0)
                            all_imgs_grid = all_imgs.reshape(n_samples, n_emotions, *all_imgs.shape[1:])
                            writer.add_images(
                                "Validation/All_Emotions_Grid", 
                                all_imgs_grid.flatten(0, 1), 
                                step, 
                                dataformats='NCHW'
                            )
                            
                            for emo_idx, emo_name in enumerate(labels):
                                emo_imgs = all_imgs_grid[:, emo_idx + 1]  # +1 to skip original
                                writer.add_images(
                                    f"Validation/Emotion_{emo_name}",
                                    emo_imgs,
                                    step,
                                    dataformats='NCHW'
                                )

                    model.train()

    except KeyboardInterrupt:
        print("Training interrupted by user")
        if 'total_loss' in locals():
            save_checkpoint(model_path, epoch, i, base_model, optimizer, total_loss)

    print("Training completed!")
    writer.close()

if __name__ == '__main__':
    train()
