import torch
import torch.nn.functional as F
from torchvision.transforms import Resize, ToTensor, Compose, Normalize
from torchvision.utils import save_image
from model import WaveletDiffusionModel, DWT, IWT
from dataset import Affectnet
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import time


def create_comparison_grid(images_dict, labels, save_path):
    """
    Tạo grid so sánh các kết quả với số steps khác nhau

    Args:
        images_dict: Dictionary {num_steps: generated_image_tensor}
        labels: List emotion labels
        save_path: Path to save the comparison grid
    """
    num_configs = len(images_dict)
    num_emotions = len(labels)

    fig, axes = plt.subplots(num_configs, num_emotions + 1, figsize=(3*(num_emotions+1), 3*num_configs))

    if num_configs == 1:
        axes = axes.reshape(1, -1)

    for config_idx, (config_name, images) in enumerate(images_dict.items()):
        # Original image
        orig_img = (images[0].cpu().permute(1, 2, 0).numpy() + 1) / 2
        orig_img = np.clip(orig_img, 0, 1)

        axes[config_idx, 0].imshow(orig_img)
        axes[config_idx, 0].set_title(f"Original\n{config_name}")
        axes[config_idx, 0].axis('off')

        # Generated images for each emotion
        for emo_idx in range(num_emotions):
            gen_img = (images[emo_idx + 1].cpu().permute(1, 2, 0).numpy() + 1) / 2
            gen_img = np.clip(gen_img, 0, 1)

            axes[config_idx, emo_idx + 1].imshow(gen_img)
            if config_idx == 0:
                axes[config_idx, emo_idx + 1].set_title(labels[emo_idx])
            axes[config_idx, emo_idx + 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison grid: {save_path}")


def add_text_overlay(image_tensor, text, position='bottom'):
    """
    Add text overlay to image tensor

    Args:
        image_tensor: (C, H, W) tensor in [-1, 1]
        text: Text to display
        position: 'top' or 'bottom'
    """
    img_np = ((image_tensor.cpu().permute(1, 2, 0).numpy() + 1) / 2 * 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()

    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Position text
    x = (img.width - text_width) // 2
    y = 5 if position == 'top' else img.height - text_height - 5

    # Draw background
    draw.rectangle([x-3, y-2, x+text_width+3, y+text_height+2], fill=(0, 0, 0, 200))
    draw.text((x, y), text, fill=(255, 255, 255), font=font)

    # Convert back to tensor
    img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1)
    img_tensor = img_tensor * 2 - 1  # Back to [-1, 1]

    return img_tensor


def test_denoising_steps():
    """
    Test model với các cấu hình num_steps và denoising_strength khác nhau
    """
    print("=" * 80)
    print("INFERENCE TEST - Finding Optimal Denoising Steps")
    print("=" * 80)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Hyperparameters
    image_size = 224
    labels = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"]

    # Model configuration (must match training)
    use_film = True
    use_adagn = False

    print(f"\nModel Configuration:")
    print(f"- Using FiLM: {use_film}")
    print(f"- Using AdaGN: {use_adagn}")

    # Paths
    model_path = "/mnt/ias-data/tam/data/WaveletDiffusion_v1/model"
    output_dir = "inference_test_results"
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print("\n" + "=" * 80)
    print("Loading Model...")
    print("=" * 80)

    model = WaveletDiffusionModel(
        num_emotions=len(labels),
        use_film=use_film,
        use_adagn=use_adagn
    ).to(device)

    checkpoint_path = "C:/Users/tam/Desktop/Data/FEG/modal/WaveletDiffusionV8/denoise/waveletDiffusionV8_denoise/WaveletDiffusionV8/model/best_model.pt"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(model_path, "last_model.pt")

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded checkpoint from: {checkpoint_path}")
        print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  - Loss: {checkpoint.get('loss', 'N/A'):.4f}")
    else:
        print(f"✗ No checkpoint found at {model_path}")
        return

    model.eval()

    # Load test data
    print("\n" + "=" * 80)
    print("Loading Test Data...")
    print("=" * 80)

    transform = Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    test_dataset = Affectnet(is_train=False, transform=transform)
    print(f"✓ Test dataset size: {len(test_dataset)}")

    # Select test samples
    num_test_samples = 3
    test_indices = np.random.choice(len(test_dataset), num_test_samples, replace=False)

    # Test configurations
    test_configs = [
        # (num_steps, denoising_strength, description)
        (20, 0.3, "20steps_strength0.3"),
        (50, 0.3, "50steps_strength0.3"),
        (100, 0.3, "100steps_strength0.3"),
        (200, 0.3, "200steps_strength0.3"),

        (50, 0.2, "50steps_strength0.2"),
        (50, 0.4, "50steps_strength0.4"),
        (50, 0.5, "50steps_strength0.5"),

        (100, 0.2, "100steps_strength0.2"),
        (100, 0.4, "100steps_strength0.4"),
        (100, 0.5, "100steps_strength0.5"),
    ]

    print(f"\nTest Configurations: {len(test_configs)}")
    for i, (steps, strength, desc) in enumerate(test_configs):
        print(f"  {i+1}. {desc}: num_steps={steps}, denoising_strength={strength}")

    # Run inference
    print("\n" + "=" * 80)
    print("Running Inference...")
    print("=" * 80)

    with torch.no_grad():
        for sample_idx, data_idx in enumerate(test_indices):
            print(f"\n--- Sample {sample_idx + 1}/{num_test_samples} (Index: {data_idx}) ---")

            img, orig_emotion, _, _ = test_dataset[data_idx]
            img = img.unsqueeze(0).to(device)

            print(f"Original emotion: {labels[orig_emotion]}")

            # Store results for comparison
            results = {}
            timing_results = {}

            for num_steps, strength, config_name in tqdm(test_configs, desc="Testing configs"):
                config_images = []

                # Add original image
                config_images.append(img[0])

                # Measure inference time
                start_time = time.time()

                # Generate for each emotion
                for target_emotion_id in range(len(labels)):
                    target_emotion = torch.tensor([target_emotion_id], device=device)

                    generated = model.sample(
                        img,
                        target_emotion,
                        num_steps=num_steps,
                        denoising_strength=strength
                    )

                    config_images.append(generated[0])

                inference_time = time.time() - start_time
                timing_results[config_name] = inference_time / len(labels)  # Average per emotion

                # Stack all images for this config
                results[config_name] = config_images

            # Create comparison grid
            save_path = os.path.join(output_dir, f"sample_{sample_idx}_comparison.png")

            # Prepare images dict for plotting
            images_dict = {}
            for config_name, config_images in results.items():
                images_dict[config_name] = config_images

            create_comparison_grid(images_dict, labels, save_path)

            # Print timing results
            print("\nInference Time (avg per emotion):")
            for config_name, avg_time in sorted(timing_results.items(), key=lambda x: x[1]):
                print(f"  {config_name}: {avg_time:.3f}s")

            # Save individual config results
            for config_name, config_images in results.items():
                config_dir = os.path.join(output_dir, f"sample_{sample_idx}", config_name)
                os.makedirs(config_dir, exist_ok=True)

                # Save original
                save_image((config_images[0] + 1) / 2,
                          os.path.join(config_dir, "0_original.png"))

                # Save generated emotions
                for emo_idx in range(len(labels)):
                    # Add text overlay
                    img_with_text = add_text_overlay(
                        config_images[emo_idx + 1],
                        labels[emo_idx]
                    )
                    save_image((img_with_text + 1) / 2,
                              os.path.join(config_dir, f"{emo_idx+1}_{labels[emo_idx]}.png"))

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n✓ Tested {len(test_configs)} configurations on {num_test_samples} samples")
    print(f"✓ Results saved to: {output_dir}")
    print("\nRecommendations:")
    print("  - Check the comparison grids to find the best balance between:")
    print("    * Image quality")
    print("    * Emotion transfer effectiveness")
    print("    * Identity preservation")
    print("    * Inference speed")
    print("\n  - Generally:")
    print("    * More steps = Better quality but slower")
    print("    * Higher strength = Stronger emotion change but may lose identity")
    print("    * Start with 50-100 steps and 0.3-0.4 strength")
    print("=" * 80)


def test_single_config(num_steps=100, denoising_strength=0.4, num_samples=5):
    """
    Test với một cấu hình cụ thể trên nhiều samples

    Args:
        num_steps: Number of denoising steps
        denoising_strength: Denoising strength (0.0-1.0)
        num_samples: Number of test samples
    """
    print("=" * 80)
    print(f"SINGLE CONFIG TEST: steps={num_steps}, strength={denoising_strength}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 224
    labels = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"]

    # Load model
    model = WaveletDiffusionModel(num_emotions=len(labels), use_film=True, use_adagn=False).to(device)

    model_path = "/mnt/ias-data/tam/data/WaveletDiffusion_v1/model"
    checkpoint_path = os.path.join(model_path, "best_model.pt")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(model_path, "last_model.pt")

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded checkpoint")
    else:
        print(f"✗ No checkpoint found")
        return

    model.eval()

    # Load data
    transform = Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    test_dataset = Affectnet(is_train=False, transform=transform)
    test_indices = np.random.choice(len(test_dataset), num_samples, replace=False)

    output_dir = f"inference_single_test_steps{num_steps}_strength{denoising_strength}"
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for sample_idx, data_idx in enumerate(tqdm(test_indices, desc="Processing samples")):
            img, orig_emotion, _, _ = test_dataset[data_idx]
            img = img.unsqueeze(0).to(device)

            # Create grid: original + all emotions
            grid_images = [img[0]]  # Original

            for target_emotion_id in range(len(labels)):
                target_emotion = torch.tensor([target_emotion_id], device=device)
                generated = model.sample(img, target_emotion, num_steps=num_steps,
                                        denoising_strength=denoising_strength)
                grid_images.append(generated[0])

            # Save grid
            grid = torch.stack(grid_images)
            save_image((grid + 1) / 2,
                      os.path.join(output_dir, f"sample_{sample_idx}_grid.png"),
                      nrow=len(labels)+1, padding=2, pad_value=1)

    print(f"\n✓ Results saved to: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test denoising steps for Wavelet Diffusion")
    parser.add_argument("--mode", type=str, default="compare", choices=["compare", "single"],
                       help="Test mode: 'compare' multiple configs or 'single' config")
    parser.add_argument("--steps", type=int, default=100,
                       help="Number of denoising steps (for single mode)")
    parser.add_argument("--strength", type=float, default=0.4,
                       help="Denoising strength 0.0-1.0 (for single mode)")
    parser.add_argument("--num_samples", type=int, default=3,
                       help="Number of test samples")

    args = parser.parse_args()

    if args.mode == "compare":
        test_denoising_steps()
    else:
        test_single_config(
            num_steps=args.steps,
            denoising_strength=args.strength,
            num_samples=args.num_samples
        )
