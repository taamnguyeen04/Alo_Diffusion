import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from model import WaveletDiffusionModel, DWT, IWT
from dataset import Affectnet
from torchvision.transforms import Resize, ToTensor, Compose, Normalize
from torch.utils.data import DataLoader
import os

def test_sampling_comparison():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup
    image_size = 224
    labels = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"]

    # Data transform
    transform = Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load dataset
    print("Loading dataset...")
    val_dataset = Affectnet(root="C:/Users/tam/Documents/data/FEG", is_train=False, transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False, drop_last=True)

    # Load model
    print("Loading model...")
    model = WaveletDiffusionModel(num_emotions=len(labels)).to(device)

    # Try to load checkpoint
    model_path = "WaveletDiffusion5/model/best_model.pt"
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {model_path}")
    else:
        print("No checkpoint found, using random weights")

    model.eval()

    # Get test sample
    x_test, expr_test, _, _ = next(iter(val_dataloader))
    x_test = x_test.to(device)
    expr_test = expr_test.to(device)

    print(f"Model num_timesteps: {model.num_timesteps}")
    print(f"Test image shape: {x_test.shape}")
    print(f"Original emotions: {expr_test}")

    # Test different target emotions
    target_emotions = [1, 2, 6]  # Happy, Sad, Anger
    step_counts = [75, 150, 500, model.num_timesteps]  # Different step counts

    with torch.no_grad():
        all_images = [x_test]  # Original images

        for target_emotion in target_emotions:
            emotion_tensor = torch.full((x_test.shape[0],), target_emotion, device=device)

            for num_steps in step_counts:
                print(f"Generating with emotion {labels[target_emotion]}, {num_steps} steps...")

                try:
                    generated = model.sample(x_test, emotion_tensor, num_steps=num_steps, denoising_strength=0.1)
                    all_images.append(generated)
                    print(f"  Success: {generated.shape}")
                except Exception as e:
                    print(f"  Error with {num_steps} steps: {e}")
                    # Add black image as placeholder
                    black_img = torch.zeros_like(x_test)
                    all_images.append(black_img)

        # Save comparison
        all_images = torch.cat(all_images, dim=0)
        all_images = (all_images.clamp(-1, 1) + 1) / 2  # [-1,1] -> [0,1]

        # Create output directory
        os.makedirs("sampling_comparison", exist_ok=True)

        save_image(
            all_images,
            "sampling_comparison/step_comparison.png",
            nrow=len(step_counts) + 1,  # Original + 4 step counts
            normalize=False
        )

        print(f"Saved comparison image: sampling_comparison/step_comparison.png")
        print(f"Layout: Original | {target_emotions[0]}({step_counts}) | {target_emotions[1]}({step_counts}) | {target_emotions[2]}({step_counts})")
        print(f"Step counts: {step_counts}")

if __name__ == "__main__":
    test_sampling_comparison()