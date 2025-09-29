import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.transforms import Resize, ToTensor, Compose, Normalize
from torch.utils.data import DataLoader
from model import WaveletDiffusionModel
from dataset import Affectnet
import os

def test_denoising_strength():
    """Test with different denoising strength values"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    labels = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"]
    image_size = 224

    # Load model
    model = WaveletDiffusionModel(num_emotions=len(labels)).to(device)

    # Load checkpoint
    checkpoint_path = "WaveletDiffusion/model/best_model.pt"

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {checkpoint_path}")
    else:
        print("Checkpoint not found!")
        return

    # Data transforms
    transform = Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load test data
    val_dataset = Affectnet(root="C:/Users/tam/Documents/data/FEG", is_train=False, transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False)

    # Test with one image
    x_test, expr_test, _, _ = next(iter(val_dataloader))
    x_test = x_test.to(device)
    expr_test = expr_test.to(device)

    # Test with different denoising strength values
    strength_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    target_emotion = 1  # Happy

    model.eval()
    os.makedirs("test_strength_results", exist_ok=True)

    with torch.no_grad():
        all_results = []

        # Original image
        original_img = (x_test.clamp(-1, 1) + 1) / 2
        all_results.append(original_img)

        # Test with different strength values
        for strength in strength_values:
            print(f"Testing with denoising_strength = {strength}")

            emotion_tensor = torch.full((1,), target_emotion, device=device)
            generated = model.sample(
                x_test,
                emotion_tensor,
                num_steps=50,
                denoising_strength=strength
            )

            # Normalize to save image
            generated_norm = (generated.clamp(-1, 1) + 1) / 2
            all_results.append(generated_norm)

            # Save individual images
            save_image(
                generated_norm,
                f"test_strength_results/strength_{strength:.1f}.png",
                normalize=False
            )

        # Save all in one image
        all_imgs = torch.cat(all_results, dim=0)
        save_image(
            all_imgs,
            "test_strength_results/strength_comparison.png",
            nrow=len(strength_values) + 1,  # +1 for original
            normalize=False
        )

        print("Saved test results to 'test_strength_results' directory")
        print("strength_comparison.png contains all comparison results")
        print("From left to right: Original, strength=0.1, 0.3, 0.5, 0.7, 0.9, 1.0")

def test_multiple_emotions():
    """Test denoising strength with multiple emotions"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"]
    image_size = 224

    # Load model
    model = WaveletDiffusionModel(num_emotions=len(labels)).to(device)
    checkpoint_path = "WaveletDiffusion/model/best_model.pt"

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {checkpoint_path}")
    else:
        print("Checkpoint not found!")
        return

    # Data transforms
    transform = Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load test data
    val_dataset = Affectnet(root="C:/Users/tam/Documents/data/FEG", is_train=False, transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False)

    x_test, _, _, _ = next(iter(val_dataloader))
    x_test = x_test.to(device)

    model.eval()
    os.makedirs("test_emotions_strength", exist_ok=True)

    # Test with strength = 0.9
    strength = 0.9

    with torch.no_grad():
        all_results = []

        # Original image
        original_img = (x_test.clamp(-1, 1) + 1) / 2
        all_results.append(original_img)

        # Test with each emotion
        for emotion_id, emotion_name in enumerate(labels):
            print(f"Generating {emotion_name} with strength = {strength}")

            emotion_tensor = torch.full((1,), emotion_id, device=device)
            generated = model.sample(
                x_test,
                emotion_tensor,
                num_steps=50,
                denoising_strength=strength
            )

            generated_norm = (generated.clamp(-1, 1) + 1) / 2
            all_results.append(generated_norm)

            # Save individual images
            save_image(
                generated_norm,
                f"test_emotions_strength/{emotion_name.lower()}_strength_{strength:.1f}.png",
                normalize=False
            )

        # Save all emotions
        all_imgs = torch.cat(all_results, dim=0)
        save_image(
            all_imgs,
            f"test_emotions_strength/all_emotions_strength_{strength:.1f}.png",
            nrow=len(labels) + 1,  # +1 for original
            normalize=False
        )

        print(f"Saved test results to 'test_emotions_strength' directory")
        print(f"all_emotions_strength_{strength:.1f}.png contains all emotions")

if __name__ == "__main__":
    print("=== Test Denoising Strength Values ===")
    test_denoising_strength()

    print("\n=== Test Multiple Emotions with Strength = 0.9 ===")
    test_multiple_emotions()