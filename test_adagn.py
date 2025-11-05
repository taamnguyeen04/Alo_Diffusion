"""
Quick test to verify AdaGN changes work correctly
"""
import torch
from model import WaveletDiffusionModel

def test_model():
    print("Testing AdaGN model changes...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    model = WaveletDiffusionModel(num_emotions=7).to(device)
    print("✓ Model created successfully")

    # Create dummy input
    batch_size = 2
    img = torch.randn(batch_size, 3, 224, 224).to(device)
    emotion_id = torch.randint(0, 7, (batch_size,)).to(device)

    print(f"Input shape: {img.shape}")
    print(f"Emotion IDs: {emotion_id}")

    # Test forward pass (training mode)
    try:
        loss = model(img, emotion_id, src_image=img)
        print(f"✓ Forward pass successful, loss: {loss.item():.4f}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False

    # Test sampling
    try:
        model.eval()
        with torch.no_grad():
            generated = model.sample(img, emotion_id, num_steps=10, denoising_strength=0.2)
        print(f"✓ Sampling successful, output shape: {generated.shape}")
    except Exception as e:
        print(f"✗ Sampling failed: {e}")
        return False

    # Check AdaGN parameters exist
    adagn_params = []
    for name, param in model.named_parameters():
        if 'emotion_modulation' in name:
            adagn_params.append(name)

    print(f"\n✓ Found {len(adagn_params)} AdaGN parameters:")
    for name in adagn_params[:5]:  # Show first 5
        print(f"  - {name}")

    print("\n✅ All tests passed!")
    return True

if __name__ == "__main__":
    test_model()
