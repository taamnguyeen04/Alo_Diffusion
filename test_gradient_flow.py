#!/usr/bin/env python3
"""
Test script to check gradient flow and attention maps in WaveletDiffusionModel
"""

import torch
import torch.nn as nn
from model import WaveletDiffusionModel

def test_gradient_flow():
    """Test if gradients flow back to UNet properly"""
    print("Testing gradient flow and attention maps...")

    # Create model
    model = WaveletDiffusionModel(num_emotions=7)
    model.train()

    # Create dummy data
    batch_size = 2
    height, width = 64, 64  # Small size for testing

    # Input image (RGB)
    x = torch.randn(batch_size, 3, height, width)

    # Emotion IDs
    emotion_id = torch.randint(0, 7, (batch_size,))

    # Source image for residual connections
    src_image = torch.randn(batch_size, 3, height, width)

    print(f"Input shape: {x.shape}")
    print(f"Emotion IDs: {emotion_id}")
    print()

    # Forward pass
    print("=== Forward pass ===")
    loss = model(x, emotion_id, src_image)
    print(f"Loss: {loss.item():.6f}")
    print()

    # Backward pass
    print("=== Backward pass ===")
    loss.backward()
    print("Backward pass completed")
    print()

    # Check if gradients exist in key layers
    print("=== Gradient check ===")

    # Check UNet input conv gradients
    if model.unet.input_conv.weight.grad is not None:
        grad_mean = model.unet.input_conv.weight.grad.abs().mean().item()
        print(f"UNet input_conv weight grad mean: {grad_mean:.6f}")
    else:
        print("UNet input_conv weight grad is None!")

    # Check emotion projection gradients
    if hasattr(model.unet, 'emotion_projection') and model.unet.emotion_projection is not None:
        if model.unet.emotion_projection.weight.grad is not None:
            grad_mean = model.unet.emotion_projection.weight.grad.abs().mean().item()
            print(f"Emotion projection grad mean: {grad_mean:.6f}")
        else:
            print("Emotion projection grad is None!")

    # Check first encoder block gradients
    first_encoder = model.unet.encoder_blocks[0][0]
    if first_encoder.conv1.weight.grad is not None:
        grad_mean = first_encoder.conv1.weight.grad.abs().mean().item()
        print(f"First encoder conv1 grad mean: {grad_mean:.6f}")
    else:
        print("First encoder conv1 grad is None!")

    # Check cross-attention gradients if present
    if hasattr(first_encoder, 'cross_attn'):
        if first_encoder.cross_attn.to_q.weight.grad is not None:
            grad_mean = first_encoder.cross_attn.to_q.weight.grad.abs().mean().item()
            print(f"Cross-attention to_q grad mean: {grad_mean:.6f}")
        else:
            print("Cross-attention to_q grad is None!")

    print()
    print("=== Test completed ===")

    # Interpretation guide
    print("\n=== Interpretation Guide ===")
    print("Gradient flow:")
    print("- Grad mean > 1e-6: Good gradient flow")
    print("- Grad mean < 1e-8: Very weak/no gradient flow")
    print("- Grad mean = 0: No gradient flow (problematic)")
    print()
    print("Attention maps:")
    print("- Attention mean ~0.14: Uniform attention (not learning)")
    print("- Attention mean varying: Model is learning to use conditions")

if __name__ == "__main__":
    test_gradient_flow()