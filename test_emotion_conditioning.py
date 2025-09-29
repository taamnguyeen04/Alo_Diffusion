#!/usr/bin/env python3
"""
Test script to verify emotion conditioning strength with FiLM and embedding
"""

import torch
import torch.nn.functional as F
from model import WaveletDiffusionModel

def test_emotion_conditioning():
    """Test if emotion conditioning has stronger effect now"""
    print("Testing enhanced emotion conditioning...")

    model = WaveletDiffusionModel(num_emotions=7)
    model.eval()

    batch_size = 4
    height, width = 64, 64

    # Create identical input images
    x = torch.randn(1, 3, height, width).repeat(batch_size, 1, 1, 1)
    src_image = torch.randn(1, 3, height, width).repeat(batch_size, 1, 1, 1)

    # Different emotion IDs for each sample
    emotion_ids = torch.tensor([0, 1, 2, 6])  # Different emotions

    print(f"Input shape: {x.shape}")
    print(f"Emotion IDs: {emotion_ids}")
    print(f"All images are identical: {torch.allclose(x[0], x[1])}")
    print()

    with torch.no_grad():
        # Test emotion embedding differences
        print("=== Emotion Embedding Analysis ===")
        emotion_embeddings = model.unet.emotion_embedding(emotion_ids)

        print(f"Emotion embedding shape: {emotion_embeddings.shape}")
        print(f"Embedding magnitude range: {emotion_embeddings.abs().min().item():.4f} - {emotion_embeddings.abs().max().item():.4f}")

        # Check if embeddings are different
        embedding_distances = []
        for i in range(len(emotion_ids)):
            for j in range(i+1, len(emotion_ids)):
                dist = F.cosine_similarity(emotion_embeddings[i:i+1], emotion_embeddings[j:j+1]).item()
                embedding_distances.append(dist)
                print(f"Cosine similarity emotion {emotion_ids[i]} vs {emotion_ids[j]}: {dist:.4f}")

        avg_similarity = sum(embedding_distances) / len(embedding_distances)
        print(f"Average embedding similarity: {avg_similarity:.4f}")
        print("(Lower similarity = more distinct emotions)")
        print()

        # Test model outputs with different emotions
        print("=== Model Output Analysis ===")

        # Sample with different emotions
        outputs = []
        for i, emotion_id in enumerate(emotion_ids):
            output = model.sample(
                src_image[i:i+1],
                emotion_id.unsqueeze(0),
                num_steps=5,  # Few steps for testing
                denoising_strength=0.5
            )
            outputs.append(output)
            print(f"Emotion {emotion_id.item()}: output range [{output.min().item():.3f}, {output.max().item():.3f}]")

        # Compare output differences
        print()
        print("=== Output Difference Analysis ===")
        output_tensor = torch.cat(outputs, dim=0)

        output_differences = []
        for i in range(len(emotion_ids)):
            for j in range(i+1, len(emotion_ids)):
                diff = F.mse_loss(output_tensor[i], output_tensor[j]).item()
                output_differences.append(diff)
                print(f"MSE between emotion {emotion_ids[i]} and {emotion_ids[j]}: {diff:.6f}")

        avg_output_diff = sum(output_differences) / len(output_differences)
        print(f"Average output difference: {avg_output_diff:.6f}")
        print()

        # Test FiLM parameter ranges
        print("=== FiLM Parameter Analysis ===")
        # Forward pass through a ResBlock to see FiLM parameters
        test_resblock = model.unet.encoder_blocks[0][0]

        # Get FiLM parameters for different emotions
        for i, emotion_id in enumerate(emotion_ids[:2]):  # Test first 2 emotions
            emb = model.unet.emotion_embedding(emotion_id.unsqueeze(0))
            gamma = test_resblock.film_gamma(F.silu(emb))
            beta = test_resblock.film_beta(F.silu(emb))

            print(f"Emotion {emotion_id.item()}:")
            print(f"  Gamma range: [{gamma.min().item():.4f}, {gamma.max().item():.4f}]")
            print(f"  Beta range: [{beta.min().item():.4f}, {beta.max().item():.4f}]")
            print(f"  Gamma mean: {gamma.mean().item():.4f}")
            print(f"  Beta mean: {beta.mean().item():.4f}")

    print()
    print("=== Interpretation ===")
    print("Good emotion conditioning:")
    print("- Low embedding similarity (<0.5)")
    print("- High output differences (>0.001)")
    print("- FiLM gamma around 1.0, beta around 0.0")
    print("- Varied FiLM parameters across emotions")

if __name__ == "__main__":
    test_emotion_conditioning()