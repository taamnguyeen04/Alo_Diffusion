import torch
import torch.nn.functional as F
from torchvision.transforms import Resize, ToTensor, Compose, Normalize
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from model import WaveletDiffusionModel, DWT, IWT
from criterions import Emotion_model

class WaveletDiffusionInference:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.labels = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"]
        
        # Load model
        self.model = WaveletDiffusionModel(num_emotions=len(self.labels)).to(self.device)
        
        # Load checkpoint
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {model_path}")
        else:
            print(f"Warning: Model file {model_path} not found. Using random weights.")
        
        self.model.eval()
        
        # Initialize wavelet transforms
        self.dwt = DWT().to(self.device)
        self.iwt = IWT().to(self.device)
        
        # Sử dụng ImageNet normalization thống nhất
        self.transform = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet norm
        ])
        
        # Transform ngược để hiển thị
        self.denormalize = Compose([
            Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                     std=[1/0.229, 1/0.224, 1/0.225])
        ])
        
        # Initialize emotion model for validation
        try:
            self.emotion_model = Emotion_model()
            print("Emotion validation model loaded successfully")
        except:
            print("Warning: Could not load emotion validation model")
            self.emotion_model = None
    
    def load_image(self, image_path):
        """Load and preprocess image"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        return image_tensor, image
    
    def visualize_wavelet_components(self, image_tensor, title="Wavelet Components"):
        """Visualize wavelet decomposition components"""
        with torch.no_grad():
            wavelet_coeffs = self.dwt(image_tensor)
            
            # Split into LL, LH, HL, HH components
            C = image_tensor.shape[1]
            ll = wavelet_coeffs[:, :C, :, :]      # Low-Low (approximation)
            lh = wavelet_coeffs[:, C:2*C, :, :]   # Low-High (horizontal details)
            hl = wavelet_coeffs[:, 2*C:3*C, :, :] # High-Low (vertical details)
            hh = wavelet_coeffs[:, 3*C:, :, :]    # High-High (diagonal details)
            
            # Reconstruct image from wavelet
            reconstructed = self.iwt(wavelet_coeffs)
            
            # Convert to displayable format
            def to_display(tensor):
                img = tensor.squeeze(0).cpu()
                img = (img + 1) / 2  # [-1,1] -> [0,1]
                img = torch.clamp(img, 0, 1)
                return img.permute(1, 2, 0).numpy()
            
            # Create visualization
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(title, fontsize=16)
            
            # Original image
            axes[0, 0].imshow(to_display(image_tensor))
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            # Wavelet components
            axes[0, 1].imshow(to_display(ll), cmap='gray')
            axes[0, 1].set_title('LL (Approximation)')
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(to_display(lh), cmap='gray')
            axes[0, 2].set_title('LH (Horizontal Details)')
            axes[0, 2].axis('off')
            
            axes[1, 0].imshow(to_display(hl), cmap='gray')
            axes[1, 0].set_title('HL (Vertical Details)')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(to_display(hh), cmap='gray')
            axes[1, 1].set_title('HH (Diagonal Details)')
            axes[1, 1].axis('off')
            
            # Reconstructed image
            axes[1, 2].imshow(to_display(reconstructed))
            axes[1, 2].set_title('Reconstructed')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            return fig
    
    def get_intermediate_outputs(self, src_image, target_emotion_id, num_steps=50, denoising_strength=0.3):
        """Get intermediate outputs from each major block in the model"""
        with torch.no_grad():
            device = src_image.device
            B = src_image.shape[0]
            src_wavelet = self.dwt(src_image)
            
            # Calculate start timestep
            start_timestep = int(denoising_strength * self.model.num_timesteps)
            start_timestep = max(1, start_timestep)
            
            # Create timesteps
            timesteps = torch.linspace(start_timestep - 1, 0,
                                     min(num_steps, start_timestep),
                                     dtype=torch.long, device=device)
            
            # Add noise to source image
            if start_timestep > 0:
                noise = torch.randn_like(src_wavelet)
                alpha_start = self.model.sqrt_alphas_cumprod[start_timestep]
                sigma_start = self.model.sqrt_one_minus_alphas_cumprod[start_timestep]
                x = alpha_start * src_wavelet + sigma_start * noise
            else:
                x = src_wavelet
            
            # Store intermediate results
            intermediate_results = {
                'initial_noisy': self.iwt(x.clone()),
                'timesteps': [],
                'denoised_steps': [],
                'wavelet_steps': [],
                'unet_features': []
            }
            
            # Sample with intermediate collection
            for i, t in enumerate(timesteps):
                t_tensor = torch.full((B,), t.item(), device=device, dtype=torch.long)
                
                # Get UNet prediction and intermediate features
                noise_pred = self.model.unet(x, t_tensor, target_emotion_id, src_image)
                
                # Calculate denoised result
                alpha_t = self.model.alphas_cumprod[t.item()]
                alpha_prev = self.model.alphas_cumprod[timesteps[i+1].item()] if i < len(timesteps) - 1 else torch.tensor(1.0, device=device)
                alpha_t = alpha_t.to(device)
                alpha_prev = alpha_prev.to(device)
                
                pred_x0 = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
                pred_x0 = torch.clamp(pred_x0, -3, 3)
                
                # Store intermediate results every few steps
                if i % max(1, len(timesteps) // 8) == 0 or i == len(timesteps) - 1:
                    intermediate_results['timesteps'].append(t.item())
                    intermediate_results['denoised_steps'].append(self.iwt(pred_x0.clone()))
                    intermediate_results['wavelet_steps'].append(pred_x0.clone())
                
                # Update x for next iteration
                if i < len(timesteps) - 1:
                    x = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1 - alpha_prev) * noise_pred
                else:
                    x = pred_x0
            
            # Final result
            final_result = self.iwt(x)
            final_result = torch.clamp(final_result, -1, 1)
            
            intermediate_results['final_result'] = final_result
            
            return intermediate_results
    
    def visualize_sampling_process(self, intermediate_results, emotion_name):
        """Visualize the sampling process"""
        denoised_steps = intermediate_results['denoised_steps']
        timesteps = intermediate_results['timesteps']
        
        # Create grid visualization
        n_steps = len(denoised_steps)
        cols = min(4, n_steps)
        rows = (n_steps + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle(f'Denoising Process - Target Emotion: {emotion_name}', fontsize=16)
        
        for i, (step_img, timestep) in enumerate(zip(denoised_steps, timesteps)):
            row = i // cols
            col = i % cols
            
            # Convert to displayable format
            img = step_img.squeeze(0).cpu()
            img = (img + 1) / 2  # [-1,1] -> [0,1]
            img = torch.clamp(img, 0, 1)
            img = img.permute(1, 2, 0).numpy()
            
            axes[row, col].imshow(img)
            axes[row, col].set_title(f'Step {i+1}\nTimestep: {timestep}')
            axes[row, col].axis('off')
        
        # Hide unused subplots
        for i in range(n_steps, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        return fig
    
    def analyze_frequency_changes(self, src_image, result_image):
        """Analyze frequency domain changes"""
        with torch.no_grad():
            src_wavelet = self.dwt(src_image)
            result_wavelet = self.dwt(result_image)
            
            C = src_image.shape[1]
            
            # Split components
            src_ll = src_wavelet[:, :C, :, :]
            src_hi = src_wavelet[:, C:, :, :]
            result_ll = result_wavelet[:, :C, :, :]
            result_hi = result_wavelet[:, C:, :, :]
            
            # Calculate differences
            ll_diff = torch.abs(result_ll - src_ll).mean().item()
            hi_diff = torch.abs(result_hi - src_hi).mean().item()
            
            # Visualize frequency analysis
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            fig.suptitle('Frequency Domain Analysis', fontsize=16)
            
            def to_display(tensor):
                img = tensor.squeeze(0).cpu()
                img = (img + 1) / 2
                img = torch.clamp(img, 0, 1)
                return img.permute(1, 2, 0).numpy()
            
            # Source components
            axes[0, 0].imshow(to_display(src_ll), cmap='gray')
            axes[0, 0].set_title('Source LL')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(to_display(src_hi[:, :3, :, :]), cmap='gray')
            axes[0, 1].set_title('Source High Freq')
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(to_display(src_image))
            axes[0, 2].set_title('Source Image')
            axes[0, 2].axis('off')
            
            # Result components
            axes[1, 0].imshow(to_display(result_ll), cmap='gray')
            axes[1, 0].set_title('Result LL')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(to_display(result_hi[:, :3, :, :]), cmap='gray')
            axes[1, 1].set_title('Result High Freq')
            axes[1, 1].axis('off')
            
            axes[1, 2].imshow(to_display(result_image))
            axes[1, 2].set_title('Result Image')
            axes[1, 2].axis('off')
            
            # Difference visualization
            ll_diff_vis = torch.abs(result_ll - src_ll)
            hi_diff_vis = torch.abs(result_hi - src_hi)
            
            axes[0, 3].imshow(to_display(ll_diff_vis), cmap='hot')
            axes[0, 3].set_title(f'LL Diff (avg: {ll_diff:.4f})')
            axes[0, 3].axis('off')
            
            axes[1, 3].imshow(to_display(hi_diff_vis[:, :3, :, :]), cmap='hot')
            axes[1, 3].set_title(f'High Freq Diff (avg: {hi_diff:.4f})')
            axes[1, 3].axis('off')
            
            plt.tight_layout()
            return fig, ll_diff, hi_diff
    
    def predict_emotion(self, image_tensor):
        """Predict emotion using the validation model"""
        if self.emotion_model is None:
            return "Unknown", 0.0
        
        try:
            # Convert to format expected by emotion model
            img_norm = (image_tensor + 1) / 2  # [-1,1] -> [0,1]
            img_resized = F.interpolate(img_norm, size=(224, 224), mode='bilinear')
            
            # ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(image_tensor.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(image_tensor.device)
            img_normalized = (img_resized - mean) / std
            
            with torch.no_grad():
                out, _, _ = self.emotion_model.model(img_normalized)
                probabilities = F.softmax(out, dim=1)
                confidence, pred = torch.max(probabilities, 1)
                
                emotion_label = self.labels[pred.item()]
                confidence_score = confidence.item()
                
                return emotion_label, confidence_score
        except Exception as e:
            print(f"Error in emotion prediction: {e}")
            return "Unknown", 0.0
    
    def inference(self, image_path, target_emotion, num_steps=50, 
                 denoising_strength=0.3, color_preservation=0.7, save_dir="inference_results"):
        """Main inference function"""
        # Create output directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Load image
        image_tensor, original_pil = self.load_image(image_path)
        
        # Get target emotion ID
        if isinstance(target_emotion, str):
            if target_emotion.capitalize() in self.labels:
                target_emotion_id = self.labels.index(target_emotion.capitalize())
            else:
                raise ValueError(f"Unknown emotion: {target_emotion}. Available: {self.labels}")
        else:
            target_emotion_id = target_emotion
        
        target_emotion_tensor = torch.tensor([target_emotion_id], device=self.device)
        target_emotion_name = self.labels[target_emotion_id]
        
        print(f"Processing image: {image_path}")
        print(f"Target emotion: {target_emotion_name} (ID: {target_emotion_id})")
        
        # Predict original emotion
        orig_emotion, orig_confidence = self.predict_emotion(image_tensor)
        print(f"Original emotion: {orig_emotion} (confidence: {orig_confidence:.3f})")
        
        # 1. Visualize wavelet decomposition of original image
        print("1. Analyzing wavelet decomposition...")
        wavelet_fig = self.visualize_wavelet_components(image_tensor, "Original Image Wavelet Analysis")
        wavelet_fig.savefig(f"{save_dir}/01_wavelet_decomposition.png", dpi=150, bbox_inches='tight')
        plt.close(wavelet_fig)
        
        # 2. Get intermediate outputs during sampling
        print("2. Running inference with intermediate outputs...")
        intermediate_results = self.get_intermediate_outputs(
            image_tensor, target_emotion_tensor, num_steps, denoising_strength
        )
        
        # 3. Visualize sampling process
        print("3. Visualizing sampling process...")
        sampling_fig = self.visualize_sampling_process(intermediate_results, target_emotion_name)
        sampling_fig.savefig(f"{save_dir}/02_sampling_process.png", dpi=150, bbox_inches='tight')
        plt.close(sampling_fig)
        
        # 4. Analyze frequency changes
        print("4. Analyzing frequency domain changes...")
        final_result = intermediate_results['final_result']
        freq_fig, ll_diff, hi_diff = self.analyze_frequency_changes(image_tensor, final_result)
        freq_fig.savefig(f"{save_dir}/03_frequency_analysis.png", dpi=150, bbox_inches='tight')
        plt.close(freq_fig)
        
        # 5. Predict final emotion
        final_emotion, final_confidence = self.predict_emotion(final_result)
        print(f"Final emotion: {final_emotion} (confidence: {final_confidence:.3f})")
        
        # 6. Save comparison image
        print("5. Saving final comparison...")
        comparison_tensor = torch.cat([image_tensor, final_result], dim=0)
        comparison_tensor = (comparison_tensor + 1) / 2  # [-1,1] -> [0,1]
        save_image(comparison_tensor, f"{save_dir}/04_comparison.png", nrow=2, normalize=False)
        
        # 7. Create summary visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original
        orig_img = (image_tensor.squeeze(0).cpu() + 1) / 2
        orig_img = torch.clamp(orig_img, 0, 1).permute(1, 2, 0).numpy()
        axes[0].imshow(orig_img)
        axes[0].set_title(f'Original\n{orig_emotion} ({orig_confidence:.3f})')
        axes[0].axis('off')
        
        # Result
        result_img = (final_result.squeeze(0).cpu() + 1) / 2
        result_img = torch.clamp(result_img, 0, 1).permute(1, 2, 0).numpy()
        axes[1].imshow(result_img)
        axes[1].set_title(f'Target: {target_emotion_name}\nPredicted: {final_emotion} ({final_confidence:.3f})')
        axes[1].axis('off')
        
        plt.suptitle(f'Emotion Transfer Results\nLL Change: {ll_diff:.4f}, High Freq Change: {hi_diff:.4f}')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/05_summary.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nResults saved to: {save_dir}")
        print(f"- Wavelet decomposition: 01_wavelet_decomposition.png")
        print(f"- Sampling process: 02_sampling_process.png")
        print(f"- Frequency analysis: 03_frequency_analysis.png")
        print(f"- Comparison: 04_comparison.png")
        print(f"- Summary: 05_summary.png")
        
        return {
            'original_emotion': orig_emotion,
            'original_confidence': orig_confidence,
            'target_emotion': target_emotion_name,
            'final_emotion': final_emotion,
            'final_confidence': final_confidence,
            'll_change': ll_diff,
            'hi_freq_change': hi_diff,
            'final_image': final_result
        }

if __name__ == "__main__":
    # Sử dụng trực tiếp không cần command line arguments
    inferencer = WaveletDiffusionInference('best_model.pt')
    
    # Thay đổi các tham số này theo ý muốn
    test_image = r"C:\Users\tam\Documents\data\FEG\Manually_Annotated_Images\Manually_Annotated_Images\1\4e5906ae29e54d80d6334e902b9230b8fb0c30309b5f76e3dca82b66.JPG"  # Đường dẫn ảnh test
    target_emotion = "happy"       # Cảm xúc mục tiêu
    num_steps = 50                 # Số bước sampling
    denoising_strength = 0.3       # Độ mạnh denoising
    output_dir = "inference_results"  # Thư mục output
    
    if os.path.exists(test_image):
        results = inferencer.inference(
            image_path=test_image,
            target_emotion=target_emotion,
            num_steps=num_steps,
            denoising_strength=denoising_strength,
            save_dir=output_dir
        )
        print("\nInference completed successfully!")
        print(f"Original: {results['original_emotion']} -> Target: {results['target_emotion']} -> Final: {results['final_emotion']}")
    else:
        print(f"Test image {test_image} not found.")
        print("Vui lòng thay đổi đường dẫn 'test_image' trong code.")
