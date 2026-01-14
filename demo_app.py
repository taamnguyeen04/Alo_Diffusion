"""
python demo_app.py --mode image --path "eval_data/source/42_354.png" --emotion anger
python demo_app.py --mode webcam --emotion anger
"""
import torch
import cv2
import argparse
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from model import WaveletDiffusionModel
from xai_utils import denormalize_image, tensor_to_numpy

EMOTIONS = {
    'neutral': 0,
    'happy': 1,
    'sad': 2,
    'surprise': 3,
    'fear': 4,
    'disgust': 5,
    'anger': 6
}

REVERSE_EMOTIONS = {v: k for k, v in EMOTIONS.items()}

class DemoApp:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Loading model on {self.device}...")
        
        self.model = WaveletDiffusionModel(
            num_emotions=7, 
            use_film=True, 
            use_adagn=False
        ).to(self.device)
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
            
        self.model.eval()
        
        # Optimize 1: FP16
        if self.device.type == 'cuda':
            print("Enabling FP16 (Half Precision) for speed...")
            self.model.half()
            
        print("Model loaded successfully!")
        
        # Transform for model input
        self.transform = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_face(self, frame):
        """Detect largest face in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None
            
        # Return largest face
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        return largest_face

    def process_face(self, face_img, emotion_id, num_steps=30):
        """Run diffusion on a face crop"""
        # Convert CV2 (BGR) to PIL (RGB)
        pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        
        # Preprocess
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        # FP16 Cast
        if self.device.type == 'cuda':
            input_tensor = input_tensor.half()
        
        # Generate
        with torch.no_grad():
            generated = self.model.sample(
                input_tensor,
                torch.tensor([emotion_id], device=self.device),
                num_steps=num_steps,
                denoising_strength=0.4
            )
            
        # Postprocess
        # generated is likely fp16, convert to float32 for numpy
        gen_tensor = generated[0].float()
        gen_np = tensor_to_numpy(denormalize_image(gen_tensor))
        
        # Convert back to BGR for OpenCV
        return cv2.cvtColor(gen_np, cv2.COLOR_RGB2BGR)

    def run_image_mode(self, image_path, emotion_name):
        print(f"\nrunning IMAGE mode: {image_path} -> {emotion_name}")
        emotion_id = EMOTIONS.get(emotion_name.lower())
        if emotion_id is None:
            print(f"Invalid emotion. Choose from: {list(EMOTIONS.keys())}")
            return

        frame = cv2.imread(image_path)
        if frame is None:
            print("Could not read image.")
            return

        face_rect = self.detect_face(frame)
        
        if face_rect is not None:
            x, y, w, h = face_rect
            print(f"Face detected at {face_rect}")
            
            # Crop with margin
            margin = int(w * 0.2)
            y1 = max(0, y - margin)
            y2 = min(frame.shape[0], y + h + margin)
            x1 = max(0, x - margin)
            x2 = min(frame.shape[1], x + w + margin)
            
            face_crop = frame[y1:y2, x1:x2]
            processed_crop = self.process_face(face_crop, emotion_id, num_steps=50) # High quality for image
            
            # Resize processed crop back to original slot size
            processed_crop = cv2.resize(processed_crop, (x2-x1, y2-y1))
            
            # Paste back
            result = frame.copy()
            result[y1:y2, x1:x2] = processed_crop
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            print("No face detected, processing whole image...")
            processed_img = self.process_face(frame, emotion_id, num_steps=50)
            result = cv2.resize(processed_img, (frame.shape[1], frame.shape[0]))

        # Display
        cv2.imshow(f"Result ({emotion_name})", result)
        print("Press any key to close window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        save_path = f"demo_output_{emotion_name}.png"
        cv2.imwrite(save_path, result)
        print(f"Saved result to {save_path}")

    def run_webcam_mode(self, emotion_name):
        print(f"\nrunning WEBCAM mode -> target: {emotion_name}")
        print("Controls:")
        print("  'q': Quit")
        print("  's': Snapshot (Process current frame once)")
        print("  'c': Toggle Continuous Mode (Process every frame)")
        print("  '+': Increase Steps (Better Quality, Slower)")
        print("  '-': Decrease Steps (Faster, Lower Quality)")
        
        emotion_id = EMOTIONS.get(emotion_name.lower())
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Could not open webcam")
            return

        continuous = False
        current_steps = 10 # Default for speed
        import time
        fps_list = []
        
        while True:
            # start_time = time.time()
            ret, frame = cap.read()
            if not ret: break

            # Draw rectangle around face
            face_rect = self.detect_face(frame)
            display_frame = frame.copy()
            
            mode_str = 'CONTINUOUS' if continuous else 'SNAPSHOT'
            status_text = f"Target: {emotion_name.upper()} | Mode: {mode_str} | Steps: {current_steps}"
            
            if face_rect is not None:
                x, y, w, h = face_rect
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Continuous Processing Logic
                if continuous:
                    # Crop with margin
                    margin = int(w * 0.2)
                    y1 = max(0, y - margin)
                    y2 = min(frame.shape[0], y + h + margin)
                    x1 = max(0, x - margin)
                    x2 = min(frame.shape[1], x + w + margin)
                    
                    face_crop = frame[y1:y2, x1:x2]
                    
                    # Process
                    t0 = time.time()
                    processed_crop = self.process_face(face_crop, emotion_id, num_steps=current_steps)
                    dt = time.time() - t0
                    
                    fps = 1.0 / (dt + 1e-9)
                    fps_list.append(fps)
                    if len(fps_list) > 10: fps_list.pop(0)
                    avg_fps = sum(fps_list) / len(fps_list)
                    
                    cv2.imshow(f'Processed Result ({emotion_name})', processed_crop)
                    
                    cv2.putText(display_frame, f"Inference FPS: {avg_fps:.2f}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            else:
                status_text += " (No Face)"

            cv2.putText(display_frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Wavelet Diffusion Demo', display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                if face_rect is None:
                    print("No face to process!")
                    continue
                
                print(f"Processing one frame with {current_steps} steps...")
                x, y, w, h = face_rect
                margin = int(w * 0.2)
                y1 = max(0, y - margin)
                y2 = min(frame.shape[0], y + h + margin)
                x1 = max(0, x - margin)
                x2 = min(frame.shape[1], x + w + margin)
                face_crop = frame[y1:y2, x1:x2]
                processed_crop = self.process_face(face_crop, emotion_id, num_steps=current_steps)
                cv2.imshow(f'Processed Result ({emotion_name})', processed_crop)
                print("Done!")
                
            elif key == ord('c'):
                continuous = not continuous
                print(f"Continuous mode: {continuous}")
                if not continuous:
                    cv2.destroyWindow(f'Processed Result ({emotion_name})')
            
            elif key == ord('='): # '+' key usually requires shift, '=' is the unshifted key
                current_steps = min(current_steps + 1, 100)
                print(f"Steps increased to: {current_steps}")
            
            elif key == ord('-'):
                current_steps = max(current_steps - 1, 1)
                print(f"Steps decreased to: {current_steps}")

        cap.release()
        cv2.destroyAllWindows()

import os
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wavelet Diffusion Demo")
    parser.add_argument('--mode', type=str, required=True, choices=['image', 'webcam'], help='Mode: image or webcam')
    parser.add_argument('--path', type=str, help='Path to image (required for image mode)')
    parser.add_argument('--emotion', type=str, required=True, help='Target emotion (neutral, happy, sad, surprise, fear, disgust, anger)')
    parser.add_argument('--checkpoint', type=str, default=r"WaveletDiffusion_phase1/model/best_model.pt", help='Model checkpoint path')
    
    args = parser.parse_args()
    
    try:
        app = DemoApp(args.checkpoint)
        
        if args.mode == 'image':
            if not args.path:
                print("Error: --path is required for image mode")
            else:
                app.run_image_mode(args.path, args.emotion)
        elif args.mode == 'webcam':
            app.run_webcam_mode(args.emotion)
            
    except Exception as e:
        print(f"Error: {e}")
