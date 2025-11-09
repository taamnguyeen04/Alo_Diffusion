from moviepy.editor import ImageSequenceClip
import os

folder = "C:/Users/tam/Desktop/Data/FEG/ias/denoising_strenth 06/out"
images = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".png")])
clip = ImageSequenceClip(images, fps=2)
clip.write_videofile("animation.mp4", codec="libx264")
