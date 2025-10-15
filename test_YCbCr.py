from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 1. Mở ảnh khuôn mặt RGB
test_image = r"C:\Users\tam\Documents\data\FEG\Manually_Annotated_Images\Manually_Annotated_Images\1\4e5906ae29e54d80d6334e902b9230b8fb0c30309b5f76e3dca82b66.JPG"  # Đường dẫn ảnh test

img = Image.open(test_image).convert("RGB")   # thay bằng ảnh của bạn
img_np = np.array(img) / 255.0

# 2. Chuyển RGB → YCbCr
img_ycbcr = img.convert("YCbCr")
y, cb, cr = img_ycbcr.split()
y = np.array(y, dtype=np.float32)
cb = np.array(cb, dtype=np.float32)
cr = np.array(cr, dtype=np.float32)

# 3. Tạo “mask” vùng miệng – để demo cho việc thay đổi biểu cảm
H, W = y.shape
yy, xx = np.mgrid[:H, :W]
mask = np.exp(-((xx - W/2)**2 + (yy - H*0.75)**2) / (2*(H*0.15)**2))  # vùng miệng
mask = (mask - mask.min()) / (mask.max() - mask.min())

# 4. Thay đổi Y — làm sáng vùng miệng (như đang cười)
y_mod = y - 80 * mask   # tăng độ sáng vùng miệng
y_mod = np.clip(y_mod, 0, 255).astype(np.uint8)

# 5. Ghép lại và chuyển ngược sang RGB
img_new_ycbcr = Image.merge("YCbCr", [Image.fromarray(y_mod.astype(np.uint8)),
                                      Image.fromarray(cb.astype(np.uint8)),
                                      Image.fromarray(cr.astype(np.uint8))])
img_new = img_new_ycbcr.convert("RGB")

# 6. Hiển thị kết quả
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.title("Ảnh gốc (RGB)")
plt.imshow(img)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Chỉ thay đổi kênh Y (độ sáng)")
plt.imshow(img_new)
plt.axis("off")
plt.show()
