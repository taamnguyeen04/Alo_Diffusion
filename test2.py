import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math, os, pandas as pd
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
# ================================================================
# 1️⃣ LỚP DWT (Discrete Wavelet Transform)
# ================================================================
class DWT(nn.Module):
    """Biến đổi wavelet rời rạc (DWT) + Chuẩn hoá đầu ra"""
    def __init__(self, normalize=True, mean_std=None):
        """
        Args:
            normalize (bool): Bật/tắt chuẩn hoá.
            mean_std (dict or None): Nếu cung cấp, dùng giá trị mean/std cố định
                {'LL': (mean, std), 'LH': (...), 'HL': (...), 'HH': (...)}.
                Nếu None, sẽ chuẩn hoá theo batch động.
        """
        super(DWT, self).__init__()
        self.low = torch.tensor([1., 1.]) / math.sqrt(2)
        self.high = torch.tensor([1., -1.]) / math.sqrt(2)
        self.normalize = normalize
        self.mean_std = mean_std

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, "Height and Width must be even"

        low = self.low.to(x.device)
        high = self.high.to(x.device)

        ll = torch.outer(low, low).unsqueeze(0).unsqueeze(0)
        lh = torch.outer(low, high).unsqueeze(0).unsqueeze(0)
        hl = torch.outer(high, low).unsqueeze(0).unsqueeze(0)
        hh = torch.outer(high, high).unsqueeze(0).unsqueeze(0)

        ll = ll.repeat(C, 1, 1, 1)
        lh = lh.repeat(C, 1, 1, 1)
        hl = hl.repeat(C, 1, 1, 1)
        hh = hh.repeat(C, 1, 1, 1)

        # Biến đổi
        x_ll = F.conv2d(x, ll, stride=2, groups=C)
        x_lh = F.conv2d(x, lh, stride=2, groups=C)
        x_hl = F.conv2d(x, hl, stride=2, groups=C)
        x_hh = F.conv2d(x, hh, stride=2, groups=C)

        # ====== 🔹 Bước CHUẨN HOÁ ======
        if self.normalize:
            if self.mean_std is not None:
                # 🔸 Chuẩn hoá theo giá trị thống kê cố định (toàn dataset)
                for i, (name, tensor) in enumerate(zip(
                    ['LL', 'LH', 'HL', 'HH'],
                    [x_ll, x_lh, x_hl, x_hh]
                )):
                    mean, std = self.mean_std[name]
                    tensor.sub_(mean).div_(std + 1e-8)
            else:
                # 🔸 Chuẩn hoá động theo batch (z-score)
                for tensor in [x_ll, x_lh, x_hl, x_hh]:
                    mean = tensor.mean(dim=[1,2,3], keepdim=True)
                    std = tensor.std(dim=[1,2,3], keepdim=True)
                    tensor.sub_(mean).div_(std + 1e-8)

        return torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)



# ================================================================
# 2️⃣ DATASET AffectNet
# ================================================================
class Affectnet(Dataset):
    def __init__(self, is_train=True, transform=None, root=None):
        root = "C:/Users/tam/Desktop/Data/FEG"
        image_path = os.path.join(root, "Manually_Annotated_Images")
        self.transform = transform

        label_path = os.path.join(root, "training.csv" if is_train else "validation.csv")
        list_label = pd.read_csv(label_path)

        # chỉ lấy các nhãn hợp lệ (0–6)
        valid_labels = list_label[list_label['expression'] < 7].copy()

        # tạo đường dẫn đầy đủ tới ảnh
        valid_labels['full_image_path'] = valid_labels['subDirectory_filePath'].apply(
            lambda x: os.path.join(image_path, x)
        )

        # lọc các ảnh tồn tại thực sự
        valid_labels = valid_labels[valid_labels['full_image_path'].apply(os.path.isfile)]

        self.list_image = valid_labels['full_image_path'].tolist()

    def __len__(self):
        return len(self.list_image)

    def __getitem__(self, item):
        try:
            image = Image.open(self.list_image[item]).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image
        except (FileNotFoundError, OSError):
            next_item = (item + 1) % len(self.list_image)
            return self.__getitem__(next_item)


# ================================================================
# 3️⃣ TÍNH MEAN & STD CHO 4 KÊNH DWT
# ================================================================
def compute_wavelet_stats(dataset, dwt_model, device, max_samples=300):
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    ll_vals, lh_vals, hl_vals, hh_vals = [], [], [], []

    for i, (img,) in enumerate(tqdm(loader, total=min(max_samples, len(dataset)))):
        if i >= max_samples:
            break

        img = img.to(device)
        if img.dim() == 3:
            img = img.unsqueeze(0)  # thêm batch dimension nếu thiếu

        y = dwt_model(img)
        C = y.shape[1] // 4
        y_ll, y_lh, y_hl, y_hh = y[:, :C], y[:, C:2*C], y[:, 2*C:3*C], y[:, 3*C:]

        ll_vals.append(y_ll.detach().cpu().numpy().ravel())
        lh_vals.append(y_lh.detach().cpu().numpy().ravel())
        hl_vals.append(y_hl.detach().cpu().numpy().ravel())
        hh_vals.append(y_hh.detach().cpu().numpy().ravel())

    # nối toàn bộ và tính mean/std
    ll_vals = np.concatenate(ll_vals)
    lh_vals = np.concatenate(lh_vals)
    hl_vals = np.concatenate(hl_vals)
    hh_vals = np.concatenate(hh_vals)

    stats = {
        "LL": (ll_vals.mean(), ll_vals.std()),
        "LH": (lh_vals.mean(), lh_vals.std()),
        "HL": (hl_vals.mean(), hl_vals.std()),
        "HH": (hh_vals.mean(), hh_vals.std())
    }

    return stats

def show_dwt_results(original_tensor, dwt_output, title="DWT Visualization"):
    """
    Hàm tách tensor đầu ra và hiển thị 4 thành phần tần số
    """
    # Tách output (đang nối theo dim 1) thành 4 phần bằng nhau
    # Input shape gốc là C kênh, output là 4*C kênh.
    # Thứ tự trong code DWT là: LL, LH, HL, HH
    parts = torch.chunk(dwt_output, 4, dim=1) 
    names = ["LL (Low-Low)", "LH (Low-High)", "HL (High-Low)", "HH (High-High)"]
    
    plt.figure(figsize=(15, 8))
    
    # Hiển thị ảnh gốc
    plt.subplot(2, 3, 2) # Vị trí giữa dòng trên
    img_np = original_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    # Đưa về range 0-1 để hiển thị chuẩn nếu chưa
    img_np = np.clip(img_np, 0, 1)
    plt.imshow(img_np)
    plt.title("Original Image")
    plt.axis('off')

    # Hiển thị 4 băng tần
    # Vì output đã được normalize (z-score), giá trị có thể âm và > 1.
    # Để hiển thị đẹp bằng plt, ta cần đưa về range [0, 1] (Min-Max Scaling)
    for i, part in enumerate(parts):
        plt.subplot(2, 4, 5 + i) # Các vị trí dòng dưới
        
        # Chuyển sang numpy: (H, W, C)
        band_img = part.detach().squeeze().permute(1, 2, 0).cpu().numpy()
        
        # Min-max scaling cho từng band để nhìn rõ chi tiết
        band_min, band_max = band_img.min(), band_img.max()
        band_display = (band_img - band_min) / (band_max - band_min + 1e-8)
        
        plt.imshow(band_display)
        plt.title(names[i])
        plt.axis('off')
        
        # Thêm thông tin thống kê
        print(f"Band {names[i][:2]}: Min={band_min:.2f}, Max={band_max:.2f}, Mean={band_img.mean():.2f}")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# ================================================================
# 4️⃣ CHẠY DEMO
# ================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mean_std_dict = {
        'LL': (-0.191473, 1.083653),
        'LH': (-0.000004, 0.069750),
        'HL': (-0.000353, 0.062807),
        'HH': (-0.000007, 0.022190),
    }
    dwt = DWT(normalize=True, mean_std=mean_std_dict).to(device)



    # transform = T.Compose([
    #     T.Resize((256, 256)),
    #     T.ToTensor(),
    #     T.Normalize(0.5, 0.5)
    # ])

    # dataset = Affectnet(is_train=True, transform=transform)

    # print(f"Tổng số ảnh hợp lệ: {len(dataset)}")

    # stats = compute_wavelet_stats(dataset, dwt, device, max_samples=300)

    # # in kết quả đẹp
    # print("\n📊 Mean & Std của 4 kênh tần số:")
    # print("-" * 40)
    # for k, (m, s) in stats.items():
    #     print(f"{k:>3}: mean = {m:8.6f} | std = {s:8.6f}")
    # print("-" * 40)

    # # (Tùy chọn) Lưu ra CSV
    # pd.DataFrame(stats, index=["mean", "std"]).to_csv("wavelet_stats.csv")
    # print("✅ Kết quả đã lưu: wavelet_stats.csv")
    img_path = "C:/Users/tam/Pictures/happy.png"  # <--- THAY ĐƯỜNG DẪN ẢNH CỦA BẠN Ở ĐÂY
    
    try:
        img = cv2.imread(img_path)
        if img is None:
            print("Không tìm thấy ảnh, đang tạo ảnh ngẫu nhiên để test...")
            img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 4. Tiền xử lý ảnh
        H, W, C = img.shape
        # Resize về số chẵn gần nhất nếu cần
        new_H = H if H % 2 == 0 else H - 1
        new_W = W if W % 2 == 0 else W - 1
        if new_H != H or new_W != W:
            img = cv2.resize(img, (new_W, new_H))
            
        # Chuyển sang Tensor: (B, C, H, W) và Normalize về [0, 1]
        img_tensor = torch.from_numpy(img).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

        # 5. Forward qua DWT
        with torch.no_grad():
            outputs = dwt(img_tensor)

        # 6. Hiển thị
        show_dwt_results(img_tensor, outputs)
        
    except Exception as e:
        print(f"Có lỗi xảy ra: {e}")