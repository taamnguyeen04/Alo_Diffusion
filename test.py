import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T

# ==== Lớp DWT và IWT bạn đã có ====
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.low = torch.tensor([1., 1.]) / math.sqrt(2)
        self.high = torch.tensor([1., -1.]) / math.sqrt(2)

    def forward(self, x):
        B, C, H, W = x.shape
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

        x_ll = F.conv2d(x, ll, stride=2, groups=C)
        x_lh = F.conv2d(x, lh, stride=2, groups=C)
        x_hl = F.conv2d(x, hl, stride=2, groups=C)
        x_hh = F.conv2d(x, hh, stride=2, groups=C)

        return torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.low = torch.tensor([1., 1.]) / math.sqrt(2)
        self.high = torch.tensor([1., -1.]) / math.sqrt(2)

    def forward(self, x):
        B, C4, H, W = x.shape
        C = C4 // 4
        x_ll, x_lh, x_hl, x_hh = x[:, :C], x[:, C:2*C], x[:, 2*C:3*C], x[:, 3*C:]

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

        x_ll_up = F.conv_transpose2d(x_ll, ll, stride=2, groups=C)
        x_lh_up = F.conv_transpose2d(x_lh, lh, stride=2, groups=C)
        x_hl_up = F.conv_transpose2d(x_hl, hl, stride=2, groups=C)
        x_hh_up = F.conv_transpose2d(x_hh, hh, stride=2, groups=C)

        return x_ll_up + x_lh_up + x_hl_up + x_hh_up


# ==== Demo ====
device = "cuda" if torch.cuda.is_available() else "cpu"
dwt = DWT().to(device)
iwt = IWT().to(device)

# 1. Đọc ảnh và chuẩn hóa về [-1, 1]
img = Image.open("C:/Users/tam/Desktop/Data/FEG/Manually_Annotated_Images/5/62f335a61d8faab300d922d455031cbcfaed6f57b73633a76de1a99f.jpg").convert("RGB")
transform = T.Compose([
    T.Resize((256,256)),
    T.ToTensor(),             # [0,1]
    T.Normalize(0.5, 0.5)     # [0,1] -> [-1,1]
])
x = transform(img).unsqueeze(0).to(device)

# 2. DWT -> biến đổi nhẹ để vượt khỏi [-1,1]
y = dwt(x)

# # "phá" bằng cách nhân mạnh các tần số cao để làm vượt [-1,1]
# C = y.shape[1] // 4
# y[:, :C] *= 1   # phóng đại HL, HH bands

# # 3. IWT khôi phục lại ảnh
# x_recon = iwt(y)

# # 4. Chuyển về [0,1] để hiển thị
# x_recon_vis = (x_recon.clamp(-2, 2) + 1) / 2  # deliberately cho phép vượt [-1,1]
# x_vis = (x + 1) / 2

# # 5. Hiển thị kết quả
# plt.figure(figsize=(10,4))
# plt.subplot(1,2,1)
# plt.imshow(x_vis[0].permute(1,2,0).cpu())
# plt.title("Trước")
# plt.axis("off")

# plt.subplot(1,2,2)
# plt.imshow(x_recon_vis[0].permute(1,2,0).cpu())
# plt.title("Sau")
# plt.axis("off")

# plt.show()

# # In min/max để bạn thấy rõ biên độ vượt
# print("Giá trị min/max sau IWT:", x_recon.min().item(), x_recon.max().item())
B, C4, H, W = y.shape
C = C4 // 4
y_ll, y_lh, y_hl, y_hh = y[:, :C], y[:, C:2*C], y[:, 2*C:3*C], y[:, 3*C:]

# 4. Hàm tiện ích để vẽ histogram
def plot_histogram(tensor, title):
    data = tensor.detach().cpu().numpy().ravel()
    plt.hist(data, bins=100, color='steelblue', alpha=0.7)
    plt.title(title)
    plt.xlabel('Giá trị pixel')
    plt.ylabel('Tần suất')
    plt.grid(True)

# 5. Vẽ biểu đồ
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plot_histogram(y_ll, "LL (Low-Low)")

plt.subplot(2, 2, 2)
plot_histogram(y_lh, "LH (Low-High)")

plt.subplot(2, 2, 3)
plot_histogram(y_hl, "HL (High-Low)")

plt.subplot(2, 2, 4)
plot_histogram(y_hh, "HH (High-High)")

plt.tight_layout()
plt.show()