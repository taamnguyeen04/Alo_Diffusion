import os
import time
import numpy as np
from PIL import Image
from aim import Run, Image as AimImage

# ============================================================
# ⚙️ 1. Thiết lập kết nối tới AIM Server
# ============================================================
os.environ["AIM_SERVER_URL"] = "http://159.89.193.7:43800"

# Tạo session mới trên AIM (experiment name có thể đổi tùy bạn)
run = Run(experiment="AIM_Connection_Test")

# Ghi siêu tham số
run["hparams"] = {
    "batch_size": 8,
    "learning_rate": 1e-3,
    "epochs": 3,
}

print("✅ Started AIM logging test...")

# ============================================================
# 📉 2. Log dữ liệu giả (loss)
# ============================================================
for epoch in range(3):
    for step in range(10):
        train_loss = np.exp(-0.1 * step) + np.random.rand() * 0.02
        val_loss = np.exp(-0.12 * step) + np.random.rand() * 0.02

        # Log scalar metrics
        run.track(train_loss, name="Loss/Train", step=step, epoch=epoch, context={"subset": "train"})
        run.track(val_loss, name="Loss/Val", step=step, epoch=epoch, context={"subset": "val"})

        print(f"[Epoch {epoch}] Step {step} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")
        time.sleep(0.1)

# ============================================================
# 🖼️ 3. Log hình ảnh giả
# ============================================================
# Tạo 1 ảnh RGB ngẫu nhiên (64x64)
fake_img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
pil_img = Image.fromarray(fake_img)

run.track(AimImage(pil_img), name="Generated/Image_Sample", step=0, context={"subset": "demo"})

# ============================================================
# 📝 4. Log text (ghi chú)
# ============================================================
run.track_text("✅ AIM test run completed successfully!", name="Notes")
run["info/status"] = "success"
run["info/hostname"] = os.uname().nodename

print("✅ Finished logging to AIM.")
print("👉 Check the run at: http://159.89.193.7:43800/runs")

# Đóng kết nối
run.close()
