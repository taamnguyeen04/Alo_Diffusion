import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from datetime import datetime
# scp "WaveletDiffusion_film04\runs\exp\events.out.tfevents.1763323544.modal.272.0" root@159.89.193.7:/root/tb_logs/run_5/
# ==============================
# Cấu hình 1337  modal volume put affectnet "C:\Users\tam\Documents\GitHub\Alo_Diffusion\affecnet7_epoch6_acc0.6569.pth" /
# ==============================
# modal volume create affectnet  
# modal volume put affectnet "C:\Users\tam\Desktop\Data\FEG\validation.csv" /
# modal volume put affectnet "C:\Users\tam\Desktop\Data\FEG\training.csv" / 
# modal volume put affectnet "C:\Users\tam\Documents\GitHub\Alo_Diffusion\affecnet7_epoch6_acc0.6569.pth"   
BASE_DIR = "C:/Users/tam/Desktop/Data/FEG/Manually_Annotated_Images"
VOLUME_NAME = "affectnet"   # tên volume trên Modal
START = 1201                   # thư mục bắt đầu (ví dụ: 1)
END = 1337
# thư mục kết thúc (ví dụ: 300)

# File lưu danh sách folder bị lỗi
ERROR_LOG_FILE = "upload_errors.txt"
RETRY_FILE = "retry_list.txt"  # File chứa danh sách cần retry

cpu_cores = multiprocessing.cpu_count()
MAX_WORKERS = min(cpu_cores * 2, 8)  # tối đa 8 luồng cho ổn định

def upload_folder(folder_name, retries=3):
    local_path = os.path.join(BASE_DIR, folder_name)
    remote_path = f"/{folder_name}"
    cmd = ["modal", "volume", "put", "-f", VOLUME_NAME, local_path, remote_path]

    for attempt in range(1, retries+1):
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
        if result.returncode == 0:
            return f"✅ Uploaded {folder_name}"
        else:
            if attempt < retries:
                print(f"🔁 Retry {folder_name} (lần {attempt})...")
                return None
            else:
                return f"❌ Failed {folder_name}: {result.stderr}"
    return None


def main():
    # Tạo danh sách thư mục theo khoảng START..END
    folders = [str(i) for i in range(START, END + 1) if os.path.isdir(os.path.join(BASE_DIR, str(i)))]

    print(f"🔎 Upload từ thư mục {START} đến {END}, tổng {len(folders)} folders...")
    print(f"⚡ Dùng {MAX_WORKERS} luồng song song")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(upload_folder, f): f for f in folders}
        for future in as_completed(futures):
            print(future.result())

if __name__ == "__main__":
    main()


