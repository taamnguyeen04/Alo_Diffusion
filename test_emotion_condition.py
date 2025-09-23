import torch
import torch.nn.functional as F
from model import WaveletDiffusionModel, DWT, IWT
from dataset import Affectnet
from torchvision.transforms import Resize, ToTensor, Compose, Normalize
from torch.utils.data import DataLoader
import os

def test_emotion_sensitivity():
    """Test A: Kiểm tra model có nhạy cảm với emotion condition không"""
    print("=== TEST A: EMOTION SENSITIVITY ===")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WaveletDiffusionModel(num_emotions=7)

    # Load checkpoint nếu có
    checkpoint_path = r"C:\Users\tam\Documents\data\FEG\WD v4 21000 step\best_model.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded model from {checkpoint_path}")
    else:
        print("⚠️ Không tìm thấy checkpoint, sử dụng model chưa train")

    model.to(device)
    model.eval()

    # Load validation data
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.5402, 0.4410, 0.3938], std=[0.2914, 0.2657, 0.2609])
    ])

    try:
        val_dataset = Affectnet(root="C:/Users/tam/Documents/data/FEG", is_train=False, transform=transform)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        print(f"✅ Loaded validation dataset: {len(val_dataset)} samples")
    except:
        print("❌ Không thể load dataset, tạo random data để test")
        x_val = torch.randn(1, 3, 224, 224).to(device)
        val_dataloader = None

    # Get one validation image
    if val_dataloader:
        x_val = next(iter(val_dataloader))[0].to(device)[:1]

    dwt = DWT()

    with torch.no_grad():
        x_wave = dwt(x_val)
        t_test = torch.tensor([50], device=device)
        x_noisy, _ = model.forward_process(x_wave, t_test)

        # Test với 2 emotions khác nhau
        pred_em0 = model.unet(x_noisy, t_test, torch.tensor([0], device=device), x_val)
        pred_em1 = model.unet(x_noisy, t_test, torch.tensor([1], device=device), x_val)

        diff = (pred_em0 - pred_em1).abs().mean().item()
        print(f"📊 Mean abs diff between emotion 0 vs 1: {diff:.6f}")

        if diff < 1e-5:
            print("❌ VẤN ĐỀ: Model hầu như không phân biệt emotions (diff ≈ 0)")
        elif diff < 1e-3:
            print("⚠️ CẢNH BÁO: Model phân biệt emotions rất yếu")
        else:
            print("✅ OK: Model có phân biệt emotions")

    return diff

def test_residual_connection():
    """Test B: Kiểm tra residual connection có quá mạnh không"""
    print("\n=== TEST B: RESIDUAL CONNECTION STRENGTH ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WaveletDiffusionModel(num_emotions=7)

    # Load checkpoint nếu có
    checkpoint_path = r"C:\Users\tam\Documents\data\FEG\WD v4 21000 step\best_model.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    model.eval()

    # Tạo test data
    x_val = torch.randn(1, 3, 224, 224).to(device)

    with torch.no_grad():
        # Kiểm tra res_blocks đầu tiên
        try:
            res_feat0 = model.unet.res_blocks[0](x_val)
            res_magnitude = res_feat0.abs().mean().item()
            input_magnitude = x_val.abs().mean().item()

            print(f"📊 Input magnitude: {input_magnitude:.6f}")
            print(f"📊 Residual feature magnitude: {res_magnitude:.6f}")
            print(f"📊 Ratio (res/input): {res_magnitude/input_magnitude:.2f}")

            if res_magnitude > input_magnitude * 10:
                print("❌ VẤN ĐỀ: Residual connection quá mạnh, có thể áp đảo features chính")
            elif res_magnitude > input_magnitude * 3:
                print("⚠️ CẢNH BÁO: Residual connection khá mạnh")
            else:
                print("✅ OK: Residual connection có độ mạnh hợp lý")

        except Exception as e:
            print(f"❌ Lỗi khi test residual: {e}")

    return res_magnitude if 'res_magnitude' in locals() else 0

def test_generation_difference():
    """Test C: Kiểm tra generation với emotions khác nhau"""
    print("\n=== TEST C: GENERATION DIFFERENCE ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WaveletDiffusionModel(num_emotions=7)

    # Load checkpoint nếu có
    checkpoint_path = r"C:\Users\tam\Documents\data\FEG\WD v4 21000 step\best_model.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    model.eval()

    # Tạo test image
    x_val = torch.randn(1, 3, 224, 224).to(device)

    print("🔄 Generating với 2 emotions khác nhau (có thể mất vài phút)...")

    with torch.no_grad():
        # Generate với emotion 0 và 1
        out_em0 = model.sample(x_val, torch.tensor([0], device=device), num_steps=200)  # Giảm steps để nhanh hơn
        out_em1 = model.sample(x_val, torch.tensor([1], device=device), num_steps=200)

        diff = (out_em0 - out_em1).abs().mean().item()
        print(f"📊 Diff between generated images: {diff:.6f}")

        if diff < 1e-4:
            print("❌ VẤN ĐỀ: Các generated images hầu như giống hệt nhau")
        elif diff < 1e-2:
            print("⚠️ CẢNH BÁO: Generated images khá giống nhau")
        else:
            print("✅ OK: Generated images có sự khác biệt rõ ràng")

        # Save images để kiểm tra visual
        try:
            from torchvision.utils import save_image
            save_image(out_em0, "test_emotion_0.png", normalize=True)
            save_image(out_em1, "test_emotion_1.png", normalize=True)
            save_image(x_val, "test_input.png", normalize=True)
            print("💾 Đã lưu test images: test_emotion_0.png, test_emotion_1.png, test_input.png")
        except Exception as e:
            print(f"⚠️ Không thể lưu images: {e}")

    return diff

def run_all_tests():
    """Chạy tất cả các tests"""
    print("🧪 BẮT ĐẦU KIỂM TRA MODEL EMOTION CONDITIONING\n")

    # Test A: Emotion sensitivity
    diff_a = test_emotion_sensitivity()

    # Test B: Residual connection
    res_mag = test_residual_connection()

    # Test C: Generation difference
    diff_c = test_generation_difference()

    # Tổng kết
    print("\n" + "="*50)
    print("📋 TỔNG KẾT KIỂM TRA:")
    print(f"   A. Emotion sensitivity: {diff_a:.6f}")
    print(f"   B. Residual magnitude: {res_mag:.6f}")
    print(f"   C. Generation difference: {diff_c:.6f}")

    # Đưa ra kết luận tổng thể
    issues = []
    if diff_a < 1e-5:
        issues.append("Model không phân biệt emotions")
    if res_mag > 10:  # Assuming input magnitude ~1
        issues.append("Residual connection quá mạnh")
    if diff_c < 1e-4:
        issues.append("Generated images quá giống nhau")

    if issues:
        print(f"\n❌ CÓ VẤN ĐỀ: {', '.join(issues)}")
        print("\n💡 ĐỀ XUẤT:")
        if diff_a < 1e-5:
            print("   - Kiểm tra emotion embedding có được sử dụng đúng")
            print("   - Tăng learning rate cho emotion components")
            print("   - Kiểm tra loss function có encourage emotion conditioning")
        if res_mag > 10:
            print("   - Giảm scale của residual connections")
            print("   - Thêm normalization cho residual features")
        if diff_c < 1e-4:
            print("   - Tăng strength của emotion conditioning")
            print("   - Kiểm tra sampling process")
    else:
        print("\n✅ KHÔNG CÓ VẤN ĐỀ NGHIÊM TRỌNG: Model có vẻ học được emotion conditioning")

if __name__ == "__main__":
    run_all_tests()