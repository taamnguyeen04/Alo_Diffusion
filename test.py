import torch
from pytorch_wavelets.dtcwt.transform2d import DTCWTForward

def test_dtcwt_levels():
    # 1. Tạo một bức ảnh giả (Dummy Image) kích thước 224x224, 3 kênh màu
    # Batch size = 1, Channels = 3, Height = 224, Width = 224
    x = torch.randn(1, 3, 224, 224)
    print("="*50)
    print(f"ẢNH GỐC ĐẦU VÀO: {x.shape}")
    print("="*50)

    # ---------------------------------------------------------
    # BÀI TEST 1: Chạy DTCWT với J = 1 (Như code cũ của bạn)
    # ---------------------------------------------------------
    dtcwt_j1 = DTCWTForward(J=1, biort='near_sym_b', qshift='qshift_b')
    yl_1, yh_1 = dtcwt_j1(x)
    
    print("\n[ KẾT QUẢ LEVEL 1 (J = 1) ]")
    print(f"-> Kích thước LL (Low-pass): {yl_1.shape}  <-- XEM KỸ DÒNG NÀY!")
    print(f"-> Kích thước HF (High-pass): {yh_1[0].shape}")
    print("   (HF đã bị giảm xuống 112x112, nhưng LL VẪN LÀ 224x224)")

    # ---------------------------------------------------------
    # BÀI TEST 2: Chạy DTCWT với J = 2 (Đề xuất tối ưu)
    # ---------------------------------------------------------
    dtcwt_j2 = DTCWTForward(J=2, biort='near_sym_b', qshift='qshift_b')
    yl_2, yh_2 = dtcwt_j2(x)
    
    print("\n[ KẾT QUẢ LEVEL 2 (J = 2) ]")
    print(f"-> Kích thước LL (Low-pass): {yl_2.shape}  <-- ĐÃ ĐƯỢC NÉN CHUẨN!")
    print(f"-> Kích thước HF Level 1 (yh[0]): {yh_2[0].shape}")
    print(f"-> Kích thước HF Level 2 (yh[1]): {yh_2[1].shape}")
    print("   (Nhờ J=2, LL đã được toán học nén hoàn hảo xuống 112x112)")
    print("="*50)

if __name__ == '__main__':
    # test_dtcwt_levels()
    dtcwt_j2 = DTCWTForward(J=3, biort='near_sym_b', qshift='qshift_b')
    x = torch.randn(1, 3, 224, 224)
    print(dtcwt_j2(x)[0].shape)
    print(dtcwt_j2(x)[1][0].shape)
    print(dtcwt_j2(x)[1][1].shape)
    print(dtcwt_j2(x)[1][2].shape)
    print(dtcwt_j2(x)[1][3].shape)