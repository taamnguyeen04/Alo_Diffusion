# Chỉnh sửa Biểu cảm Khuôn mặt qua Rectified Flow và Biến đổi Wavelet
*(Facial Emotion Editing via Rectified Flow and Wavelets)*

## Dữ liệu

Chúng tôi sử dụng phiên bản đã được xử lý của bộ dữ liệu **AffectNet** bao gồm 7 cảm xúc cơ bản: *Bình thường (Neutral), Vui vẻ (Happy), Buồn (Sad), Bất ngờ (Surprise), Sợ hãi (Fear), Ghê tởm (Disgust), Tức giận (Anger)*.

**Link tải dữ liệu:** [AffectNet-New (Kaggle)](https://www.kaggle.com/datasets/minhtmnguyntrn/affectnet-new)


## Kiến trúc Mô hình

Hệ thống được cấu thành từ 5 module chính:
1. **DTCWT Encoder (Bộ mã hóa DTCWT)**: Chuyển đổi ảnh RGB sang không gian màu YCbCr và áp dụng Biến đổi Wavelet Phức Cây Kép (DTCWT) 2 cấp để trích xuất Tần số thấp và các biên độ/pha của Tần số cao.
2. **LL Diffusion UNet**: Mạng UNet có điều kiện, làm nhiệm vụ dự đoán trường vector vận tốc của Rectified Flow trên dải Tần số thấp (LL). Nó tích hợp embedding thời gian và điều kiện cảm xúc thông qua Cross-Attention và cơ chế FiLM/AdaGN.
3. **WPTL Layer (Lớp Vận chuyển Pha Wavelet)**: Một module tính toán toán học không tham số, đóng vai trò trích xuất bộ pha cục bộ và giải ma trận bình phương tối thiểu tuyến tính nhằm tìm ra trường dịch chuyển (displacement field) $u(x)$ để warp biên độ và pha của HF một cách mượt mà nhất, chống chệch khối không gian.
4. **Birth/Death Innovation**: Dựa vào Năng lượng Định hướng (Oriented Energy) để tính toán các mặt nạ Sinh/Diệt. Nhiệm vụ của nó là vẽ nên các chi tiết tần số cao mới (chưa từng xuất hiện ở ảnh gốc) do sự thay đổi cục diện cấu trúc khuôn mặt khuôn mặt (ví dụ: hé răng, nếp nhăn đuôi mắt).
5. **DTCWT Decoder (Bộ giải mã DTCWT)**: Gom cấu trúc LL (đã dịch chuyển biểu cảm) và hệ số phức HF (đã warp trường và qua mạng Innovation) để tái tạo thành bức ảnh RGB mang biểu cảm đích cuối cùng.

## Hàm Mục tiêu Huấn luyện

Mô hình được tối ưu qua một hàm Loss đa mục tiêu toàn diện:
- **$L_{flow}$**: Khớp vector vận tốc Rectified Flow trên dải LL.
- **$L_{mag}$**: Cố định và bảo toàn biên độ dải HF ở những khu vực sinh học (birth regions).
- **$L_{chroma}$**: Ràng buộc tính nhất quán không gian màu trên 2 kênh Cb/Cr của LL.
- **$L_{smooth}$**: Độ mượt Total-Variation (TV) trên trường dịch chuyển quang $u$.
- **$L_{rec}$**: Loss tái tạo (reconstruction) L1 trực tiếp trên không gian điểm ảnh RGB.
- **$L_{LPIPS}$**: Chất lượng nhận thức qua mạng VGG.
- **$L_{ID}$**: Mất mát bảo toàn danh tính nhân vật qua ArcFace cosine similarity.
- **$L_{cls}$**: Định hướng ngữ nghĩa cảm xúc qua bộ phân loại DAN (DAN Classifier) đã pretrained.

