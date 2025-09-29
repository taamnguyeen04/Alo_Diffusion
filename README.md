# Wavelet Diffusion chỉnh sửa cảm xúc khuôn mặt trong ảnh

## 1. Giới thiệu

Bài toán được giải quyết là **chỉnh sửa cảm xúc khuôn mặt**: từ một ảnh gốc, sinh ra ảnh mới với khuôn mặt giữ nguyên đặc trưng nhận dạng nhưng thay đổi cảm xúc theo mục tiêu.
Mô hình sử dụng **Diffusion model** kết hợp **biến đổi wavelet rời rạc (DWT/IWT)** để xử lý ảnh theo miền tần số. Điều này giúp bảo toàn thông tin nhận dạng (high-frequency details) trong khi điều chỉnh hiệu quả các thành phần cảm xúc (low-frequency structures). Ngoài ra, mô hình còn sử dụng **embedding cảm xúc**, **cross-attention**, và **các hàm mất mát đa mục tiêu** nhằm cân bằng giữa việc giữ nguyên danh tính và biểu đạt cảm xúc.

---

## 2. Thành phần chính của mô hình

### 2.1. Trích chọn và nhận diện cảm xúc phụ trợ

File `criterions.py` định nghĩa:

* **AdaptiveLossWeighter**: Điều chỉnh trọng số loss động theo tiến trình huấn luyện. Ban đầu nhấn mạnh vào việc học biểu đạt cảm xúc, sau đó cân bằng dần với mất mát nhận dạng.
* **Emotion_model (DAN)**: Một mạng học từ ResNet18 với cơ chế **Spatial Attention** + **Channel Attention** để phân loại 7 loại cảm xúc (neutral, happy, sad, surprise, fear, disgust, anger).
  Mô hình này được dùng để giám sát việc sinh ảnh (ép ảnh sinh ra thể hiện cảm xúc mục tiêu).

### 2.2. Wavelet Diffusion Model

File `model.py` là trung tâm, gồm:

* **DWT/IWT**: Biến đổi wavelet rời rạc và ngược.

  * DWT phân tách ảnh thành 4 thành phần (LL, LH, HL, HH).
  * IWT tái tạo lại ảnh từ các thành phần này.
    Điều này cho phép xử lý **low-frequency (cảm xúc, cấu trúc khuôn mặt)** và **high-frequency (chi tiết nhận dạng)** riêng biệt.

* **TimeEmbedding**: Mã hóa timestep trong quá trình diffusion.

* **ResBlock + CrossAttention**: Residual block có điều kiện theo timestep và embedding cảm xúc. FiLM (Feature-wise Linear Modulation) và cross-attention giúp trộn embedding cảm xúc vào đặc trưng ảnh.

* **FrequencyBottleneckBlock**: Chỉ xử lý thành phần tần số thấp, giữ nguyên thành phần cao → giúp thay đổi biểu cảm nhưng bảo toàn danh tính.

* **FreqAwareDownsample / FreqAwareUpsample**: Thao tác giảm/tăng kích thước có tính đến miền tần số.

* **WaveletResidualConnection**: Giữ lại đặc trưng gốc từ ảnh source để bảo toàn nhận dạng.

* **WaveletUNet**: Cấu trúc UNet với encoder–bottleneck–decoder, nhưng thay maxpool/upsample bằng DWT/IWT. Có embedding cảm xúc điều kiện vào mọi mức.

* **WaveletDiffusionModel**:

  * Forward process: thêm nhiễu vào wavelet coefficients.
  * Training: học dự đoán nhiễu.
  * Sampling (DDIM-like): từ ảnh gốc, thêm nhiễu rồi khử nhiễu dần để sinh ra ảnh với cảm xúc mục tiêu.

---

## 3. Quá trình huấn luyện

File `train.py` thực hiện:

* **Dataset**: AffectNet – tập dữ liệu lớn về cảm xúc khuôn mặt.

* **Các hàm mất mát**:

  1. **DDPM loss**: L1 giữa noise dự đoán và noise thực.
  2. **DAN Expression loss**: Cross-entropy từ mô hình phụ trợ DAN để đảm bảo cảm xúc mục tiêu.
  3. **Wavelet loss**: So sánh thành phần LL/HH giữa ảnh gốc và sinh → cân bằng giữ đặc trưng.
  4. **Identity loss**: Cosine similarity giữa feature ResNet50 của ảnh gốc và ảnh sinh.
  5. **LPIPS loss**: Đo mức độ giống nhau về nhận thức thị giác.
     → Các loss được **AdaptiveLossWeighter** điều chỉnh trọng số trong quá trình train.

* **Checkpointing**: Lưu `last_model.pt` và `best_model.pt` dựa trên loss tốt nhất.

* **Logging và Visualization**:

  * TensorBoard log loss theo batch.
  * Sinh ảnh mẫu định kỳ cho tất cả cảm xúc mục tiêu để đánh giá trực quan.

---

## 4. Điểm mạnh của mô hình

1. **Khai thác miền tần số với Wavelet**:

   * Low-frequency → chỉnh sửa cảm xúc.
   * High-frequency → giữ chi tiết danh tính.

2. **Điều kiện cảm xúc đa dạng**:

   * Embedding cảm xúc học được thay vì one-hot.
   * Cross-attention + FiLM đảm bảo tín hiệu cảm xúc truyền vào mọi tầng.

3. **Cân bằng loss động**:
   AdaptiveLossWeighter giúp mô hình không bị lệch về một mục tiêu duy nhất (ví dụ: thay đổi cảm xúc mạnh nhưng mất nhận dạng).

4. **Sampling linh hoạt**:
   Tham số `denoising_strength` cho phép điều chỉnh mức độ thay đổi cảm xúc (từ nhẹ đến cực đoan).