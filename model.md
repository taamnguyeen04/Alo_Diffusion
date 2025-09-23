## 1. **`DWT` (Discrete Wavelet Transform)**

* **Ý nghĩa**: Biến đổi wavelet rời rạc, phân tách ảnh thành 4 thành phần tần số:

  * LL (thấp/thấp), LH (thấp/cao), HL (cao/thấp), HH (cao/cao).
* **Hoạt động**:

  * Dùng convolution với bộ lọc low-pass và high-pass theo chiều ngang/dọc.
  * Kết quả: giảm kích thước không gian (H/2, W/2), tăng số kênh (x4).
* **Mục đích**: Giúp mô hình xử lý tách biệt thông tin tổng thể (low-frequency) và chi tiết (high-frequency).

---

## 2. **`IWT` (Inverse Wavelet Transform)**

* **Ý nghĩa**: Phép biến đổi ngược của DWT để tái tạo ảnh từ các thành phần tần số.
* **Hoạt động**:

  * Nhận đầu vào `(B, 4*C, H, W)`.
  * Tách thành LL, LH, HL, HH.
  * Dùng `conv_transpose2d` để upsample và cộng lại.
* **Mục đích**: Khôi phục ảnh từ hệ số wavelet đã qua xử lý.

---

## 3. **`TimeEmbedding`**

* **Ý nghĩa**: Mã hóa thông tin timestep (số bước trong diffusion).
* **Hoạt động**:

  * Dùng sinusoidal positional encoding giống Transformer.
  * Mỗi timestep → vector đặc trưng có tính tuần hoàn.
* **Mục đích**: Cho phép mô hình biết ảnh đang ở bước nhiễu nào.

---

## 4. **`CrossAttention`**

* **Ý nghĩa**: Cơ chế cross-attention để trộn đặc trưng ảnh với embedding cảm xúc.
* **Hoạt động**:

  * Lấy feature ảnh (query).
  * Lấy emotion embedding (key, value).
  * Tính attention → kết hợp thông tin cảm xúc vào đặc trưng ảnh.
* **Mục đích**: Kiểm soát mạnh mẽ hơn việc thay đổi cảm xúc trong ảnh.

---

## 5. **`ResBlock`**

* **Ý nghĩa**: Residual block cải tiến, kết hợp cả timestep và emotion embedding.
* **Hoạt động**:

  * Chuẩn hóa (GroupNorm) → conv → cộng time embedding + emotion embedding.
  * Nếu bật `use_cross_attn` → dùng CrossAttention.
  * Conv tiếp → cộng shortcut (residual).
* **Mục đích**:

  * Giữ thông tin gốc (residual).
  * Kết hợp timestep và emotion vào đặc trưng.

---

## 6. **`FrequencyBottleneckBlock`**

* **Ý nghĩa**: Xử lý đặc biệt cho tần số thấp, giữ nguyên tần số cao.
* **Hoạt động**:

  * DWT ảnh thành LL + HI.
  * LL đi qua 2 ResBlock (tích hợp thời gian + cảm xúc).
  * Nối lại với HI rồi IWT khôi phục.
* **Mục đích**: Nhấn mạnh xử lý cấu trúc tổng thể (low-frequency) và giữ chi tiết (high-frequency).

---

## 7. **`FreqAwareDownsample`**

* **Ý nghĩa**: Downsampling có nhận biết tần số.
* **Hoạt động**:

  * DWT → tăng kênh x4.
  * Conv giảm kênh → GroupNorm.
  * Cộng time + emotion embedding.
  * Trả về (feature downsampled, high-frequency skip).
* **Mục đích**: Giữ lại chi tiết tần số cao cho skip connection sau này.

---

## 8. **`FreqAwareUpsample`**

* **Ý nghĩa**: Upsampling có nhận biết tần số.
* **Hoạt động**:

  * Conv + chuẩn hóa + cộng time + emotion embedding.
  * Nối với high-frequency skip từ encoder.
  * IWT để upsample.
* **Mục đích**: Phục hồi ảnh trong decoder, đồng thời kết hợp chi tiết tần số cao.

---

## 9. **`WaveletResidualConnection`**

* **Ý nghĩa**: Kết nối tần số từ ảnh gốc (source image) để giữ đặc trưng nhận dạng.
* **Hoạt động**:

  * Áp dụng DWT nhiều lần (theo `downsample_level`).
  * Conv để giảm số kênh.
* **Mục đích**: Đưa đặc trưng gốc của ảnh vào UNet → giúp bảo toàn danh tính.

---

## 10. **`WaveletUNet`**

* **Ý nghĩa**: UNet nhúng wavelet, có điều kiện cảm xúc.
* **Cấu trúc**:

  * **Encoder**: nhiều tầng ResBlock + FreqAwareDownsample.
  * **Residual connections**: thêm WaveletResidualConnection từ ảnh gốc.
  * **Bottleneck**: ResBlock + FrequencyBottleneckBlock.
  * **Decoder**: nhiều tầng FreqAwareUpsample + ResBlock.
  * **Aux heads**: dự đoán cảm xúc và VA từ bottleneck.
* **Đầu ra**: dự đoán nhiễu (`noise_pred`), hoặc thêm nhãn phụ (`expr_pred`, `va_pred`).

---

## 11. **`WaveletDiffusionModel`**

* **Ý nghĩa**: Mô hình diffusion hoàn chỉnh để chỉnh sửa cảm xúc.
* **Thành phần**:

  * DWT/IWT.
  * WaveletUNet.
  * Các tham số diffusion (`betas`, `alphas`, `alphas_cumprod`).
* **Hàm chính**:

  * `forward_process(x0, t)`: thêm nhiễu vào wavelet ở bước `t`.
  * `forward(x, emotion_id, src_image)`: training → dự đoán nhiễu và tính loss.
  * `sample(src_image, target_emotion_id)`: inference → dùng DDIM sampling để sinh ảnh mới theo cảm xúc mục tiêu.
