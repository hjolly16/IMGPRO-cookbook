import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- HÀM HỖ TRỢ TỪ CÁC BÀI TRƯỚC ---
def tao_lpf_gaussian(shape, d0):
    """Tạo bộ lọc thông thấp Gaussian (Gaussian Low-Pass Filter)."""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows).reshape(-1, 1) - crow
    v = np.arange(cols).reshape(1, -1) - ccol
    D = np.sqrt(u**2 + v**2)
    return np.exp(-D**2 / (2 * d0**2))

def loc_tan_so(img_gray, H):
    """Áp dụng ma trận lọc H vào ảnh trong miền tần số."""
    rows, cols = img_gray.shape
    # Tối ưu kích thước cho thuật toán FFT
    opt_r = cv2.getOptimalDFTSize(rows)
    opt_c = cv2.getOptimalDFTSize(cols)
    padded = np.zeros((opt_r, opt_c))
    padded[:rows, :cols] = img_gray

    # Biến đổi sang miền tần số
    dft = np.fft.fft2(padded)
    dft_shift = np.fft.fftshift(dft)

    # Đảm bảo ma trận lọc khớp kích thước với ảnh đã padding
    if H.shape != (opt_r, opt_c):
        H = cv2.resize(H, (opt_c, opt_r))

    # Nhân trực tiếp ma trận lọc và phổ biên độ
    filtered = dft_shift * H
    
    # Biến đổi ngược về miền không gian
    f_ishift = np.fft.ifftshift(filtered)
    result = np.real(np.fft.ifft2(f_ishift))
    
    # Cắt về kích thước ban đầu và chuẩn hóa
    result = result[:rows, :cols]
    return np.clip(result, 0, 255).astype(np.uint8)

# === BƯỚC 1: CHUẨN BỊ DỮ LIỆU ===
img = cv2.imread("images/sample.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    # Tạo ảnh giả lập nếu không tìm thấy file
    img = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(img, (128, 128), 50, 200, -1)
    cv2.rectangle(img, (30, 30), (100, 80), 150, -1)
    img = cv2.GaussianBlur(img, (5, 5), 2)

h, w = img.shape
shape = (cv2.getOptimalDFTSize(h), cv2.getOptimalDFTSize(w))

# Ngưỡng tần số cắt
d0 = 30

# === BƯỚC 2: XÂY DỰNG CÁC BỘ LỌC MIỀN TẦN SỐ ===
# 1. Gaussian Low-Pass Filter (Làm mờ)
H_lpf = tao_lpf_gaussian(shape, d0)

# 2. Gaussian High-Pass Filter (Trích xuất biên - làm mất độ sáng nền)
H_hpf = 1.0 - H_lpf

# 3. High-Frequency Emphasis Filter (Tăng nét - giữ độ sáng nền)
# Công thức: H_hfe = a + b * H_hpf (với a=0.5, b=2.0)
a, b = 0.5, 2.0
H_hfe = a + b * H_hpf

# Thực hiện lọc
result_lpf = loc_tan_so(img, H_lpf)
result_hpf = loc_tan_so(img, H_hpf)
result_hfe = loc_tan_so(img, H_hfe)

# === BƯỚC 3: SO SÁNH VỚI UNSHARP MASKING (MIỀN KHÔNG GIAN) ===
# Unsharp Masking = Original + (Original - Blurred)
blur = cv2.GaussianBlur(img, (0, 0), 3)
unsharp = cv2.addWeighted(img, 2.0, blur, -1.0, 0)

# === BƯỚC 4: HIỂN THỊ VÀ PHÂN TÍCH ===
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

axes[0, 0].imshow(img, cmap="gray")
axes[0, 0].set_title("1. Ảnh gốc")
axes[0, 0].axis("off")

axes[0, 1].imshow(result_lpf, cmap="gray")
axes[0, 1].set_title(f"2. Gaussian LPF (D0={d0})\n(Làm mờ/Mất chi tiết cao)")
axes[0, 1].axis("off")

axes[0, 2].imshow(result_hpf, cmap="gray")
axes[0, 2].set_title(f"3. Gaussian HPF (D0={d0})\n(Chỉ giữ biên - Ảnh tối)")
axes[0, 2].axis("off")

axes[1, 0].imshow(result_hfe, cmap="gray")
axes[1, 0].set_title(f"4. HFE (a={a}, b={b})\n(Giữ độ sáng + Nhấn biên)")
axes[1, 0].axis("off")

axes[1, 1].imshow(unsharp, cmap="gray")
axes[1, 1].set_title("5. Unsharp Masking\n(Thực hiện ở miền không gian)")
axes[1, 1].axis("off")

# Profile cường độ điểm ảnh để soi độ sắc nét
row_mid = h // 2
axes[1, 2].plot(img[row_mid, :], "b-", alpha=0.5, label="Gốc")
axes[1, 2].plot(result_hfe[row_mid, :], "r-", label="HFE (Miền tần số)")
axes[1, 2].plot(unsharp[row_mid, :], "g--", label="Unsharp (Không gian)")
axes[1, 2].set_xlabel("Vị trí Pixel (cột)")
axes[1, 2].set_ylabel("Cường độ sáng")
axes[1, 2].set_title(f"Đồ thị Profile hàng {row_mid}")
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.suptitle("Tăng cường độ nét: HPF vs High-Frequency Emphasis", fontsize=16)
plt.tight_layout()
os.makedirs("output", exist_ok=True)
plt.savefig("output/hpf_hfe_analysis.png", dpi=150)
plt.show()

print("KẾT LUẬN TUTORIAL:")
print(f"- HPF làm mất thành phần DC nên ảnh rất tối.")
print(f"- HFE dùng hệ số a={a} để bù đắp độ sáng và b={b} để làm sắc nét đường biên.")
print(f"- Kết quả HFE tương đương với Unsharp Masking nhưng linh hoạt hơn trong việc chọn dải tần số.")