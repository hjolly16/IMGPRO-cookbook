import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- CÁC HÀM TẠO BỘ LỌC (MASK) TRÊN MIỀN TẦN SỐ ---

def tao_lpf_gaussian(shape, d0):
    """Tạo mặt nạ Gaussian Low-Pass Filter: H(u,v) = exp(-D^2 / 2D0^2)."""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows).reshape(-1, 1) - crow
    v = np.arange(cols).reshape(1, -1) - ccol
    D = np.sqrt(u**2 + v**2)
    H = np.exp(-(D**2) / (2 * (d0**2)))
    return H

def tao_lpf_ideal(shape, d0):
    """Tạo mặt nạ Ideal Low-Pass Filter: Cắt bỏ hoàn toàn nếu D > D0."""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows).reshape(-1, 1) - crow
    v = np.arange(cols).reshape(1, -1) - ccol
    D = np.sqrt(u**2 + v**2)
    H = (D <= d0).astype(np.float64)
    return H

def tao_lpf_butterworth(shape, d0, n=2):
    """Tạo mặt nạ Butterworth Low-Pass Filter: H(u,v) = 1 / [1 + (D/D0)^2n]."""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows).reshape(-1, 1) - crow
    v = np.arange(cols).reshape(1, -1) - ccol
    D = np.sqrt(u**2 + v**2)
    D[D == 0] = 1e-10 # Tránh lỗi chia cho 0 tại tâm
    H = 1.0 / (1.0 + (D / d0)**(2 * n))
    return H

def loc_tan_so(img_gray, H):
    """Hàm thực hiện lọc ảnh bằng cách nhân ma trận trong miền tần số."""
    rows, cols = img_gray.shape
    # 1. Padding để tối ưu tốc độ FFT
    opt_r = cv2.getOptimalDFTSize(rows)
    opt_c = cv2.getOptimalDFTSize(cols)
    padded = np.zeros((opt_r, opt_c))
    padded[:rows, :cols] = img_gray

    # 2. Chuyển sang miền tần số
    dft = np.fft.fft2(padded)
    dft_shift = np.fft.fftshift(dft)

    # 3. Áp dụng bộ lọc (Nhân trực tiếp từng pixel)
    # Đảm bảo H có kích thước khớp với ảnh đã padded
    if H.shape != (opt_r, opt_c):
        H_resized = cv2.resize(H, (opt_c, opt_r))
    else:
        H_resized = H

    filtered = dft_shift * H_resized

    # 4. Biến đổi ngược về miền không gian
    f_ishift = np.fft.ifftshift(filtered)
    result = np.real(np.fft.ifft2(f_ishift))

    # 5. Cắt bỏ phần padding và chuẩn hóa dải pixel
    result = result[:rows, :cols]
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

# === BƯỚC 1: ĐỌC VÀ CHUẨN BỊ ẢNH ===
img = cv2.imread("images/sample.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    img = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(img, (128, 128), 60, 255, -1)

h, w = img.shape
shape = (cv2.getOptimalDFTSize(h), cv2.getOptimalDFTSize(w))

# === BƯỚC 2: THỰC HIỆN SO SÁNH ===
d0_list = [20, 50, 100] # Các ngưỡng tần số cắt khác nhau
filter_funcs = [
    ("Ideal", tao_lpf_ideal),
    ("Butterworth (n=2)", lambda s, d: tao_lpf_butterworth(s, d, 2)),
    ("Gaussian", tao_lpf_gaussian),
]

fig, axes = plt.subplots(4, len(d0_list), figsize=(15, 18))

# Hiển thị mặt nạ Gaussian làm mẫu ở hàng đầu
for idx, d0 in enumerate(d0_list):
    H_vis = tao_lpf_gaussian(shape, d0)
    axes[0, idx].imshow(H_vis, cmap="gray")
    axes[0, idx].set_title(f"Gaussian Mask (D0={d0})")
    axes[0, idx].axis("off")

# Hiển thị kết quả của 3 bộ lọc qua 3 tần số cắt
for row_idx, (name, func) in enumerate(filter_funcs):
    for col_idx, d0 in enumerate(d0_list):
        H = func(shape, d0)
        result = loc_tan_so(img, H)
        axes[row_idx + 1, col_idx].imshow(result, cmap="gray")
        axes[row_idx + 1, col_idx].set_title(f"{name}, D0={d0}")
        axes[row_idx + 1, col_idx].axis("off")

plt.suptitle("So sánh các bộ lọc thông thấp (LPF) trong miền tần số", fontsize=16)
plt.tight_layout()
os.makedirs("output", exist_ok=True)
plt.savefig("output/lpf_comparison_analysis.png", dpi=150)
plt.show()