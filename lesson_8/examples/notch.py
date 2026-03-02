import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# === BƯỚC 1: TẠO DỮ LIỆU NHIỄU TUẦN HOÀN (PERIODIC NOISE) ===
img = cv2.imread("images/sample.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    img = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(img, (128, 128), 60, 180, -1)
    cv2.rectangle(img, (30, 30), (100, 90), 200, -1)

img = cv2.resize(img, (256, 256)).astype(np.float64)
h, w = img.shape

# Giả lập nhiễu sọc chéo bằng cách cộng các hàm Sin
freq1, freq2 = 20, 35
y_grid, x_grid = np.mgrid[0:h, 0:w]
noise = 40 * np.sin(2 * np.pi * freq1 * x_grid / w) + \
        30 * np.sin(2 * np.pi * freq2 * y_grid / h)

img_noisy = np.clip(img + noise, 0, 255).astype(np.uint8)

# === BƯỚC 2: PHÂN TÍCH PHỔ FOURIER ===
opt_h, opt_w = cv2.getOptimalDFTSize(h), cv2.getOptimalDFTSize(w)
padded = np.zeros((opt_h, opt_w))
padded[:h, :w] = img_noisy

dft = np.fft.fft2(padded)
dft_shift = np.fft.fftshift(dft)
magnitude = np.abs(dft_shift)
spectrum = np.log1p(magnitude) # Chuẩn hóa log để quan sát đốm nhiễu

# === BƯỚC 3: TỰ ĐỘNG PHÁT HIỆN ĐỐM NHIỄU (PEAK DETECTION) ===
crow, ccol = opt_h // 2, opt_w // 2
mag_search = magnitude.copy()

# Loại bỏ vùng tâm (tần số thấp) để không xóa nhầm nội dung ảnh
dc_radius = 10
y_idx, x_idx = np.ogrid[:opt_h, :opt_w]
dc_mask = ((y_idx - crow) ** 2 + (x_idx - ccol) ** 2) <= dc_radius ** 2
mag_search[dc_mask] = 0

# Tìm K đỉnh có năng lượng cao nhất (thường xuất hiện theo cặp đối xứng)
K = 4 
peaks = []
mag_flat = mag_search.copy()

for _ in range(K):
    idx = np.unravel_index(np.argmax(mag_flat), mag_flat.shape)
    peaks.append(idx)
    # Sau khi tìm thấy 1 đỉnh, xóa vùng xung quanh nó để tìm đỉnh tiếp theo
    peak_mask = ((y_idx - idx[0]) ** 2 + (x_idx - idx[1]) ** 2) <= 15 ** 2
    mag_flat[peak_mask] = 0

# === BƯỚC 4: TẠO MẶT NẠ NOTCH GAUSSIAN ===
def tao_notch_gaussian(shape, peaks, d0=10):
    rows, cols = shape
    H = np.ones((rows, cols), dtype=np.float64)
    for (py, px) in peaks:
        y_idx_l, x_idx_l = np.ogrid[:rows, :cols]
        D = np.sqrt((y_idx_l - py) ** 2 + (x_idx_l - px) ** 2)
        notch = 1.0 - np.exp(-D ** 2 / (2 * d0 ** 2))
        H *= notch
    return H

H_notch = tao_notch_gaussian((opt_h, opt_w), peaks, d0=8)

# === BƯỚC 5: LỌC VÀ TÁI TẠO ẢNH ===
filtered_shift = dft_shift * H_notch
result = np.real(np.fft.ifft2(np.fft.ifftshift(filtered_shift)))
result = np.clip(result[:h, :w], 0, 255).astype(np.uint8)

# === BƯỚC 6: HIỂN THỊ VÀ ĐÁNH GIÁ (PSNR) ===
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

axes[0, 0].imshow(img.astype(np.uint8), cmap="gray")
axes[0, 0].set_title("1. Ảnh gốc (Sạch)")
axes[0, 0].axis("off")

axes[0, 1].imshow(img_noisy, cmap="gray")
axes[0, 1].set_title("2. Ảnh bị nhiễu sọc tuần hoàn")
axes[0, 1].axis("off")

axes[0, 2].imshow(spectrum, cmap="gray")
for (py, px) in peaks:
    axes[0, 2].add_patch(plt.Circle((px, py), 12, color="red", fill=False, linewidth=2))
axes[0, 2].set_title("3. Phổ DFT - Phát hiện đốm nhiễu")
axes[0, 2].axis("off")

axes[1, 0].imshow(H_notch, cmap="gray")
axes[1, 0].set_title("4. Bộ lọc Notch (Mặt nạ xóa)")
axes[1, 0].axis("off")

axes[1, 1].imshow(np.log1p(np.abs(filtered_shift)), cmap="gray")
axes[1, 1].set_title("5. Phổ sau khi 'đục lỗ' xóa nhiễu")
axes[1, 1].axis("off")

axes[1, 2].imshow(result, cmap="gray")
axes[1, 2].set_title("6. Kết quả sau phục hồi")
axes[1, 2].axis("off")

plt.suptitle("Lọc Notch: Khôi phục ảnh từ nhiễu tần số", fontsize=16)
plt.tight_layout()
os.makedirs("output", exist_ok=True)
plt.savefig("output/notch_filter_final.png", dpi=150)
plt.show()

# Đánh giá khách quan bằng PSNR (Peak Signal-to-Noise Ratio)
psnr_noisy = cv2.PSNR(img.astype(np.uint8), img_noisy)
psnr_final = cv2.PSNR(img.astype(np.uint8), result)
print(f"Chất lượng ảnh nhiễu: {psnr_noisy:.2f} dB")
print(f"Chất lượng sau lọc Notch: {psnr_final:.2f} dB (Tăng {psnr_final - psnr_noisy:.2f} dB)")