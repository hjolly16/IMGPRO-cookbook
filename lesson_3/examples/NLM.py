import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# === BƯỚC 1: ĐẦU VÀO ===
img_path = "images/sample.jpg"
if not os.path.exists(img_path):
    print(f"Lỗi: Không tìm thấy file tại {img_path}")
    exit()

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError("Không thể giải mã ảnh!")

# Tạo nhiễu Gaussian mạnh (sigma=30) để thử thách các bộ lọc
nhieu = np.random.normal(0, 30, img.shape)
img_nhieu = np.clip(img.astype(np.float64) + nhieu, 0, 255).astype(np.uint8)

def tinh_psnr(goc, ket_qua):
    """Tính Peak Signal-to-Noise Ratio (PSNR)."""
    mse = np.mean((goc.astype(np.float64) - ket_qua.astype(np.float64))** 2)
    if mse == 0:
        return float("inf")
    # Công thức LaTeX: $$PSNR = 10 \cdot \log_{10} \left( \frac{MAX_I^2}{MSE} \right)$$
    return 10 * np.log10(255.0** 2 / mse)

# === BƯỚC 2: ÁP DỤNG CÁC BỘ LỌC ===
bo_loc = {}

# 1. Gaussian Blur (Truyền thống)
start = time.perf_counter()
bo_loc["Gaussian 5x5"] = cv2.GaussianBlur(img_nhieu, (5, 5), 1.5)
t_gauss = (time.perf_counter() - start) * 1000

# 2. Bilateral Filter (Giữ cạnh biên)
start = time.perf_counter()
bo_loc["Bilateral"] = cv2.bilateralFilter(img_nhieu, 9, 75, 75)
t_bilat = (time.perf_counter() - start) * 1000

# 3. Non-Local Means (NLM) - h=10 (Mức độ khử vừa phải)
# h: Tham số quyết định sức mạnh bộ lọc. h cao = mịn hơn nhưng mờ hơn.
start = time.perf_counter()
bo_loc["NLM h=10"] = cv2.fastNlMeansDenoising(img_nhieu, None, h=10, 
                                            templateWindowSize=7, searchWindowSize=21)
t_nlm = (time.perf_counter() - start) * 1000

# 4. Non-Local Means (NLM) - h=20 (Mức độ khử mạnh)
start = time.perf_counter()
bo_loc["NLM h=20"] = cv2.fastNlMeansDenoising(img_nhieu, None, h=20, 
                                            templateWindowSize=7, searchWindowSize=21)
t_nlm2 = (time.perf_counter() - start) * 1000

# === BƯỚC 3: PHÂN TÍCH HIỆU NĂNG ===
print(f"{'Bộ lọc':<20}{'PSNR (dB)':>12}{'Thời gian (ms)':>18}")
print("-" * 52)
print(f"{'Ảnh nhiễu':<20}{tinh_psnr(img, img_nhieu):>12.2f}{'—':>18}")

results_data = []
for ten, anh in bo_loc.items():
    psnr_val = tinh_psnr(img, anh)
    # Gán thời gian tương ứng
    thoi_gian = t_gauss if "Gaussian" in ten else t_bilat if "Bilateral" in ten else t_nlm if "h=10" in ten else t_nlm2
    print(f"{ten:<20}{psnr_val:>12.2f}{thoi_gian:>18.1f}")

# === BƯỚC 4: HIỂN THỊ ===
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes[0, 0].imshow(img, cmap="gray")
axes[0, 0].set_title("1. Ảnh gốc (Sạch)")
axes[0, 0].axis("off")

axes[0, 1].imshow(img_nhieu, cmap="gray")
axes[0, 1].set_title(f"2. Nhiễu Gaussian (σ=30)\nPSNR: {tinh_psnr(img, img_nhieu):.2f} dB")
axes[0, 1].axis("off")

# Vòng lặp hiển thị các bộ lọc
for idx, (ten, anh) in enumerate(bo_loc.items()):
    r, c = (idx + 2) // 3, (idx + 2) % 3
    axes[r, c].imshow(anh, cmap="gray")
    axes[r, c].set_title(f"{idx+3}. {ten}\nPSNR: {tinh_psnr(img, anh):.2f} dB")
    axes[r, c].axis("off")

plt.suptitle("So sánh Non-Local Means (NLM) với các bộ lọc cơ bản", fontsize=16)
plt.tight_layout()
os.makedirs("output", exist_ok=True)
plt.savefig("output/nlm_benchmark.png", dpi=150)
plt.show()