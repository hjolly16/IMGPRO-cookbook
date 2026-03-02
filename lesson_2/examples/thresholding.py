import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# === BƯỚC 1: ĐẦU VÀO & TIỀN XỬ LÝ ===
img_path = "images/sample.jpg"
if not os.path.exists(img_path):
    print(f"Lỗi: Không tìm thấy ảnh tại {img_path}")
    exit()

# Đọc ảnh xám
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError("Không thể giải mã ảnh!")

# Tiền xử lý: Lọc Gaussian nhẹ (5x5) để làm mượt ảnh, giảm nhiễu hạt
# Bước này giúp việc phân ngưỡng chính xác hơn, tránh các đốm nhiễu nhỏ
img_blur = cv2.GaussianBlur(img, (5, 5), 0)

# === BƯỚC 2: CÁC KỸ THUẬT PHÂN NGƯỠNG ===

# 1. Ngưỡng cố định (Simple Thresholding)
# Tất cả pixel > 127 sẽ thành 255 (trắng), ngược lại thành 0 (đen)
_, img_fixed = cv2.threshold(img_blur, 127, 255, cv2.THRESH_BINARY)

# 2. Ngưỡng tự động Otsu (Otsu's Thresholding)
# Thuật toán tự tính toán ngưỡng tối ưu dựa trên lược đồ (histogram) của ảnh
nguong_otsu, img_otsu = cv2.threshold(
    img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
print(f"✓ Ngưỡng Otsu tự động tìm được: {nguong_otsu}")

# 3. Ngưỡng thích nghi trung bình (Adaptive Mean Thresholding)
# Ngưỡng tại mỗi vùng nhỏ (15x15) là trung bình cộng của vùng đó trừ đi hằng số C
img_adapt_mean = cv2.adaptiveThreshold(
    img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
    cv2.THRESH_BINARY, blockSize=15, C=10
)

# 4. Ngưỡng thích nghi Gaussian (Adaptive Gaussian Thresholding)
# Tương tự Mean nhưng tính trung bình có trọng số theo phân phối Gaussian
img_adapt_gauss = cv2.adaptiveThreshold(
    img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv2.THRESH_BINARY, blockSize=15, C=10
)

# === BƯỚC 3: HIỂN THỊ SO SÁNH ===
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Hàng 1
axes[0, 0].imshow(img, cmap="gray")
axes[0, 0].set_title("1. Ảnh gốc")
axes[0, 0].axis("off")

# Vẽ Histogram và vị trí các ngưỡng
axes[0, 1].hist(img.ravel(), 256, [0, 256], color="gray", alpha=0.6)
axes[0, 1].axvline(x=127, color="red", linestyle="--", label="T=127")
axes[0, 1].axvline(x=nguong_otsu, color="blue", linestyle="--", label=f"Otsu={nguong_otsu:.0f}")
axes[0, 1].set_title("2. Lược đồ & Vị trí ngưỡng")
axes[0, 1].legend()

axes[0, 2].imshow(img_fixed, cmap="gray")
axes[0, 2].set_title("3. Ngưỡng cố định (T=127)")
axes[0, 2].axis("off")

# Hàng 2
axes[1, 0].imshow(img_otsu, cmap="gray")
axes[1, 0].set_title(f"4. Otsu (T={nguong_otsu:.0f})")
axes[1, 0].axis("off")

axes[1, 1].imshow(img_adapt_mean, cmap="gray")
axes[1, 1].set_title("5. Adaptive Mean")
axes[1, 1].axis("off")

axes[1, 2].imshow(img_adapt_gauss, cmap="gray")
axes[1, 2].set_title("6. Adaptive Gaussian")
axes[1, 2].axis("off")

plt.tight_layout()
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f"{output_dir}/so_sanh_phan_nguong.png", dpi=150)
plt.show()