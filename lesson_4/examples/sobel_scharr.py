import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# === BƯỚC 1: ĐẦU VÀO ===
img_path = "images/sample.jpg"
if not os.path.exists(img_path):
    print(f"Lỗi: Không tìm thấy file tại {img_path}")
    exit()

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError("Không thể giải mã ảnh!")

# === BƯỚC 2: TÍNH TOÁN GRADIENT VỚI SOBEL ===
# Quan trọng: Dùng CV_64F để giữ lại giá trị âm (chuyển từ trắng sang đen)
# dx=1, dy=0: Đạo hàm theo phương X (Biên dọc)
sobel_x = cv2.Sobel(img, cv2.CV_64F, dx=1, dy=0, ksize=3)
# dx=0, dy=1: Đạo hàm theo phương Y (Biên ngang)
sobel_y = cv2.Sobel(img, cv2.CV_64F, dx=0, dy=1, ksize=3)

# Tính biên độ (Magnitude) và hướng (Direction) của Gradient
magnitude_sobel = np.sqrt(sobel_x**2 + sobel_y**2)
magnitude_sobel = np.clip(magnitude_sobel, 0, 255).astype(np.uint8)

# Hướng gradient (tính theo độ)
direction = np.arctan2(sobel_y, sobel_x) * 180 / np.pi

# === BƯỚC 3: TÍNH TOÁN VỚI TOÁN TỬ SCHARR ===
# Scharr cho kết quả chính xác hơn Sobel khi biên nằm nghiêng hoặc kernel nhỏ (3x3)
scharr_x = cv2.Scharr(img, cv2.CV_64F, dx=1, dy=0)
scharr_y = cv2.Scharr(img, cv2.CV_64F, dx=0, dy=1)

magnitude_scharr = np.sqrt(scharr_x**2 + scharr_y**2)
magnitude_scharr = np.clip(
    magnitude_scharr / magnitude_scharr.max() * 255, 0, 255
).astype(np.uint8)

# === BƯỚC 4: MINH HỌA SAI LẦM KHI DÙNG UINT8 TRỰC TIẾP ===
# Nếu ddepth = -1 hoặc uint8, các giá trị âm sẽ bị gán về 0 (Mất biên!)
sobel_x_sai = cv2.Sobel(img, -1, dx=1, dy=0, ksize=3)

# === BƯỚC 5: HIỂN THỊ ===
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Hiển thị ảnh gốc
axes[0, 0].imshow(img, cmap="gray")
axes[0, 0].set_title("1. Ảnh gốc")
axes[0, 0].axis("off")

# Sobel X (Biên dọc) - Dùng np.abs để hiển thị cả gradient âm và dương
axes[0, 1].imshow(np.abs(sobel_x), cmap="gray")
axes[0, 1].set_title("2. Sobel X (CV_64F)\nPhát hiện biên dọc")
axes[0, 1].axis("off")

# Sobel Y (Biên ngang)
axes[0, 2].imshow(np.abs(sobel_y), cmap="gray")
axes[0, 2].set_title("3. Sobel Y (CV_64F)\nPhát hiện biên ngang")
axes[0, 2].axis("off")

# Biên độ Sobel (Tổng hợp cả X và Y)
axes[1, 0].imshow(magnitude_sobel, cmap="gray")
axes[1, 0].set_title("4. Biên độ Sobel (Cạnh tổng quát)")
axes[1, 0].axis("off")

# Biên độ Scharr
axes[1, 1].imshow(magnitude_scharr, cmap="gray")
axes[1, 1].set_title("5. Biên độ Scharr\n(Chi tiết sắc nét hơn)")
axes[1, 1].axis("off")

# Minh họa lỗi uint8
axes[1, 2].imshow(sobel_x_sai, cmap="gray")
axes[1, 2].set_title("6. Sobel X (uint8 - SAI!)\nMất nửa thông tin biên")
axes[1, 2].axis("off")

plt.suptitle("Toán tử Gradient: Sobel & Scharr", fontsize=16)
plt.tight_layout()

# Lưu kết quả
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f"{output_dir}/gradient_analysis.png", dpi=150)
plt.show()