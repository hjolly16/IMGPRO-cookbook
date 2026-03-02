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

sigma = 1.5
alpha = 2.0  # Dùng alpha cao để dễ quan sát lỗi

# === BƯỚC 2: CÁC KỸ THUẬT LÀM NÉT CHỐNG RINGING ===

# 1. USM THÔNG THƯỜNG (Có hiện tượng Ringing/Halo)
blur_gauss = cv2.GaussianBlur(img, (0, 0), sigma)
detail_gauss = img.astype(np.float64) - blur_gauss.astype(np.float64)
img_usm = np.clip(img.astype(np.float64) + alpha * detail_gauss, 0, 255).astype(np.uint8)

# 2. CLAMPED SHARPENING (Giới hạn biên độ chi tiết)
# Giúp ngăn chặn các giá trị cực đoan gây ra quầng trắng/đen quá gắt
T_clamp = 30 
detail_clamped = np.clip(detail_gauss, -T_clamp, T_clamp)
img_clamped = np.clip(img.astype(np.float64) + alpha * detail_clamped, 0, 255).astype(np.uint8)

# 3. THRESHOLD SHARPENING (Chỉ làm nét vùng có chi tiết rõ rệt)
# Loại bỏ việc làm nét ở các vùng quá mịn (nơi chủ yếu là nhiễu)
T_min = 5
mask_threshold = np.abs(detail_gauss) > T_min
detail_thresh = detail_gauss * mask_threshold
img_thresh = np.clip(img.astype(np.float64) + alpha * detail_thresh, 0, 255).astype(np.uint8)

# 4. BILATERAL UNSHARP (Dùng bộ lọc song phương để giữ cạnh)
# Đây là cách "xịn" nhất vì Bilateral không làm mờ cạnh khi tạo bản đồ chi tiết
blur_bilat = cv2.bilateralFilter(img, 9, 75, 75)
detail_bilat = img.astype(np.float64) - blur_bilat.astype(np.float64)
img_bilat_usm = np.clip(img.astype(np.float64) + alpha * detail_bilat, 0, 255).astype(np.uint8)

# === BƯỚC 3: HIỂN THỊ VÀ SO SÁNH ===
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

axes[0, 0].imshow(img, cmap="gray")
axes[0, 0].set_title("1. Ảnh gốc")
axes[0, 0].axis("off")

# Vẽ khung đỏ đánh dấu vùng phóng to lên ảnh gốc
y, x, size = 150, 250, 80 # Tọa độ vùng muốn soi kỹ
rect = plt.Rectangle((x, y), size, size, linewidth=2, edgecolor='red', facecolor='none')
axes[0, 0].add_patch(rect)

axes[0, 1].imshow(img_usm, cmap="gray")
axes[0, 1].set_title(f"2. USM thường (α={alpha})\nLỗi Ringing rõ rệt")
axes[0, 1].axis("off")

axes[0, 2].imshow(img_clamped, cmap="gray")
axes[0, 2].set_title(f"3. Clamped (T={T_clamp})\nGiảm độ gắt của Halo")
axes[0, 2].axis("off")

axes[1, 0].imshow(img_thresh, cmap="gray")
axes[1, 0].set_title(f"4. Threshold (T={T_min})\nKhử nhiễu nền tốt")
axes[1, 0].axis("off")

axes[1, 1].imshow(img_bilat_usm, cmap="gray")
axes[1, 1].set_title("5. Bilateral USM\nLàm nét thông minh")
axes[1, 1].axis("off")

# Hiển thị vùng phóng to so sánh USM và Clamped
crop_usm = img_usm[y:y+size, x:x+size]
crop_clamp = img_clamped[y:y+size, x:x+size]
crop_combined = np.hstack([crop_usm, np.ones((size, 5), dtype=np.uint8)*255, crop_clamp])
axes[1, 2].imshow(crop_combined, cmap="gray")
axes[1, 2].set_title("Soi chi tiết: USM (Trái) vs Clamped (Phải)")
axes[1, 2].axis("off")

plt.tight_layout()
os.makedirs("output", exist_ok=True)
plt.savefig("output/so_sanh_chong_ringing.png", dpi=150)
plt.show()