import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# === BƯỚC 1: TẢI ẢNH ĐẦU VÀO ===
img = cv2.imread("images/sample.jpg")
if img is None:
    raise FileNotFoundError("Không tìm thấy ảnh tại đường dẫn images/sample.jpg!")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]

# === BƯỚC 2: CÁC PHÉP BIẾN ĐỔI AFFINE TỪNG PHẦN ===

# 1. TỊNH TIẾN (Translation)
# Ma trận: [[1, 0, tx], [0, 1, ty]]
tx, ty = 100, 50
M_translate = np.float32([[1, 0, tx], [0, 1, ty]])
translated = cv2.warpAffine(img_rgb, M_translate, (w, h))

# 2. QUAY QUANH TÂM (Rotation - Không bị mất góc)
angle = 30
center = (w // 2, h // 2)

# Khởi tạo ma trận quay mặc định
M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)

# Tính toán kích thước mới để chứa toàn bộ ảnh sau khi quay (không bị cắt góc)
cos_a = abs(M_rot[0, 0])
sin_a = abs(M_rot[0, 1])
new_w = int(h * sin_a + w * cos_a)
new_h = int(h * cos_a + w * sin_a)

# Điều chỉnh ma trận quay để bù đắp phần tịnh tiến vào tâm mới
M_rot[0, 2] += (new_w - w) / 2
M_rot[1, 2] += (new_h - h) / 2
rotated = cv2.warpAffine(img_rgb, M_rot, (new_w, new_h))

# 3. BIẾN DẠNG (Shear)
# Làm nghiêng ảnh theo phương ngang (sx) hoặc phương dọc (sy)
sx = 0.3  # Shear ngang
M_shear = np.float32([[1, sx, 0], [0, 1, 0]])
sheared_w = int(w + abs(sx) * h)
sheared = cv2.warpAffine(img_rgb, M_shear, (sheared_w, h))

# 4. TỔ HỢP (Combination): QUAY + CO GIÃN
# Quay 15 độ và thu nhỏ còn 80% kích thước
M_combo = cv2.getRotationMatrix2D(center, 15, 0.8)
combo = cv2.warpAffine(img_rgb, M_combo, (w, h))

# 5. BIẾN ĐỔI AFFINE TỪ 3 CẶP ĐIỂM (3-Point Transform)
# Xác định ma trận khi biết vị trí thay đổi của 3 điểm mốc
pts_src = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
pts_dst = np.float32([[w * 0.1, h * 0.15],
                      [w * 0.85, h * 0.05],
                      [w * 0.15, h * 0.9]])
M_3pts = cv2.getAffineTransform(pts_src, pts_dst)
affine_3pts = cv2.warpAffine(img_rgb, M_3pts, (w, h))

# === BƯỚC 3: HIỂN THỊ KẾT QUẢ ===
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title("Ảnh Gốc")
axes[0, 0].axis("off")

axes[0, 1].imshow(translated)
axes[0, 1].set_title(f"Tịnh tiến (tx={tx}, ty={ty})")
axes[0, 1].axis("off")

axes[0, 2].imshow(rotated)
axes[0, 2].set_title(f"Quay {angle}° (Chuẩn - Không cắt)")
axes[0, 2].axis("off")

axes[1, 0].imshow(sheared)
axes[1, 0].set_title(f"Shear (sx={sx})")
axes[1, 0].axis("off")

axes[1, 1].imshow(combo)
axes[1, 1].set_title("Quay 15° + Thu nhỏ 0.8x")
axes[1, 1].axis("off")

axes[1, 2].imshow(affine_3pts)
axes[1, 2].set_title("Affine từ 3 cặp điểm mốc")
axes[1, 2].axis("off")

plt.suptitle("Các phép biến đổi Affine trong OpenCV", fontsize=16)
plt.tight_layout()

# Tạo thư mục output và lưu kết quả
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f"{output_dir}/affine_transforms.png", dpi=150)
plt.show()

# === BƯỚC 4: XUẤT THÔNG SỐ MA TRẬN ===
print("--- Ma trận Tịnh tiến ---")
print(M_translate)
print(f"\n--- Ma trận Quay {angle}° ---")
print(M_rot)
print(f"\n--- Ma trận Shear (sx={sx}) ---")
print(M_shear)