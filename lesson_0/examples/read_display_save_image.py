import cv2
import matplotlib.pyplot as plt
import os

# === BƯỚC 0: CHUẨN BỊ THƯ MỤC ===
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# === BƯỚC 1: ĐẦU VÀO ===
img_path = "images/sample.jpg"

# Kiểm tra file có tồn tại không
if not os.path.exists(img_path):
    print(f"Lỗi: Không tìm thấy file tại {img_path}")
    exit()

# Đọc ảnh màu
img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)

if img_bgr is None:
    print("Lỗi: Không thể giải mã ảnh.")
    exit()

print(f"Kích thước: {img_bgr.shape}")
print(f"Kiểu dữ liệu: {img_bgr.dtype}")

# Đọc ảnh xám
img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img_gray is None:
    print("Lỗi: Không thể đọc ảnh xám.")
    exit()

print(f"Ảnh xám - Kích thước: {img_gray.shape}")

# === BƯỚC 2: CHUYỂN ĐỔI KHÔNG GIAN MÀU ===
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# === BƯỚC 3: HIỂN THỊ ===
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img_bgr)
axes[0].set_title("1. BGR (Sai màu trên MPL)")
axes[0].axis("off")

axes[1].imshow(img_rgb)
axes[1].set_title("2. RGB (Đúng màu)")
axes[1].axis("off")

axes[2].imshow(img_gray, cmap="gray")
axes[2].set_title("3. Ảnh xám")
axes[2].axis("off")

plt.tight_layout()
plt.savefig(f"{output_dir}/so_sanh_bgr_rgb.png", dpi=150)
plt.show()

# === BƯỚC 4: LƯU ẢNH ===
success = cv2.imwrite(f"{output_dir}/anh_rgb.png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
if success:
    print(f"Đã lưu ảnh RGB vào {output_dir}/anh_rgb.png")
