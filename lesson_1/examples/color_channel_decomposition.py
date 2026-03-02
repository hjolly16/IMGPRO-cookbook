import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# === BƯỚC 1: ĐẦU VÀO ===
img_path = "images/sample.jpg"
if not os.path.exists(img_path):
    print(f"Lỗi: Không tìm thấy file tại {img_path}")
    exit()

# Đọc ảnh bằng OpenCV (Mặc định là BGR)
img_bgr = cv2.imread(img_path)

# Chuyển sang RGB để hiển thị đúng trên Matplotlib
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# === BƯỚC 2: TÁCH KÊNH MÀU ===
# Cách 1: Dùng hàm của OpenCV (Trả về 3 mảng 2D)
B, G, R = cv2.split(img_bgr)

# Cách 2: Dùng NumPy Indexing (Thường nhanh hơn trong thực tế)
# B = img_bgr[:, :, 0]
# G = img_bgr[:, :, 1]
# R = img_bgr[:, :, 2]

# === BƯỚC 3: HIỂN THỊ SO SÁNH ===
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# --- Hàng 1: Phân biệt BGR và RGB ---
# Hiển thị trực tiếp biến BGR -> Sai màu 
axes[0, 0].imshow(img_bgr)
axes[0, 0].set_title("1. BGR (Hiển thị sai màu)")
axes[0, 0].axis("off")

# Hiển thị biến RGB -> Đúng màu chuẩn
axes[0, 1].imshow(img_rgb)
axes[0, 1].set_title("2. RGB (Hiển thị đúng màu)")
axes[0, 1].axis("off")

# Thủ thuật lật kênh nhanh bằng NumPy (không cần hàm cvtColor)
axes[0, 2].imshow(img_bgr[:, :, ::-1])
axes[0, 2].set_title("3. BGR[:,:,::-1] (Mẹo lật nhanh)")
axes[0, 2].axis("off")

# --- Hàng 2: Trực quan hóa từng kênh màu đơn lẻ ---
# Lưu ý: Bản chất mỗi kênh là một ảnh xám, ta dùng Colormap để dễ nhìn
axes[1, 0].imshow(B, cmap="Blues")
axes[1, 0].set_title("4. Kênh B (Blue)")
axes[1, 0].axis("off")

axes[1, 1].imshow(G, cmap="Greens")
axes[1, 1].set_title("5. Kênh G (Green)")
axes[1, 1].axis("off")

axes[1, 2].imshow(R, cmap="Reds")
axes[1, 2].set_title("6. Kênh R (Red)")
axes[1, 2].axis("off")

plt.tight_layout()

# === BƯỚC 4: LƯU KẾT QUẢ ===
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
plt.savefig(f"{output_dir}/kenh_mau_so_sanh.png", dpi=150)
print(f"✓ Đã lưu biểu đồ phân tích kênh màu vào {output_dir}/")
plt.show()