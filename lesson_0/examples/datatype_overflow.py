import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Tắt cảnh báo tràn số của NumPy để output trông sạch sẽ hơn (tùy chọn)
np.seterr(over='ignore')

# === PHẦN 1: MINH HỌA SỐ HỌC (SCALAR) ===
print("--- 1. BẢN CHẤT SỐ HỌC (UINT8) ---")
a = np.uint8(200)
b = np.uint8(100)

# NumPy: Modulo Arithmetic (200 + 100 = 300 -> 300 % 256 = 44)
res_np = a + b
print(f"NumPy (+)  : {a} + {b} = {res_np} (Tràn số - Modulo)")

# OpenCV: Saturation Arithmetic (200 + 100 = 300 -> Chặn tại 255)
# Lưu ý: Phải truyền vào mảng hoặc scalar kiểu tuple để tránh lỗi 'Bad argument'
res_cv2 = cv2.add(np.array([a]), np.array([b]))[0][0]
print(f"OpenCV add : {a} + {b} = {res_cv2} (Bão hòa - Saturation)")

print("-" * 30)

# === PHẦN 2: MINH HỌA TRÊN ẢNH THỰC TẾ ===
img_path = "images/sample.jpg"
if not os.path.exists(img_path):
    print(f"Lỗi: Không tìm thấy file {img_path}")
    exit()

# Đọc ảnh xám
img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img_gray is None:
    print("Lỗi: Không thể đọc ảnh!")
    exit()

# Thực hiện tăng độ sáng +100
# Cách dùng NumPy (Bị lỗi vùng sáng biến thành vùng tối)
img_overflow = img_gray + 100

# Cách dùng OpenCV (Chuẩn - Giữ nguyên độ sáng tối đa)
img_saturated = cv2.add(img_gray, 100)

# === PHẦN 3: HIỂN THỊ TRỰC QUAN ===
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Ảnh gốc
axes[0].imshow(img_gray, cmap="gray")
axes[0].set_title("1. Ảnh Gốc (Grayscale)")
axes[0].axis("off")

# Ảnh lỗi do NumPy
axes[1].imshow(img_overflow, cmap="gray")
axes[1].set_title("2. Lỗi Tràn Số (NumPy +)\nCác vùng sáng bị 'lật' thành đen")
axes[1].axis("off")

# Ảnh đúng do OpenCV
axes[2].imshow(img_saturated, cmap="gray")
axes[2].set_title("3. Bão Hòa Chuẩn (cv2.add)\nGiữ được độ trắng tối đa")
axes[2].axis("off")

plt.tight_layout()

# Lưu kết quả
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
plt.savefig(f"{output_dir}/demo_overflow.png", dpi=150)

print("\n✓ Đã xử lý xong! Hãy kiểm tra file 'demo_overflow.png'.")
plt.show()