import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# === BƯỚC 1: CHUẨN BỊ ===
img_path = "images/sample.jpg"
if not os.path.exists(img_path):
    print(f"Lỗi: Không tìm thấy file tại {img_path}")
    exit()

img_bgr = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# === BƯỚC 2: CHUYỂN ĐỔI SANG CÁC KHÔNG GIAN MÀU ===
# HSV: Tách biệt Màu sắc (Hue), Độ bão hòa (Saturation) và Độ sáng (Value)
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
# LAB: Tách biệt Độ sáng (L) và các trục màu xanh-vàng (A), xanh-đỏ (B)
img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
# YCrCb: Dùng trong nén video, tách độ sáng (Y) và tín hiệu hiệu màu (Cr, Cb)
img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

# Kiểm tra phạm vi giá trị
print("=== Phạm vi giá trị từng không gian ===")
for ten, anh in [("BGR", img_bgr), ("HSV", img_hsv), ("LAB", img_lab), ("YCrCb", img_ycrcb)]:
    print(f"  {ten:6s}: shape={anh.shape}, "
          f"Kênh 0=[{anh[:,:,0].min():>3}, {anh[:,:,0].max():>3}], "
          f"Kênh 1=[{anh[:,:,1].min():>3}, {anh[:,:,1].max():>3}], "
          f"Kênh 2=[{anh[:,:,2].min():>3}, {anh[:,:,2].max():>3}]")

# === BƯỚC 3: PHÁT HIỆN MÀU ĐỎ BẰNG HSV ===
# Lưu ý: Trong OpenCV, Hue có phạm vi [0, 179]. Màu đỏ nằm ở hai đầu phổ (0-10 và 160-179)
# Định nghĩa ngưỡng dưới cho màu đỏ
lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])
# Định nghĩa ngưỡng trên cho màu đỏ
lower_red2 = np.array([170, 70, 50])
upper_red2 = np.array([179, 255, 255])

# Tạo mặt nạ (Mask)
mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
mask_do = cv2.bitwise_or(mask1, mask2)

# Áp dụng mặt nạ lên ảnh gốc để trích xuất màu đỏ
ket_qua = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_do)
ket_qua_rgb = cv2.cvtColor(ket_qua, cv2.COLOR_BGR2RGB)

# === BƯỚC 4: HIỂN THỊ ===
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(img_rgb)
axes[0].set_title("1. Ảnh gốc (RGB)")
axes[0].axis("off")

axes[1].imshow(mask_do, cmap="gray")
axes[1].set_title("2. Mặt nạ màu đỏ (Binary Mask)")
axes[1].axis("off")

axes[2].imshow(ket_qua_rgb)
axes[2].set_title("3. Kết quả lọc màu đỏ")
axes[2].axis("off")

plt.tight_layout()

# Lưu kết quả
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
plt.savefig(f"{output_dir}/phat_hien_mau_hsv.png", dpi=150)

print(f"\n✓ Đã lưu kết quả lọc màu vào '{output_dir}/phat_hien_mau_hsv.png'")
plt.show()