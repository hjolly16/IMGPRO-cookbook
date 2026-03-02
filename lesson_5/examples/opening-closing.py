import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# === BƯỚC 1: TẠO ẢNH CHUẨN (GROUND TRUTH) ===
img = np.zeros((300, 400), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (200, 250), 255, -1)
cv2.circle(img, (310, 150), 80, 255, -1)

# === BƯỚC 2: MÔ PHỎNG NHIỄU HAI LOẠI ===
img_noisy = img.copy()

# 1. Nhiễu trắng bên ngoài (Salt noise / Chấm nhiễu)
np.random.seed(42)
salt = np.random.rand(300, 400) < 0.008
img_noisy[salt & (img == 0)] = 255

# 2. Nhiễu đen bên trong (Pepper noise / Lỗ hổng)
pepper = np.random.rand(300, 400) < 0.02
img_noisy[pepper & (img == 255)] = 0

# Tạo phần tử cấu trúc (Structuring Element) hình Elip 5x5
se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# === BƯỚC 3: ÁP DỤNG CÁC PHÉP TOÁN HÌNH THÁI HỌC ===
# 1. Phép Mở (Opening): Xóa nhiễu trắng ngoài nền
opened = cv2.morphologyEx(img_noisy, cv2.MORPH_OPEN, se)

# 2. Phép Đóng (Closing): Lấp lỗ đen trong vật thể
closed = cv2.morphologyEx(img_noisy, cv2.MORPH_CLOSE, se)

# 3. Kết hợp tuần tự để loại bỏ cả 2 loại nhiễu
open_then_close = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, se)
close_then_open = cv2.morphologyEx(closed, cv2.MORPH_OPEN, se)

# === BƯỚC 4: ĐÁNH GIÁ CHẤT LƯỢNG (METRICS) ===
def iou(a, b):
    """Tính Intersection over Union (IoU) giữa 2 ảnh nhị phân."""
    a_bin = a > 0
    b_bin = b > 0
    intersection = np.sum(a_bin & b_bin)
    union = np.sum(a_bin | b_bin)
    return intersection / max(union, 1)

print("--- ĐÁNH GIÁ ĐỘ CHÍNH XÁC BẰNG IoU ---")
print(f"1. Ảnh bị nhiễu       : {iou(img_noisy, img):.4f}")
print(f"2. Chỉ dùng Opening   : {iou(opened, img):.4f} (Chỉ sạch nền, lỗ vẫn còn)")
print(f"3. Chỉ dùng Closing   : {iou(closed, img):.4f} (Chỉ kín lỗ, nền vẫn bẩn)")
print(f"4. Opening -> Closing : {iou(open_then_close, img):.4f} (Khuyên dùng)")
print(f"5. Closing -> Opening : {iou(close_then_open, img):.4f} (Khuyên dùng)")

# === BƯỚC 5: HIỂN THỊ KẾT QUẢ TRỰC QUAN ===
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

axes[0, 0].imshow(img, cmap="gray")
axes[0, 0].set_title("1. Ảnh gốc (Ground Truth)")
axes[0, 0].axis("off")

axes[0, 1].imshow(img_noisy, cmap="gray")
axes[0, 1].set_title("2. Ảnh nhiễu (Salt + Pepper)")
axes[0, 1].axis("off")

axes[0, 2].imshow(opened, cmap="gray")
axes[0, 2].set_title("3. Opening\n(Xóa đốm trắng, giữ lỗ đen)")
axes[0, 2].axis("off")

axes[1, 0].imshow(closed, cmap="gray")
axes[1, 0].set_title("4. Closing\n(Lấp lỗ đen, giữ đốm trắng)")
axes[1, 0].axis("off")

axes[1, 1].imshow(open_then_close, cmap="gray")
axes[1, 1].set_title("5. Opening → Closing\n(Sạch hoàn hảo)")
axes[1, 1].axis("off")

axes[1, 2].imshow(close_then_open, cmap="gray")
axes[1, 2].set_title("6. Closing → Opening\n(Sạch hoàn hảo)")
axes[1, 2].axis("off")

plt.suptitle("Làm sạch ảnh bằng Hình thái học (Opening & Closing)", fontsize=16)
plt.tight_layout()

# Lưu kết quả
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f"{output_dir}/morph_open_close.png", dpi=150)
plt.show()