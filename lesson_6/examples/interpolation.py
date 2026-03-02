import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# === BƯỚC 1: TẠO MASK NHỊ PHÂN MẪU ===
# Tạo Mask gốc kích thước 200x300 (H x W)
mask = np.zeros((200, 300), dtype=np.uint8)
# Vẽ một hình tròn và một hình chữ nhật trắng (255)
cv2.circle(mask, (150, 100), 60, 255, -1)
cv2.rectangle(mask, (30, 30), (90, 80), 255, -1)

# Mục tiêu: Phóng to Mask lên gấp đôi (600x400)
target_size = (600, 400) 

# === BƯỚC 2: BA PHƯƠNG PHÁP RESIZE MASK ===

# 1. Nearest (Gần nhất) — Lựa chọn CHUẨN cho Mask nhị phân
mask_nearest = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

# 2. Bilinear (Tuyến tính) — SAI LẦM: Gây ra hiện tượng nhòe biên (anti-aliasing)
mask_bilinear = cv2.resize(mask, target_size, interpolation=cv2.INTER_LINEAR)

# 3. Bilinear + Phân ngưỡng (Threshold) — Kỹ thuật nâng cao để làm mượt biên
mask_bilinear_thresh = cv2.resize(mask, target_size, interpolation=cv2.INTER_LINEAR)
_, mask_bilinear_thresh = cv2.threshold(mask_bilinear_thresh, 127, 255, cv2.THRESH_BINARY)

# === BƯỚC 3: PHÂN TÍCH GIÁ TRỊ PIXEL ===
vals_nearest = np.unique(mask_nearest)
vals_bilinear = np.unique(mask_bilinear)
vals_bl_thresh = np.unique(mask_bilinear_thresh)

print("=== KIỂM TRA GIÁ TRỊ PIXEL DUY NHẤT ===")
print(f"- Nearest:             {vals_nearest}  ← CHUẨN (Chỉ 0 và 255)")
print(f"- Bilinear:            {vals_bilinear[:5]}... ({len(vals_bilinear)} giá trị) ← SAI (Bị lẫn màu xám)")
print(f"- Bilinear + Threshold:{vals_bl_thresh}  ← CHUẨN")

# === BƯỚC 4: HIỂN THỊ KẾT QUẢ ===
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Hàng 1: Toàn cảnh Mask
axes[0, 0].imshow(mask, cmap="gray", interpolation="nearest")
axes[0, 0].set_title(f"Mask gốc (200x300)")
axes[0, 0].axis("off")

axes[0, 1].imshow(mask_nearest, cmap="gray", interpolation="nearest")
axes[0, 1].set_title(f"1. Nearest (Chuẩn)\n{len(vals_nearest)} giá trị pixel")
axes[0, 1].axis("off")

axes[0, 2].imshow(mask_bilinear, cmap="gray", interpolation="nearest")
axes[0, 2].set_title(f"2. Bilinear (Lỗi)\n{len(vals_bilinear)} giá trị pixel")
axes[0, 2].axis("off")

# --- Hàng 2: Zoom vào vùng biên để thấy sự khác biệt ---
y_start, y_end = 150, 250
x_start, x_end = 130, 230 # Cạnh trái nằm ở x=180, nằm giữa khoảng 130-230

axes[1, 0].imshow(mask_nearest[y_start:y_end, x_start:x_end], cmap="gray", interpolation="nearest")
axes[1, 0].set_title("Nearest (Zoom biên)\nBiên bậc thang rõ nét")
axes[1, 0].axis("off")

axes[1, 1].imshow(mask_bilinear[y_start:y_end, x_start:x_end], cmap="gray", interpolation="nearest")
axes[1, 1].set_title("Bilinear (Zoom biên)\nViền xám lan dần — SAI!")
axes[1, 1].axis("off")

axes[1, 2].imshow(mask_bilinear_thresh[y_start:y_end, x_start:x_end], cmap="gray", interpolation="nearest")
axes[1, 2].set_title("Bilinear + Threshold\nBiên mượt, bảo toàn nhị phân")
axes[1, 2].axis("off")

plt.suptitle("Resize Mask: Tầm quan trọng của việc chọn nội suy đúng", fontsize=16)
plt.tight_layout()

# Lưu kết quả
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f"{output_dir}/resize_mask.png", dpi=150)
plt.show()

# === BƯỚC 5: ĐÁNH GIÁ SỐ THÀNH PHẦN LIÊN THÔNG ===
n_nearest = cv2.connectedComponents(mask_nearest)[0] - 1
n_bilinear_raw = cv2.connectedComponents((mask_bilinear > 127).astype(np.uint8))[0] - 1

print(f"\n=== SỐ VẬT THỂ PHÁT HIỆN ===")
print(f"- Nearest:  {n_nearest}")
print(f"- Bilinear: {n_bilinear_raw}")