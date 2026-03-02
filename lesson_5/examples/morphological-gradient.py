import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# === BƯỚC 1: TẠO ẢNH NHỊ PHÂN MẪU ===
img = np.zeros((300, 400), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (180, 200), 255, -1)
cv2.circle(img, (300, 150), 70, 255, -1)
# Vẽ thêm tam giác
pts = np.array([[130, 230], [80, 290], [180, 290]], dtype=np.int32)
cv2.fillPoly(img, [pts], 255)

# Phần tử cấu trúc (SE) hình Elip 3x3
se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# === BƯỚC 2: BA LOẠI GRADIENT HÌNH THÁI HỌC ===
# 1. Gradient tiêu chuẩn (Standard Gradient) = Giãn - Co
grad_std = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, se)

# 2. Gradient ngoài (External Gradient) = Giãn - Gốc
dilated = cv2.dilate(img, se, iterations=1)
grad_ext = cv2.subtract(dilated, img)

# 3. Gradient trong (Internal Gradient) = Gốc - Co
eroded = cv2.erode(img, se, iterations=1)
grad_int = cv2.subtract(img, eroded)

# === BƯỚC 3: SO SÁNH VỚI THUẬT TOÁN CANNY ===
# Canny hoạt động dựa trên đạo hàm cường độ sáng
canny = cv2.Canny(img, 100, 200)

# === BƯỚC 4: HIỂN THỊ TRỰC QUAN ===
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

axes[0, 0].imshow(img, cmap="gray")
axes[0, 0].set_title("1. Ảnh nhị phân gốc")
axes[0, 0].axis("off")

axes[0, 1].imshow(grad_std, cmap="gray")
axes[0, 1].set_title(f"2. Gradient Tiêu chuẩn (Giãn - Co)\nSố pixel: {np.sum(grad_std > 0)}")
axes[0, 1].axis("off")

axes[0, 2].imshow(grad_ext, cmap="gray")
axes[0, 2].set_title(f"3. Gradient Ngoài (Giãn - Gốc)\nSố pixel: {np.sum(grad_ext > 0)}")
axes[0, 2].axis("off")

axes[1, 0].imshow(grad_int, cmap="gray")
axes[1, 0].set_title(f"4. Gradient Trong (Gốc - Co)\nSố pixel: {np.sum(grad_int > 0)}")
axes[1, 0].axis("off")

axes[1, 1].imshow(canny, cmap="gray")
axes[1, 1].set_title(f"5. Thuật toán Canny\nSố pixel: {np.sum(canny > 0)}")
axes[1, 1].axis("off")

# Overlay gradient ngoài lên ảnh gốc (Mô phỏng vẽ viền bao quanh)
overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
overlay[grad_ext > 0] = [255, 0, 0]  # Vẽ viền màu đỏ
axes[1, 2].imshow(overlay)
axes[1, 2].set_title("6. Overlay Gradient Ngoài\n(Viền đỏ bao quanh vật thể)")
axes[1, 2].axis("off")

plt.suptitle("So sánh Gradient Hình thái học và Canny Edge Detection", fontsize=16)
plt.tight_layout()

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f"{output_dir}/morph_gradient_comparison.png", dpi=150)
plt.show()

# In thống kê
print(f"--- THỐNG KÊ SỐ PIXEL BIÊN ---")
print(f"Gradient tiêu chuẩn : {np.sum(grad_std > 0)}")
print(f"Gradient ngoài      : {np.sum(grad_ext > 0)}")
print(f"Gradient trong      : {np.sum(grad_int > 0)}")
print(f"Thuật toán Canny    : {np.sum(canny > 0)}")
print(f"\n=> Kiểm chứng: Tiêu chuẩn ({np.sum(grad_std > 0)}) ≈ Ngoài + Trong ({np.sum(grad_ext > 0) + np.sum(grad_int > 0)})")