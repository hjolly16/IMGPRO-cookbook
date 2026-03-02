import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# === BƯỚC 1: TẠO ẢNH MÔ PHỎNG TÀI LIỆU LỖI ===
# Tạo nền xám (200)
img = np.ones((400, 500), dtype=np.uint8) * 200

# Viết chữ đen (mô phỏng văn bản)
cv2.putText(img, "OpenCV", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.5, 30, 5)
cv2.putText(img, "Morphology", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 40, 4)
cv2.putText(img, "Pipeline", (50, 330), cv2.FONT_HERSHEY_SIMPLEX, 2.2, 50, 4)

# Thêm gradient chiếu sáng (Sáng bên trái, tối dần sang phải)
for col in range(500):
    factor = 0.5 + 0.5 * (col / 500)
    img[:, col] = np.clip(img[:, col] * factor, 0, 255).astype(np.uint8)

# Thêm nhiễu Gaussian (Mô phỏng camera kém chất lượng)
nhieu = np.random.normal(0, 15, img.shape)
img = np.clip(img.astype(np.float64) + nhieu, 0, 255).astype(np.uint8)

# === BƯỚC 2: XÂY DỰNG PIPELINE XỬ LÝ ===
print("--- PIPELINE XỬ LÝ ẢNH NHỊ PHÂN HOÀN CHỈNH ---")

# Bước 1: Lọc Bilateral (Khử nhiễu nhưng giữ nguyên độ sắc nét của biên chữ)
buoc1 = cv2.bilateralFilter(img, 9, 75, 75)
print("✓ Bước 1: Bilateral filter - Khử nhiễu hạt")

# Bước 2: Black-Hat (Trích xuất chi tiết tối trên nền sáng không đều)
se_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
buoc2 = cv2.morphologyEx(buoc1, cv2.MORPH_BLACKHAT, se_big)
print("✓ Bước 2: Black-Hat (SE=25)  - Cân bằng chiếu sáng")

# Bước 3: Phân ngưỡng Otsu (Nhị phân hóa tự động)
_, buoc3 = cv2.threshold(buoc2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print("✓ Bước 3: Otsu Thresholding  - Nhị phân hóa")

# Bước 4: Opening (Loại bỏ các chấm nhiễu trắng li ti còn sót lại)
se_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
buoc4 = cv2.morphologyEx(buoc3, cv2.MORPH_OPEN, se_small)
print("✓ Bước 4: Opening (SE=3)     - Xóa chấm nhiễu nhỏ")

# Bước 5: Closing (Lấp các khe hở, lỗ thủng nhỏ bên trong nét chữ)
se_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
buoc5 = cv2.morphologyEx(buoc4, cv2.MORPH_CLOSE, se_close)
print("✓ Bước 5: Closing (SE=5)     - Lấp khe hở nét chữ")

# Bước 6: Morphological Gradient (Trích xuất đường biên của nét chữ)
se_grad = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
buoc6 = cv2.morphologyEx(buoc5, cv2.MORPH_GRADIENT, se_grad)
print("✓ Bước 6: Gradient (SE=3)    - Trích xuất đường biên")

# === BƯỚC 3: HIỂN THỊ TRỰC QUAN PIPELINE ===
ten_buoc = [
    "0. Ảnh Gốc (Nhiễu & Tối)", "1. Bilateral", "2. Black-Hat",
    "3. Otsu", "4. Opening", "5. Closing", "6. Gradient"
]
anh_buoc = [img, buoc1, buoc2, buoc3, buoc4, buoc5, buoc6]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes_flat = axes.flatten()

for idx in range(len(ten_buoc)):
    axes_flat[idx].imshow(anh_buoc[idx], cmap="gray")
    axes_flat[idx].set_title(ten_buoc[idx])
    axes_flat[idx].axis("off")

# Ẩn ô thừa cuối cùng
axes_flat[7].set_visible(False)

plt.suptitle("Pipeline Xử lý Hình thái học (Tiền xử lý OCR)", fontsize=18)
plt.tight_layout()

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f"{output_dir}/morph_pipeline_complete.png", dpi=150)
plt.show()

# === BƯỚC 4: THỐNG KÊ KẾT QUẢ ===
print(f"\n--- THỐNG KÊ ĐÁNH GIÁ ---")
print(f"Pixel trắng sau Otsu   : {np.sum(buoc3 > 0)}")
print(f"Pixel trắng sau Opening: {np.sum(buoc4 > 0)} (Giảm {np.sum(buoc3 > 0) - np.sum(buoc4 > 0)} pixel rác)")
print(f"Pixel trắng sau Closing: {np.sum(buoc5 > 0)} (Tăng {np.sum(buoc5 > 0) - np.sum(buoc4 > 0)} pixel lấp lỗ)")
print(f"Pixel viền (Gradient)  : {np.sum(buoc6 > 0)}")
print("=> Pipeline hoạt động xuất sắc: Xóa sạch rác, lấp kín lỗ thủng và trích xuất viền chữ hoàn hảo!")