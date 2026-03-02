import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# === BƯỚC 1: ĐẦU VÀO & MÔ PHỎNG ÁNH SÁNG XẤU ===
img_path = "images/sample.jpg"
if not os.path.exists(img_path):
    print(f"Lỗi: Không tìm thấy ảnh tại {img_path}")
    exit()

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError("Không thể giải mã ảnh!")

# Mô phỏng ảnh có ánh sáng không đều (gradient từ trái sang phải)
rows, cols = img.shape
gradient = np.linspace(0.3, 1.2, cols)  # Hệ số nhân từ 0.3 đến 1.2
gradient = np.tile(gradient, (rows, 1))
img_khong_deu = np.clip(img.astype(np.float32) * gradient, 0, 255).astype(np.uint8)

# Lọc Gaussian để giảm nhiễu trước khi xử lý
img_blur = cv2.GaussianBlur(img_khong_deu, (5, 5), 0)

# === BƯỚC 2: PHÂN TÍCH THAM SỐ ADAPTIVE THRESHOLD ===
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# --- Hàng 1: Thay đổi Block Size (B), cố định C = 10 ---
axes[0, 0].imshow(img_khong_deu, cmap="gray")
axes[0, 0].set_title("1. Ảnh gốc (Lỗi ánh sáng)")
axes[0, 0].axis("off")

# Block Size phải là số lẻ: 7, 15, 31
for idx, B in enumerate([7, 15, 31]):
    # Công thức: T(x,y) = Mean(vùng BxB) - C
    ket_qua = cv2.adaptiveThreshold(
        img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=B, C=10
    )
    axes[0, idx + 1].imshow(ket_qua, cmap="gray")
    axes[0, idx + 1].set_title(f"B={B} (Vùng lân cận), C=10")
    axes[0, idx + 1].axis("off")

# --- Hàng 2: Thay đổi Hằng số (C), cố định B = 15 ---
# So sánh với Otsu để thấy sự thất bại của ngưỡng toàn cục
_, img_otsu = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
axes[1, 0].imshow(img_otsu, cmap="gray")
axes[1, 0].set_title("2. Otsu (Thất bại)")
axes[1, 0].axis("off")

for idx, C in enumerate([2, 10, 25]):
    ket_qua = cv2.adaptiveThreshold(
        img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=15, C=C
    )
    axes[1, idx + 1].imshow(ket_qua, cmap="gray")
    axes[1, idx + 1].set_title(f"B=15, C={C} (Độ chặt)")
    axes[1, idx + 1].axis("off")

plt.suptitle("Ảnh hưởng của tham số BlockSize (B) và Constant (C)", fontsize=16)
plt.tight_layout()

# Lưu kết quả
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f"{output_dir}/tham_so_adaptive.png", dpi=150)
plt.show()

print("\n=== NHẬN XÉT CHI TIẾT CHO TUTORIAL ===")
print("* BlockSize (B):")
print("  - Càng nhỏ: Càng nhạy với chi tiết li ti, nhưng dễ bị 'nhiễu muối tiêu'.")
print("  - Càng lớn: Mượt hơn, nhưng có thể làm dính các nét chữ hoặc mất chi tiết nhỏ.")
print("* Constant (C):")
print("  - Càng nhỏ (gần 0): Ngưỡng 'lỏng', giữ nhiều pixel trắng (background sạch nhưng chữ mỏng).")
print("  - Càng lớn: Ngưỡng 'chặt', chữ đậm hơn nhưng dễ làm xuất hiện các đốm đen nhiễu.")