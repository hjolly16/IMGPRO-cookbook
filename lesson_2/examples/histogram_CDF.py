import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# === BƯỚC 1: CHUẨN BỊ DỮ LIỆU ===
img_path = "images/sample.jpg"
if not os.path.exists(img_path):
    print(f"Lỗi: Không tìm thấy file tại {img_path}")
    exit()

# Đọc ảnh xám
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError("Không thể giải mã ảnh!")

# === BƯỚC 2: CÂN BẰNG LƯỢC ĐỒ (HISTOGRAM EQUALIZATION) ===
# 1. Cân bằng toàn cục (Global Histogram Equalization)
# Phương pháp này dàn đều các mức xám dựa trên toàn bộ ảnh.
img_equalized = cv2.equalizeHist(img)

# 2. Cân bằng cục bộ thích nghi (CLAHE)
# Chia ảnh thành các ô nhỏ (tile) để xử lý, tránh hiện tượng cháy sáng.
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
img_clahe = clahe.apply(img)

# === BƯỚC 3: HIỂN THỊ ẢNH VÀ BIỂU ĐỒ HISTOGRAM ===
fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# Cấu hình danh sách hiển thị để dùng vòng lặp cho gọn
data = [
    (img, "1. Ảnh gốc", "gray"),
    (img_equalized, "2. Cân bằng toàn cục (Global HE)", "blue"),
    (img_clahe, "3. Cân bằng cục bộ (CLAHE)", "green")
]

for i, (anh, tieu_de, mau) in enumerate(data):
    # Vẽ ảnh
    axes[i, 0].imshow(anh, cmap="gray")
    axes[i, 0].set_title(tieu_de)
    axes[i, 0].axis("off")
    
    # Vẽ Histogram
    axes[i, 1].hist(anh.ravel(), 256, [0, 256], color=mau, alpha=0.7)
    axes[i, 1].set_title(f"Lược đồ mức xám - {tieu_de}")
    axes[i, 1].set_xlim([0, 256])

plt.tight_layout()
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f"{output_dir}/so_sanh_histogram.png", dpi=150)
plt.show()

# === BƯỚC 4: VẼ LƯỢC ĐỒ TÍCH LŨY (CDF) ===
# CDF giúp ta thấy được cách các mức xám được phân phối lại như thế nào.
plt.figure(figsize=(10, 6))

for anh, ten, mau in data:
    # Tính Histogram
    hist = cv2.calcHist([anh], [0], None, [256], [0, 256]).ravel()
    # Tính tổng tích lũy
    cdf = hist.cumsum()
    # Chuẩn hóa về khoảng [0, 1] để so sánh
    cdf_normalized = cdf / cdf[-1]
    plt.plot(cdf_normalized, color=mau, label=ten, linewidth=2)

plt.xlabel("Mức cường độ (Pixel Value)")
plt.ylabel("Xác suất tích lũy (CDF)")
plt.title("So sánh Lược đồ tích lũy (CDF)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/so_sanh_cdf.png", dpi=150)
plt.show()

print("✓ Đã hoàn thành phân tích Histogram và CDF!")