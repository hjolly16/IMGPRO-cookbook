import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# === BƯỚC 1: CHUẨN BỊ ===
img_path = "images/sample.jpg"
if not os.path.exists(img_path):
    print(f"Lỗi: Không tìm thấy ảnh tại {img_path}")
    exit()

# Đọc ảnh ở dạng ảnh xám (Grayscale)
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError("Không thể giải mã ảnh!")

# === BƯỚC 2: BIẾN ĐỔI TUYẾN TÍNH (LINEAR TRANSFORMATION) ===
# Công thức: g(x,y) = alpha * f(x,y) + beta
# alpha > 1: tăng tương phản | beta: tăng/giảm độ sáng
alpha, beta = 1.5, 30
img_linear = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

# === BƯỚC 3: HIỆU CHỈNH GAMMA (POWER-LAW TRANSFORMATION) ===
def gamma_correction(image, gamma):
    """
    Hiệu chỉnh gamma bằng Look-Up Table (LUT).
    Sử dụng NumPy để vector hóa quá trình tính toán bảng tra.
    """
    # Tạo mảng giá trị từ 0 đến 255
    inv_gamma = gamma # Gamma thường được hiểu là số mũ trực tiếp
    
    # Xây dựng bảng tra (LUT)
    # v_out = 255 * (v_in / 255) ^ gamma
    lut = np.clip(255 * (np.arange(256) / 255.0) ** gamma, 0, 255).astype(np.uint8)
    
    return cv2.LUT(image, lut)

img_gamma_05 = gamma_correction(img, 0.5)  # Làm sáng (nén vùng tối, giãn vùng sáng)
img_gamma_22 = gamma_correction(img, 2.2)  # Làm tối (nén vùng sáng, giãn vùng tối)

# === BƯỚC 4: CO GIÃN TƯƠNG PHẢN (CONTRAST STRETCHING) ===
# Sử dụng Percentile để loại bỏ các điểm ảnh nhiễu cực sáng/cực tối (outliers)
p2, p98 = np.percentile(img, (2, 98))
img_stretch = np.clip((img.astype(np.float32) - p2) / (p98 - p2) * 255, 0, 255)
img_stretch = img_stretch.astype(np.uint8)

# === BƯỚC 5: HIỂN THỊ ===
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
danh_sach = [
    (img, "1. Ảnh Gốc"),
    (img_linear, f"2. Tuyến tính (α={alpha}, β={beta})"),
    (img_stretch, "3. Co giãn (Percentile 2-98)"),
    (img_gamma_05, "4. Gamma = 0.5 (Làm sáng)"),
    (img_gamma_22, "5. Gamma = 2.2 (Làm tối)"),
]

for i, ax in enumerate(axes.flat):
    if i < len(danh_sach):
        anh, tieu_de = danh_sach[i]
        ax.imshow(anh, cmap="gray", vmin=0, vmax=255)
        ax.set_title(tieu_de)
    ax.axis("off")

plt.tight_layout()

# Lưu kết quả
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f"{output_dir}/bien_doi_cuong_do.png", dpi=150)

print("✓ Đã thực hiện các biến đổi cường độ sáng!")
plt.show()