import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def tang_tuong_phan_clahe(img_bgr, clip_limit=2.0, tile_size=(8,8)):
    """
    Tăng tương phản bằng CLAHE trên kênh L (Lightness) của không gian LAB.
    Cách này giúp tăng độ nét mà không làm biến đổi màu sắc gốc của ảnh.
    """
    # Bước 1: Chuyển sang không gian màu LAB
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    # Bước 2: Tách các kênh màu
    # L: Độ sáng (Lightness), A & B: Thông tin màu sắc
    L, A, B = cv2.split(img_lab)

    # Bước 3: Khởi tạo và áp dụng CLAHE lên kênh L
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    L_clahe = clahe.apply(L)

    # Bước 4: Ghép các kênh lại và chuyển ngược về BGR
    img_lab_clahe = cv2.merge([L_clahe, A, B])
    ket_qua = cv2.cvtColor(img_lab_clahe, cv2.COLOR_LAB2BGR)

    return ket_qua

# === CHƯƠNG TRÌNH CHÍNH ===
img_path = "images/sample.jpg"
if not os.path.exists(img_path):
    print(f"Lỗi: Không tìm thấy ảnh tại {img_path}")
    exit()

img_bgr = cv2.imread(img_path)

# 1. Áp dụng CLAHE lên ảnh màu (thông qua không gian LAB)
img_clahe_color = tang_tuong_phan_clahe(img_bgr, clip_limit=3.0)

# 2. Áp dụng CLAHE lên ảnh xám để so sánh
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
clahe_obj = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
img_clahe_gray = clahe_obj.apply(img_gray)

# === HIỂN THỊ KẾT QUẢ ===
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Ảnh gốc
axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
axes[0].set_title("1. Ảnh gốc")
axes[0].axis("off")

# CLAHE trên ảnh xám
axes[1].imshow(img_clahe_gray, cmap="gray")
axes[1].set_title("2. CLAHE (Ảnh xám)")
axes[1].axis("off")

# CLAHE trên LAB (Ảnh màu)
axes[2].imshow(cv2.cvtColor(img_clahe_color, cv2.COLOR_BGR2RGB))
axes[2].set_title("3. CLAHE trên kênh L (LAB)\nGiữ màu, tăng chi tiết")
axes[2].axis("off")

plt.tight_layout()

# Lưu kết quả
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
plt.savefig(f"{output_dir}/clahe_comparison.png", dpi=150)

print("✓ Đã xử lý CLAHE thành công! Ảnh màu trông sẽ sắc nét và rõ chi tiết vùng tối hơn.")
plt.show()