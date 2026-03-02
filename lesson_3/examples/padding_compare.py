import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# === BƯỚC 1: ĐẦU VÀO ===
img_path = "images/sample.jpg"
if not os.path.exists(img_path):
    print(f"Lỗi: Không tìm thấy file tại {img_path}")
    exit()

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError("Không thể giải mã ảnh!")

# Cắt một vùng ảnh nhỏ (100x100) để dễ quan sát sự thay đổi ở biên
img_small = img[100:200, 100:200]
pad = 30  # Độ dày viền đắp thêm (30 pixel mỗi phía)

# === BƯỚC 2: CÁC PHƯƠNG PHÁP PADDING ===
phuong_phap = [
    ("1. CONSTANT (0)", cv2.BORDER_CONSTANT),     # Viền đen cố định
    ("2. REPLICATE", cv2.BORDER_REPLICATE),       # Lặp lại pixel biên
    ("3. REFLECT", cv2.BORDER_REFLECT),           # Đối xứng qua gương (cả biên)
    ("4. REFLECT_101", cv2.BORDER_REFLECT_101),   # Đối xứng (mặc định của filter2D)
    ("5. WRAP", cv2.BORDER_WRAP),                 # Cuộn ảnh (như lát gạch)
]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Hiển thị ảnh gốc ở ô đầu tiên
axes[0, 0].imshow(img_small, cmap="gray")
axes[0, 0].set_title("Ảnh gốc (100x100)")
axes[0, 0].axis("off")

# Vòng lặp áp dụng từng kiểu padding
for idx, (ten, kieu) in enumerate(phuong_phap):
    row, col = (idx + 1) // 3, (idx + 1) % 3
    ax = axes[row, col]
    
    # Thực hiện đắp viền
    if kieu == cv2.BORDER_CONSTANT:
        # Riêng kiểu CONSTANT có thể tùy chỉnh màu viền (value=0 là màu đen)
        img_pad = cv2.copyMakeBorder(img_small, pad, pad, pad, pad, kieu, value=0)
    else:
        img_pad = cv2.copyMakeBorder(img_small, pad, pad, pad, pad, kieu)
    
    ax.imshow(img_pad, cmap="gray")
    ax.set_title(ten)
    ax.axis("off")
    
    # Vẽ khung đỏ để đánh dấu vị trí ảnh gốc nằm trong ảnh đã đắp viền
    # Tọa độ (pad-0.5) để khung bao quanh chính xác pixel biên
    rect = plt.Rectangle((pad - 0.5, pad - 0.5), 100, 100, 
                        linewidth=2, edgecolor="red", facecolor="none")
    ax.add_patch(rect)

plt.suptitle("So sánh các phương pháp Padding (Viền đỏ = Vị trí ảnh gốc)", fontsize=16)
plt.tight_layout()

# === BƯỚC 3: LƯU KẾT QUẢ ===
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f"{output_dir}/padding_so_sanh.png", dpi=150)
plt.show()

print("✓ Đã hoàn thành minh họa các phương pháp Padding!")