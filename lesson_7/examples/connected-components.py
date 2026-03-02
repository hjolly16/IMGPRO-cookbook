import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# === BƯỚC 1: TẠO DỮ LIỆU MÔ PHỎNG (HẠT VÀ NHIỄU) ===
img = np.zeros((400, 600), dtype=np.uint8)

np.random.seed(42)
# Tạo các hạt lớn (Đối tượng mục tiêu cần giữ lại)
for _ in range(8):
    cx = np.random.randint(50, 550)
    cy = np.random.randint(50, 350)
    r = np.random.randint(20, 45)
    cv2.circle(img, (cx, cy), r, 255, -1)

# Tạo các hạt nhỏ li ti (Nhiễu cần loại bỏ)
for _ in range(50):
    cx = np.random.randint(0, 600)
    cy = np.random.randint(0, 400)
    r = np.random.randint(1, 6)
    cv2.circle(img, (cx, cy), r, 255, -1)

# === BƯỚC 2: PHÂN TÍCH THÀNH PHẦN LIÊN THÔNG VỚI THỐNG KÊ ===
# connectivity=8: Xét cả các pixel chạm nhau ở đường chéo
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    img, connectivity=8
)

print(f"Tổng số thành phần phát hiện (bao gồm nền): {num_labels}")
print(f"Số lượng đối tượng tiềm năng: {num_labels - 1}")

# === BƯỚC 3: LỌC ĐỐI TƯỢNG THEO DIỆN TÍCH (AREA FILTERING) ===
min_area = 200  # Ngưỡng diện tích tối thiểu để được coi là hạt "chuẩn"
max_area = 50000

# Khởi tạo ảnh kết quả
img_filtered = np.zeros_like(img)
img_colored = np.zeros((*img.shape, 3), dtype=np.uint8)
dem = 0

print(f"\n{'ID':>4} {'Diện tích':>10} {'Bounding Box (x,y,w,h)':>22} {'Trọng tâm':>16} {'Giữ?'}")
print("-" * 75)

# Duyệt qua các label, bắt đầu từ 1 (bỏ qua label 0 là phần nền đen)
for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    cx, cy = centroids[i]

    # Kiểm tra điều kiện lọc
    giu = min_area <= area <= max_area
    if giu:
        dem += 1
        # Tô màu ngẫu nhiên cho từng đối tượng được giữ lại
        color = np.random.randint(50, 255, 3).tolist()
        img_colored[labels == i] = color
        img_filtered[labels == i] = 255

    status = "✓" if giu else "x"
    print(f"{i:>4} {area:>10} ({x:>3},{y:>3},{w:>3},{h:>3}) ({cx:>6.1f},{cy:>6.1f}) {status:>6}")

print(f"\nKết quả sau lọc: Giữ lại {dem} đối tượng từ tổng số {num_labels - 1}")

# === BƯỚC 4: HIỂN THỊ TRỰC QUAN ===
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(img, cmap="gray")
axes[0].set_title(f"1. Ảnh gốc\n({num_labels - 1} đối tượng)")
axes[0].axis("off")

# Trực quan hóa bản đồ nhãn (Label Map) bằng Colormap nipy_spectral
# Mỗi pixel mang giá trị là ID của nó, giúp phân biệt ranh giới các vùng
label_vis = (labels / labels.max() * 255).astype(np.uint8) if labels.max() > 0 else labels
axes[1].imshow(label_vis, cmap="nipy_spectral")
axes[1].set_title("2. Bản đồ nhãn (Label Map)\nHiển thị mọi thành phần")
axes[1].axis("off")

axes[2].imshow(img_colored)
axes[2].set_title(f"3. Sau khi lọc diện tích\n(Area > {min_area} px)")
axes[2].axis("off")

plt.tight_layout()
os.makedirs("output", exist_ok=True)
plt.savefig("output/connected_components_analysis.png", dpi=150)
plt.show()