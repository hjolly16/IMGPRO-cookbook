import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# === BƯỚC 1: TẠO ẢNH MÔ PHỎNG CÁC VẬT THỂ DÍNH NHAU ===
img_bin = np.zeros((400, 500), dtype=np.uint8)

# Tạo hai hình tròn dính nhau bên trái
cv2.circle(img_bin, (150, 200), 70, 255, -1)
cv2.circle(img_bin, (260, 200), 65, 255, -1)

# Tạo cụm ba hình tròn dính nhau bên phải
cv2.circle(img_bin, (400, 120), 50, 255, -1)
cv2.circle(img_bin, (430, 200), 45, 255, -1)
cv2.circle(img_bin, (380, 280), 55, 255, -1)

# Watershed yêu cầu ảnh màu (BGR) để đánh dấu biên
img_color = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR)

# === BƯỚC 2: PIPELINE XỬ LÝ WATERSHED ===

# 1. Tiền xử lý: Sử dụng Opening để loại bỏ nhiễu nhỏ
se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
opening = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, se, iterations=2)

# 2. Xác định vùng chắc chắn là Nền (Sure Background) bằng phép Giãn
sure_bg = cv2.dilate(opening, se, iterations=3)

# 3. Sử dụng Distance Transform để tính khoảng cách từ mỗi pixel trắng đến pixel đen gần nhất
# Các pixel ở trung tâm vật thể sẽ có giá trị lớn nhất (đỉnh núi)
dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

# 4. Xác định vùng chắc chắn là Vật thể (Sure Foreground) bằng cách lấy ngưỡng Distance
alpha = 0.5  # Ngưỡng quyết định độ tách biệt (càng lớn càng dễ tách nhưng vật thể nhỏ dễ biến mất)
_, sure_fg = cv2.threshold(dist, alpha * dist.max(), 255, 0)
sure_fg = sure_fg.astype(np.uint8)

# 5. Xác định vùng chưa rõ ràng (Unknown Region) - Nơi thuật toán sẽ tìm đường biên
unknown = cv2.subtract(sure_bg, sure_fg)

# 6. Tạo nhãn (Markers) cho các hạt giống (Seeds)
num_markers, markers = cv2.connectedComponents(sure_fg)

# Chỉnh sửa Markers: Phông nền phải là 1, các đối tượng là 2, 3... 
# Vùng chưa biết (Unknown) phải được gán là 0 để Watershed tính toán
markers = markers + 1
markers[unknown == 255] = 0

print(f"Số lượng hạt giống (seeds) tìm thấy: {num_markers}")

# 7. Thực thi thuật toán Watershed
markers_result = markers.copy()
cv2.watershed(img_color, markers_result)

# === BƯỚC 3: HIỂN THỊ VÀ MINH HỌA QUY TRÌNH ===
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# --- Hàng 1: Quy trình phân tích ---
axes[0, 0].imshow(img_bin, cmap="gray")
axes[0, 0].set_title("1. Ảnh nhị phân gốc\n(Các khối đang bị dính)")
axes[0, 0].axis("off")

axes[0, 1].imshow(dist, cmap="jet")
axes[0, 1].set_title("2. Distance Transform\n(Tìm tâm vật thể)")
axes[0, 1].axis("off")

axes[0, 2].imshow(sure_fg, cmap="gray")
axes[0, 2].set_title(f"3. Sure Foreground\n(Hạt giống - alpha={alpha})")
axes[0, 2].axis("off")

axes[0, 3].imshow(unknown, cmap="gray")
axes[0, 3].set_title("4. Unknown Region\n(Vùng tìm biên)")
axes[0, 3].axis("off")

# --- Hàng 2: Kết quả thực thi ---
# Trực quan hóa Markers trước khi tràn nước
markers_vis = (markers / markers.max() * 255).astype(np.uint8) if markers.max() > 0 else markers.astype(np.uint8)
axes[1, 0].imshow(markers_vis, cmap="nipy_spectral")
axes[1, 0].set_title(f"5. Markers gốc\n({num_markers} hạt giống)")
axes[1, 0].axis("off")

# Kết quả sau khi tràn nước (mỗi đối tượng 1 nhãn màu)
markers_show = markers_result.copy().astype(np.float64)
markers_show[markers_show == -1] = 0 # Gán biên về 0 để dễ hiển thị
markers_show = (markers_show / markers_show.max() * 255).astype(np.uint8) if markers_show.max() > 0 else markers_show.astype(np.uint8)
axes[1, 1].imshow(markers_show, cmap="nipy_spectral")
axes[1, 1].set_title("6. Kết quả Watershed\n(Đã tách vùng)")
axes[1, 1].axis("off")

# Vẽ đường phân tách lên ảnh màu
img_overlay = img_color.copy()
img_overlay[markers_result == -1] = [255, 0, 0] # Đường biên màu đỏ
axes[1, 2].imshow(cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title("7. Đường biên phân tách")
axes[1, 2].axis("off")

# Thống kê kết quả
unique_labels = np.unique(markers_result)
n_objects = len(unique_labels) - 2 # Trừ đi nhãn nền (1) và nhãn biên (-1)
axes[1, 3].text(0.5, 0.5, f"KẾT QUẢ:\n\n{n_objects} đối tượng\nđã được tách rời", 
                ha="center", va="center", fontsize=15, fontweight='bold', transform=axes[1, 3].transAxes)
axes[1, 3].axis("off")

plt.suptitle("Kỹ thuật Watershed: Tách đối tượng chồng lấn", fontsize=18)
plt.tight_layout()

os.makedirs("output", exist_ok=True)
plt.savefig("output/watershed_analysis.png", dpi=150)
plt.show()