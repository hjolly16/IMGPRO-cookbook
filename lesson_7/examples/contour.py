import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# === BƯỚC 1: ĐỌC ẢNH VÀ TIỀN XỬ LÝ ===
img_path = "images/sample.jpg"
img = cv2.imread(img_path)

if img is None:
    print(f"Lỗi: Không thể tải ảnh từ {img_path}")
    exit()

# Giảm kích thước nếu ảnh quá lớn để xử lý và hiển thị nhanh hơn
max_dim = 1000
h, w = img.shape[:2]
if max(h, w) > max_dim:
    scale = max_dim / max(h, w)
    img = cv2.resize(img, (int(w * scale), int(h * scale)))

# Chuyển sang ảnh xám
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Khử nhiễu
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Cải tiến thuật toán: Dùng Canny Edge Detection kết hợp Morphology
edges = cv2.Canny(blurred, 50, 150)
kernel = np.ones((3, 3), np.uint8)
closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# === BƯỚC 2: TÌM KIẾM VÀ LỌC CONTOUR ===
# RETR_EXTERNAL chỉ lấy các đường bao ngoài cùng, bỏ qua các lỗ rỗng bên trong
contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# === BƯỚC 3: PHÂN TÍCH THUỘC TÍNH HÌNH HỌC ===
min_area = 500  # Ngưỡng lọc nhiễu (loại bỏ các hạt nhỏ)
img_result = img.copy()

print(f"{'#':>3} {'Diện tích':>10} {'Chu vi':>8} {'Circularity':>13} {'Extent':>8} {'Solidity':>10} {'Hình dạng'}")
print("-" * 80)

doi_tuong_idx = 0
for c in contours:
    area = cv2.contourArea(c)
    if area < min_area:
        continue

    doi_tuong_idx += 1
    perimeter = cv2.arcLength(c, True)
    x, y, w, h = cv2.boundingRect(c)
    
    # Tính toán Convex Hull (Bao lồi)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)

    # Tính toán các tỷ số hình học
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
    extent = area / (w * h) if w * h > 0 else 0
    solidity = area / hull_area if hull_area > 0 else 0

    # Xấp xỉ đa giác để đếm số đỉnh (Epsilon = 4% chu vi)
    approx = cv2.approxPolyDP(c, 0.04 * perimeter, True)
    n_vertices = len(approx)

    # Thuật toán nhận diện dựa trên thuộc tính
    if n_vertices == 3:
        shape = "Tam giac"
    elif n_vertices == 4:
        # Có thể kiểm tra thêm Aspect Ratio để phân biệt Hình vuông/Chữ nhật
        shape = "Tu giac"
    elif circularity > 0.85:
        shape = "Hinh tron"
    elif 0.6 < circularity <= 0.85:
        shape = "Elip"
    else:
        shape = f"Phuc tap ({n_vertices} dinh)"

    # In kết quả phân tích ra bảng
    print(f"{doi_tuong_idx:>3} {area:>10.0f} {perimeter:>8.1f} {circularity:>13.3f} {extent:>8.3f} {solidity:>10.3f} {shape}")

    # Vẽ minh họa các lớp bao phủ
    color = tuple(np.random.randint(100, 255, 3).tolist())
    cv2.drawContours(img_result, [c], -1, color, 2)              # Đường bao chính
    cv2.drawContours(img_result, [hull], -1, (0, 255, 0), 1)     # Vẽ bao lồi màu xanh lá
    cv2.rectangle(img_result, (x, y), (x + w, y + h), (0, 0, 255), 1) # Vẽ Bounding Box màu đỏ
    
    cv2.putText(img_result, f"{doi_tuong_idx}:{shape}", (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# === BƯỚC 4: HIỂN THỊ ===
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title("1. Ảnh gốc (sample.jpg)")
axes[0].axis("off")

axes[1].imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
axes[1].set_title(f"2. Kết quả: {doi_tuong_idx} đối tượng được phân tích\n(Contour + Bbox + Hull)")
axes[1].axis("off")

plt.tight_layout()
os.makedirs("output", exist_ok=True)
plt.savefig("output/contour_shapes_analysis.png", dpi=150)
plt.show()