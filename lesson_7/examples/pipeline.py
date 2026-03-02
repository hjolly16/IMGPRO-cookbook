import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def dem_doi_tuong(img_gray, min_area=100, max_area=50000, 
                   min_circularity=0.0, use_watershed=False):
    """
    Hàm thực hiện quy trình đếm đối tượng chuẩn: 
    Nhị phân -> Lọc nhiễu -> Phân đoạn -> Đo lường.
    """
    # Bước 1: Tiền xử lý - Làm mờ và Nhị phân hóa (Otsu)
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Bước 2: Morphology - Sử dụng Opening/Closing để dọn nhiễu
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, se, iterations=2)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, se, iterations=2)

    if use_watershed:
        # Bước 3: Phân đoạn nâng cao bằng Watershed
        sure_bg = cv2.dilate(clean, se, iterations=3)
        dist = cv2.distanceTransform(clean, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)
        unknown = cv2.subtract(sure_bg, sure_fg)

        n_markers, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        cv2.watershed(img_color, markers)
        labels_to_process = markers
    else:
        # Bước 3: Phân đoạn đơn giản bằng Connected Components
        _, labels_to_process, _, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)

    # Bước 4: Lọc và đo lường thuộc tính
    img_result = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    stats_list = []
    count = 0
    unique_labels = np.unique(labels_to_process)

    for label in unique_labels:
        # Loại bỏ nhãn nền (0) và nhãn biên (-1) của Watershed
        if label <= 0 or (use_watershed and label == 1):
            continue

        obj_mask = (labels_to_process == label).astype(np.uint8) * 255
        area = np.sum(obj_mask > 0)

        # Lọc theo diện tích
        if area < min_area or area > max_area:
            continue

        # Tìm Contour để tính chu vi và độ tròn
        cnt_list, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnt_list: continue

        cnt = max(cnt_list, key=cv2.contourArea)
        perimeter = cv2.arcLength(cnt, True)
        # Circularity = 1.0 là hình tròn hoàn hảo
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

        # Lọc theo độ tròn (ví dụ chỉ đếm các hạt tròn)
        if circularity < min_circularity:
            continue

        count += 1
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2

        stats_list.append({
            "id": count, "area": area, "circularity": circularity, 
            "bbox": (x, y, w, h), "center": (cx, cy)
        })

        # Vẽ kết quả lên ảnh
        color = tuple(np.random.randint(100, 255, 3).tolist())
        cv2.drawContours(img_result, [cnt], -1, color, 2)
        cv2.putText(img_result, str(count), (cx - 10, cy + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return count, stats_list, img_result

# === DEMO THỰC TẾ ===
# Tạo ảnh mẫu mô phỏng các hạt dính nhau và nhiễu
img_test = np.zeros((400, 600), dtype=np.uint8)
np.random.seed(42)
for _ in range(12):
    cx, cy = np.random.randint(40, 560), np.random.randint(40, 360)
    r = np.random.randint(15, 40)
    cv2.circle(img_test, (cx, cy), r, 220, -1)

# Thêm nhiễu Gaussian
noise = np.random.normal(0, 15, img_test.shape).astype(np.uint8)
img_test = cv2.add(img_test, noise)

# So sánh 2 phương pháp
n1, stats1, res1 = dem_doi_tuong(img_test, min_area=200, use_watershed=False)
n2, stats2, res2 = dem_doi_tuong(img_test, min_area=200, use_watershed=True)

# Hiển thị
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
axes[0].imshow(img_test, cmap="gray"); axes[0].set_title("1. Ảnh gốc (Có nhiễu & Dính hạt)"); axes[0].axis("off")
axes[1].imshow(cv2.cvtColor(res1, cv2.COLOR_BGR2RGB)); axes[1].set_title(f"2. CC đơn giản: {n1} cụm"); axes[1].axis("off")
axes[2].imshow(cv2.cvtColor(res2, cv2.COLOR_BGR2RGB)); axes[2].set_title(f"3. Watershed: {n2} hạt"); axes[2].axis("off")

plt.suptitle("Pipeline Đếm Đối Tượng: Sức mạnh của sự kết hợp thuật toán", fontsize=16)
plt.tight_layout()
os.makedirs("output", exist_ok=True)
plt.savefig("output/dem_doi_tuong_final.png", dpi=150)
plt.show()

# In bảng thống kê
print(f"{'ID':>3} {'Diện tích':>10} {'Circularity':>13} {'Tâm':>14}")
print("-" * 50)
for s in stats2[:5]: # In 5 hạt đầu tiên làm mẫu
    print(f"{s['id']:>3} {s['area']:>10} {s['circularity']:>13.3f} {str(s['center']):>14}")