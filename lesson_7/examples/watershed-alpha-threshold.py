import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# === BƯỚC 1: TẠO DỮ LIỆU MÔ PHỎNG CỤM VẬT THỂ DÍNH NHAU ===
img_bin = np.zeros((300, 400), dtype=np.uint8)
# Tạo 3 hình tròn chồng lấn lên nhau
cv2.circle(img_bin, (120, 150), 60, 255, -1)
cv2.circle(img_bin, (220, 150), 55, 255, -1)
cv2.circle(img_bin, (170, 80), 40, 255, -1)

# Watershed yêu cầu đầu vào là ảnh màu để vẽ biên
img_color = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR)

# === BƯỚC 2: TIỀN XỬ LÝ NỀN TẢNG ===
se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# Loại bỏ nhiễu biên
opening = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, se, iterations=2)
# Xác định vùng nền chắc chắn
sure_bg = cv2.dilate(opening, se, iterations=3)
# Tính toán bản đồ khoảng cách (Distance Transform)
dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

# === BƯỚC 3: KHẢO SÁT ẢNH HƯỞNG CỦA ALPHA ===
alphas = [0.1, 0.3, 0.4, 0.5, 0.6, 0.8]

fig, axes = plt.subplots(2, len(alphas), figsize=(20, 8))

# Tạo thư mục output
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

results_summary = []

for idx, alpha in enumerate(alphas):
    # 1. Tạo Sure Foreground dựa trên ngưỡng alpha
    _, sure_fg = cv2.threshold(dist, alpha * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    
    # 2. Xác định vùng chưa biết (Unknown)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 3. Đánh dấu hạt giống (Markers)
    # num_m bao gồm cả nhãn nền (background)
    num_m, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # 4. Thực thi Watershed trên bản sao của ảnh màu
    markers_ws = markers.copy()
    cv2.watershed(img_color.copy(), markers_ws)

    # Đếm số lượng đối tượng thực tế (trừ nền và biên)
    n_obj = len(np.unique(markers_ws)) - 2
    
    # Lưu thông tin để in bảng tổng kết
    num_seeds = num_m - 1
    note = "Over-seg" if n_obj > 3 else ("Chuẩn" if n_obj == 3 else "Under-seg")
    results_summary.append((alpha, num_seeds, n_obj, note))

    # --- Hiển thị Sure Foreground (Hạt giống) ---
    axes[0, idx].imshow(sure_fg, cmap="gray")
    axes[0, idx].set_title(f"Alpha = {alpha}\nSeeds: {num_seeds}")
    axes[0, idx].axis("off")

    # --- Hiển thị Kết quả phân đoạn ---
    overlay = img_color.copy()
    # Vẽ biên màu đỏ (giá trị -1 trong Watershed)
    overlay[markers_ws == -1] = [255, 0, 0]
    axes[1, idx].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1, idx].set_title(f"Kết quả: {n_obj} vùng\n({note})")
    axes[1, idx].axis("off")

plt.suptitle("Khảo sát ảnh hưởng của ngưỡng Distance (Alpha) lên thuật toán Watershed", fontsize=16)
plt.tight_layout()
plt.savefig(f"{output_dir}/watershed_alpha_analysis.png", dpi=150)
plt.show()

# === BƯỚC 4: IN BẢNG TỔNG KẾT ĐÁNH GIÁ ===
print(f"{'Alpha':>6} {'Seeds':>7} {'Đối tượng':>11} {'Nhận xét'}")
print("-" * 50)
for r in results_summary:
    print(f"{r[0]:>6.1f} {r[1]:>7} {r[2]:>11} {r[3]:>12}")

print("\n=> BÀI HỌC: Alpha quá thấp gây dính hạt giống, Alpha quá cao làm mất hạt giống.")