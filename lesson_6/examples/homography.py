import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Polygon

# Tạo thư mục output
os.makedirs("output", exist_ok=True)

# === BƯỚC 1: TẠO DỮ LIỆU MÔ PHỎNG (GIỮ NGUYÊN) ===
np.random.seed(42)
h_src, w_src = 400, 500 # Kích thước vùng nguồn giả định

# 1. Inliers (Điểm chuẩn)
src_pts_good = np.random.rand(20, 2) * [w_src, h_src]
src_pts_good = src_pts_good.astype(np.float32)

H_true = np.array([
    [0.9, -0.1, 30],
    [0.05, 0.95, 20],
    [0.0001, -0.0002, 1.0]
], dtype=np.float64)

dst_pts_good = cv2.perspectiveTransform(src_pts_good.reshape(-1, 1, 2), H_true).reshape(-1, 2)
# Thêm nhiễu nhẹ
dst_pts_good += np.random.randn(20, 2).astype(np.float32) * 1.5

# 2. Outliers (Điểm nhiễu phá hoại) - Cố tình đặt xa để gây nhiễu mạnh
src_outliers = np.random.rand(8, 2).astype(np.float32) * [w_src, h_src]
dst_outliers = np.random.rand(8, 2).astype(np.float32) * [w_src+200, h_src+200] + [100, 100]

# Gộp dữ liệu
src_all = np.vstack([src_pts_good, src_outliers])
dst_all = np.vstack([dst_pts_good, dst_outliers])

# Số lượng inliers thực tế để tô màu
n_inliers = len(src_pts_good)

# === BƯỚC 2: ƯỚC LƯỢNG HOMOGRAPHY ===
# 1. Least Squares (LS) - Nhạy cảm với nhiễu
H_ls, _ = cv2.findHomography(src_all, dst_all, 0)

# 2. RANSAC - Bền bỉ với nhiễu
# Tăng ngưỡng lên một chút để demo rõ hơn
H_ransac, mask_ransac = cv2.findHomography(src_all, dst_all, cv2.RANSAC, ransacReprojThreshold=10.0)


# === BƯỚC 3: HIỂN THỊ TRỰC QUAN (PHẦN CẢI TIẾN MỚI) ===

def draw_warped_rect(ax, H, w, h, color, label, linestyle='-'):
    """Hàm hỗ trợ: Vẽ khung chữ nhật nguồn sau khi bị warp bởi H lên đồ thị đích."""
    if H is None: return
    # Định nghĩa 4 góc của khung hình nguồn
    corners_src = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    # Warp 4 góc này sang không gian đích bằng ma trận H
    corners_dst = cv2.perspectiveTransform(corners_src, H).reshape(-1, 2)
    
    # Vẽ đa giác nối 4 điểm đích
    polygon = Polygon(corners_dst, closed=True, fill=False, 
                      edgecolor=color, linewidth=2, linestyle=linestyle, label=label)
    ax.add_patch(polygon)


fig, axes = plt.subplots(1, 3, figsize=(20, 7))

titles = ["1. Ground Truth (Mục tiêu chuẩn)", 
          "2. Least Squares (Bị nhiễu kéo lệch)", 
          "3. RANSAC (Lọc nhiễu thành công)"]
Hs = [H_true, H_ls, H_ransac]
colors_H = ['blue', 'magenta', 'green']

for i, ax in enumerate(axes):
    # 1. Vẽ các điểm dữ liệu đích (Destination Points)
    # Tô màu xanh cho inliers thực, đỏ cho outliers thực
    ax.scatter(dst_all[:n_inliers, 0], dst_all[:n_inliers, 1], c='green', marker='o', s=60, label='True Inliers')
    ax.scatter(dst_all[n_inliers:, 0], dst_all[n_inliers:, 1], c='red', marker='x', s=150, linewidth=2, label='True Outliers')
    
    # 2. Vẽ khung hình bị warp bởi ma trận H tương ứng
    # Đây là phần cốt lõi giúp trực quan hóa tác động của thuật toán
    draw_warped_rect(ax, Hs[i], w_src, h_src, colors_H[i], f'Warped by H_{["true", "ls", "ransac"][i]}')

    # Trang trí đồ thị
    ax.set_title(titles[i], fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    # Đặt giới hạn trục để nhìn rõ các điểm outlier xa
    ax.set_xlim(-100, 800)
    ax.set_ylim(-100, 700)
    if i == 0: ax.set_ylabel("Destination Y")
    ax.set_xlabel("Destination X")

plt.suptitle("Trực quan hóa: Cách LS và RANSAC 'bẻ cong' không gian khi có nhiễu", fontsize=18, y=1.02)
plt.tight_layout()
plt.savefig("output/homography_visual_impact.png", dpi=150, bbox_inches='tight')
plt.show()