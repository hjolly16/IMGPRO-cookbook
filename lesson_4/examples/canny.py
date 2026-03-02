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

# === BƯỚC 2: HÀM HỖ TRỢ AUTO-CANNY ===
def auto_canny(image, sigma=0.33):
    """
    Tự động tính toán ngưỡng Canny dựa trên giá trị trung vị (median) của ảnh.
    Giúp thuật toán hoạt động ổn định trên nhiều điều kiện ánh sáng khác nhau.
    """
    mu = np.median(image)
    # Tính toán ngưỡng thấp và ngưỡng cao dựa trên độ lệch sigma
    t_low = int(max(0, (1.0 - sigma) * mu))
    t_high = int(min(255, (1.0 + sigma) * mu))
    
    edged = cv2.Canny(image, t_low, t_high)
    return edged, t_low, t_high

# Tiền xử lý: Luôn làm mượt ảnh bằng Gaussian Blur trước khi dùng Canny
img_blur = cv2.GaussianBlur(img, (5, 5), 1.0)

# === BƯỚC 3: THÍ NGHIỆM VỚI NGƯỠNG (THRESHOLD) ===
nguong_list = [
    (30, 60, "Thấp (30, 60)"),
    (50, 150, "Trung bình (50, 150)"),
    (100, 200, "Cao (100, 200)"),
    (150, 300, "Rất cao (150, 300)"),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes[0, 0].imshow(img, cmap="gray")
axes[0, 0].set_title("1. Ảnh gốc")
axes[0, 0].axis("off")

for idx, (t_low, t_high, ten) in enumerate(nguong_list):
    edges = cv2.Canny(img_blur, t_low, t_high)
    # Tính mật độ điểm cạnh (%)
    mat_do = np.sum(edges > 0) / edges.size * 100
    row, col = (idx + 1) // 3, (idx + 1) % 3
    axes[row, col].imshow(edges, cmap="gray")
    axes[row, col].set_title(f"{ten}\nMật độ cạnh: {mat_do:.1f}%")
    axes[row, col].axis("off")

# Áp dụng Auto Canny
edges_auto, t_low_auto, t_high_auto = auto_canny(img_blur)
mat_do_auto = np.sum(edges_auto > 0) / edges_auto.size * 100
axes[1, 2].imshow(edges_auto, cmap="gray")
axes[1, 2].set_title(f"Tự động ({t_low_auto}, {t_high_auto})\nMật độ: {mat_do_auto:.1f}%")
axes[1, 2].axis("off")

plt.suptitle("Sự ảnh hưởng của bộ đôi ngưỡng (Hysteresis Thresholding)", fontsize=16)
plt.tight_layout()
os.makedirs("output", exist_ok=True)
plt.savefig("output/canny_thresholds.png", dpi=150)
plt.show()

# === BƯỚC 4: THÍ NGHIỆM VỚI ĐỘ MƯỢT (GAUSSIAN SIGMA) ===
fig2, axes2 = plt.subplots(1, 4, figsize=(18, 5))
axes2[0].imshow(img, cmap="gray")
axes2[0].set_title("Ảnh gốc")
axes2[0].axis("off")

# Khi sigma tăng, ảnh càng mờ, Canny sẽ chỉ giữ lại các cạnh biên lớn
for idx, s in enumerate([0.5, 1.5, 3.0]):
    img_g = cv2.GaussianBlur(img, (0, 0), s)
    # Giữ nguyên ngưỡng cố định để thấy sự khác biệt của Gaussian
    edges = cv2.Canny(img_g, 50, 150)
    mat_do = np.sum(edges > 0) / edges.size * 100
    axes2[idx + 1].imshow(edges, cmap="gray")
    axes2[idx + 1].set_title(f"Sigma={s}\nMật độ cạnh: {mat_do:.1f}%")
    axes2[idx + 1].axis("off")

plt.suptitle("Sự ảnh hưởng của Gaussian Blur lên kết quả Canny", fontsize=16)
plt.tight_layout()
plt.savefig("output/canny_blur_effect.png", dpi=150)
plt.show()