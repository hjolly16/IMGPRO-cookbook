import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# === BƯỚC 1: ĐẦU VÀO ===
img_path = "images/sample.jpg"
if not os.path.exists(img_path):
    print(f"Lỗi: Không tìm thấy file tại {img_path}")
    exit()

img = cv2.imread(img_path)
if img is None:
    raise ValueError("Không thể giải mã ảnh!")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# === BƯỚC 2: FAST KEYPOINTS ===
# Tạo detector FAST với ngưỡng threshold = 20
# nonmaxSuppression=True giúp loại bỏ các điểm lân cận bị lặp lại
fast = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
kp_fast = fast.detect(gray, None)

img_fast = img_rgb.copy()
# Vẽ các điểm đặc trưng tìm được (Keypoints)
for kp in kp_fast:
    x, y = int(kp.pt[0]), int(kp.pt[1])
    cv2.circle(img_fast, (x, y), 3, (255, 0, 0), -1)

# === BƯỚC 3: PHÂN TÍCH THAM SỐ THRESHOLD ===
thresholds = [5, 10, 20, 30, 50]
so_kp = []

for t in thresholds:
    detector = cv2.FastFeatureDetector_create(threshold=t, nonmaxSuppression=True)
    kp = detector.detect(gray, None)
    so_kp.append(len(kp))

# === BƯỚC 4: SO SÁNH TỐC ĐỘ (BENCHMARK) ===
# Harris (Cần chuyển sang float32)
gray_float = gray.astype(np.float32)
start = time.perf_counter()
for _ in range(50):
    cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)
t_harris = (time.perf_counter() - start) / 50 * 1000

# Shi-Tomasi
start = time.perf_counter()
for _ in range(50):
    cv2.goodFeaturesToTrack(gray, maxCorners=500, qualityLevel=0.01, minDistance=10)
t_st = (time.perf_counter() - start) / 50 * 1000

# FAST
start = time.perf_counter()
for _ in range(50):
    fast.detect(gray, None)
t_fast = (time.perf_counter() - start) / 50 * 1000

print(f"{'Phương pháp':<20}{'Thời gian (ms)':>15}{'Số Keypoints':>15}")
print("-" * 50)
print(f"{'Harris':<20}{t_harris:>13.2f}{'—':>15}")
print(f"{'Shi-Tomasi':<20}{t_st:>13.2f}{'—':>15}")
print(f"{'FAST (t=20)':<20}{t_fast:>13.2f}{len(kp_fast):>15}")
print(f"\n=> FAST nhanh hơn Harris khoảng: {t_harris/t_fast:.1f} lần")

# === BƯỚC 5: HIỂN THỊ ===
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 1. Ảnh kết quả FAST
axes[0].imshow(img_fast)
axes[0].set_title(f"FAST Detection ({len(kp_fast)} điểm)")
axes[0].axis("off")

# 2. Đồ thị độ nhạy Threshold
axes[1].plot(thresholds, so_kp, "bo-", linewidth=2, markersize=8)
axes[1].set_xlabel("Threshold (Ngưỡng)")
axes[1].set_ylabel("Số lượng Keypoints")
axes[1].set_title("Độ nhạy của Threshold")
axes[1].grid(True, alpha=0.3)

# 3. Biểu đồ so sánh tốc độ
ten_pp = ["Harris", "Shi-Tomasi", "FAST"]
thoi_gian = [t_harris, t_st, t_fast]
colors = ["#e74c3c", "#2ecc71", "#3498db"]
axes[2].bar(ten_pp, thoi_gian, color=colors)
axes[2].set_ylabel("Thời gian xử lý (ms)")
axes[2].set_title("So sánh hiệu năng")
for i, v in enumerate(thoi_gian):
    axes[2].text(i, v + 0.2, f"{v:.1f}ms", ha="center", fontweight='bold')

plt.tight_layout()
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f"{output_dir}/fast_benchmark.png", dpi=150)
plt.show()