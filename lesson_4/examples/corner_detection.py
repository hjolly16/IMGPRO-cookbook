import cv2
import numpy as np
import matplotlib.pyplot as plt
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
# Harris yêu cầu ảnh đầu vào kiểu float32
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

# === BƯỚC 2: HARRIS CORNER DETECTION ===
# blockSize: Kích thước vùng lân cận xét phát hiện góc
# ksize: Tham số khẩu độ cho toán tử Sobel dùng bên trong
# k: Tham số tự do trong công thức tính R (thường 0.04 - 0.06)
harris_response = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

# Giãn nở với kernel lớn (to hơn mặc định) để các góc hiện rõ thành các khối to trên ảnh
kernel = np.ones((11, 11), np.uint8)
harris_dilated = cv2.dilate(harris_response, kernel)

# Đánh dấu các góc tìm được lên ảnh gốc
img_harris = img_rgb.copy()
# Ngưỡng: Chỉ lấy các điểm có giá trị phản hồi R lớn (thường lấy % của Max)
nguong_harris = 0.01 * harris_response.max()
# Tô màu các vùng góc thành khối vuông màu đỏ tươi
img_harris[harris_dilated > nguong_harris] = [255, 0, 0] # Màu đỏ

# Việc đếm pixel góc tính trên bản đồ chưa giãn nở để số lượng bớt bị khuyếch đại quá mức
so_goc_harris = np.sum(harris_response > nguong_harris)

# === BƯỚC 3: SHI-TOMASI (Good Features to Track) ===
# Cải tiến của Harris, chọn các góc phân bổ đều và chất lượng hơn
corners_st = cv2.goodFeaturesToTrack(
    gray.astype(np.uint8), 
    maxCorners=200,      # Số lượng góc tối đa muốn tìm
    qualityLevel=0.01,   # Mức chất lượng tối thiểu (0-1)
    minDistance=10,      # Khoảng cách tối thiểu giữa các góc
    blockSize=3
)

img_st = img_rgb.copy()
if corners_st is not None:
    for corner in corners_st:
        x, y = corner.ravel().astype(int)
        # Vẽ điểm tròn to hơn và thêm viền đen để góc nổi bật trên mọi nền ảnh
        cv2.circle(img_st, (x, y), 8, (0, 255, 0), -1) # Màu xanh lá (lõi)
        cv2.circle(img_st, (x, y), 9, (0, 0, 0), 2)    # Viền đen bên ngoài
    so_goc_st = len(corners_st)
else:
    so_goc_st = 0

# === BƯỚC 4: HIỂN THỊ SO SÁNH ===
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(img_rgb)
axes[0].set_title("1. Ảnh gốc")
axes[0].axis("off")

axes[1].imshow(img_harris)
axes[1].set_title(f"2. Harris Corner ({so_goc_harris} điểm)\nĐiểm đỏ = góc")
axes[1].axis("off")

axes[2].imshow(img_st)
axes[2].set_title(f"3. Shi-Tomasi ({so_goc_st} điểm)\nĐiểm xanh = góc")
axes[2].axis("off")

plt.suptitle("So sánh phương pháp phát hiện góc", fontsize=16)
plt.tight_layout()
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f"{output_dir}/corner_detection_compare.png", dpi=150)
plt.show()

# === BƯỚC 5: PHÂN TÍCH BẢN ĐỒ PHẢN HỒI R ===
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

# Trực quan hóa giá trị R (Response) - nơi có giá trị cao chính là góc
im = axes2[0].imshow(harris_response, cmap="jet")
axes2[0].set_title("Bản đồ phản hồi Harris (R)\nĐỏ = R cao (Góc)")
axes2[0].axis("off")
plt.colorbar(im, ax=axes2[0], fraction=0.046, pad=0.04)

# Biểu đồ phân phối giá trị R
axes2[1].hist(harris_response.ravel(), bins=100, log=True, color="steelblue")
axes2[1].axvline(x=nguong_harris, color="red", linestyle="--", label=f"Ngưỡng chọn góc")
axes2[1].set_title("Phân phối giá trị R (Log scale)")
axes2[1].set_xlabel("Giá trị R")
axes2[1].set_ylabel("Số lượng pixel")
axes2[1].legend()

plt.tight_layout()
plt.savefig(f"{output_dir}/harris_math_analysis.png", dpi=150)
plt.show()