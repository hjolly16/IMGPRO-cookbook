import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def variance_of_laplacian(gray):
    """Tính toán độ biến thiên của Laplacian - chỉ số đo độ sắc nét phổ biến nhất."""
    # CV_64F để tránh tràn số khi tính đạo hàm bậc 2
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var()

def tenengrad(gray):
    """Thuật toán Tenengrad: Tính năng lượng của Gradient bằng toán tử Sobel."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    # Năng lượng trung bình của các cạnh
    return np.mean(gx**2 + gy**2)

def generate_blur_map(gray, block_size=32):
    """
    Tạo bản đồ vùng mờ (Blur Map).
    Giúp xác định vùng nào trong ảnh đang bị mất nét (out-of-focus).
    """
    h, w = gray.shape
    rows, cols = h // block_size, w // block_size
    bmap = np.zeros((rows, cols))

    for r in range(rows):
        for c in range(cols):
            y1, y2 = r * block_size, (r + 1) * block_size
            x1, x2 = c * block_size, (c + 1) * block_size
            block = gray[y1:y2, x1:x2]
            bmap[r, c] = variance_of_laplacian(block)
    
    # Chuẩn hóa để hiển thị đẹp hơn
    return cv2.normalize(bmap, None, 0, 1, cv2.NORM_MINMAX)

# === BƯỚC 1: CHUẨN BỊ DỮ LIỆU ===
img = cv2.imread("images/sample.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    # Tạo ảnh giả lập: Một khối vuông sắc nét trên nền nhiễu
    img = np.random.randint(80, 120, (300, 300), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (200, 200), 255, -1)
    cv2.circle(img, (150, 150), 30, 0, -1)

# === BƯỚC 2: MÔ PHỎNG CÁC MỨC ĐỘ MỜ ===
blur_kernels = [0, 5, 11, 21, 41] # Kích thước kernel Gaussian
data = []

for k in blur_kernels:
    # Nếu k=0 thì giữ nguyên ảnh gốc
    blurred = img.copy() if k == 0 else cv2.GaussianBlur(img, (k, k), 0)
    
    vol = variance_of_laplacian(blurred)
    ten = tenengrad(blurred)
    bmap = generate_blur_map(blurred, block_size=20)
    
    data.append({
        "img": blurred,
        "k": k,
        "vol": vol,
        "ten": ten,
        "bmap": bmap
    })

# === BƯỚC 3: HIỂN THỊ TRỰC QUAN ===
fig, axes = plt.subplots(2, len(blur_kernels), figsize=(20, 8))

for i, d in enumerate(data):
    # Hàng 1: Ảnh xám (Gốc/Mờ)
    axes[0, i].imshow(d["img"], cmap="gray")
    title = "Original" if d["k"] == 0 else f"Blur (k={d['k']})"
    axes[0, i].set_title(f"{title}\nVoL: {d['vol']:.1f}")
    axes[0, i].axis("off")

    # Hàng 2: Blur Map (Heatmap)
    # Màu càng 'nóng' (trắng/đỏ) nghĩa là vùng đó càng sắc nét
    im_bmap = axes[1, i].imshow(d["bmap"], cmap="hot", interpolation="nearest")
    axes[1, i].set_title("Edge Energy Map")
    axes[1, i].axis("off")

plt.suptitle("Phân tích độ sắc nét: Variance of Laplacian vs. Blur Map", fontsize=16)
plt.tight_layout()

os.makedirs("output", exist_ok=True)
plt.savefig("output/blur_detection_analysis.png", dpi=150)
plt.show()

# === BẢNG ĐÁNH GIÁ ===
print(f"{'Kernel':>8} {'VoL Score':>12} {'Tenengrad':>12} {'Xếp loại'}")
print("-" * 55)
for d in data:
    status = "Sắc nét" if d["vol"] > 500 else ("Chấp nhận" if d["vol"] > 100 else "Quá mờ!")
    label = "Gốc" if d["k"] == 0 else f"k={d['k']}"
    print(f"{label:>8} {d['vol']:>12.1f} {d['ten']:>12.1f} {status:>12}")