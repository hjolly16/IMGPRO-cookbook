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

# === BƯỚC 2: ĐỊNH NGHĨA CÁC PHƯƠNG PHÁP LÀM NÉT ===

def unsharp_mask(image, sigma, alpha):
    """
    Làm nét bằng kỹ thuật Unsharp Masking.
    Cơ chế: Ảnh gốc + Alpha * (Ảnh gốc - Ảnh làm mờ)
    """
    # 1. Tạo bản làm mờ (Low-pass filter)
    blur = cv2.GaussianBlur(image, (0, 0), sigma)
    
    # 2. Trích xuất chi tiết (High-pass filter: Gốc - Mờ)
    detail = image.astype(np.float64) - blur.astype(np.float64)
    
    # 3. Cộng chi tiết ngược lại vào ảnh gốc
    sharp = image.astype(np.float64) + alpha * detail
    
    return np.clip(sharp, 0, 255).astype(np.uint8)

def laplacian_sharpen(image, alpha=1.0):
    """
    Làm nét bằng toán tử Laplacian (Đạo hàm bậc hai).
    Cơ chế: Làm nổi bật các vùng có cường độ thay đổi đột ngột (cạnh).
    """
    # Lọc Gaussian nhẹ để giảm nhiễu trước khi tính đạo hàm (tránh làm nét nhiễu)
    blur = cv2.GaussianBlur(image, (3, 3), 0.5)
    
    # Tính Laplacian
    laplacian = cv2.Laplacian(blur, cv2.CV_64F)
    
    # Trừ Laplacian khỏi ảnh gốc (do dấu của kernel Laplacian trong OpenCV)
    sharp = image.astype(np.float64) - alpha * laplacian
    
    return np.clip(sharp, 0, 255).astype(np.uint8)

# === BƯỚC 3: ÁP DỤNG THỬ NGHIỆM ===
# USM với các tham số khác nhau
img_usm_1 = unsharp_mask(img, sigma=1.0, alpha=1.0) # Nhẹ nhàng
img_usm_2 = unsharp_mask(img, sigma=1.0, alpha=2.0) # Mạnh hơn (Alpha cao)
img_usm_3 = unsharp_mask(img, sigma=3.0, alpha=1.0) # Vùng làm nét rộng (Sigma cao)

# Laplacian
img_lap = laplacian_sharpen(img, alpha=1.0)

# Bản đồ chi tiết để minh họa tutorial (Detail Map)
blur_tmp = cv2.GaussianBlur(img, (0, 0), 1.0)
detail_map = img.astype(np.float64) - blur_tmp.astype(np.float64)
# Chuẩn hóa để hiển thị: dịch mức xám 0 về 128
detail_vis = np.clip(detail_map * 3 + 128, 0, 255).astype(np.uint8)

# === BƯỚC 4: HIỂN THỊ ===
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

axes[0, 0].imshow(img, cmap="gray")
axes[0, 0].set_title("1. Ảnh gốc")
axes[0, 0].axis("off")

axes[0, 1].imshow(img_usm_1, cmap="gray")
axes[0, 1].set_title("2. USM (σ=1.0, α=1.0)")
axes[0, 1].axis("off")

axes[0, 2].imshow(img_usm_2, cmap="gray")
axes[0, 2].set_title("3. USM (σ=1.0, α=2.0)\n(Tăng độ rực chi tiết)")
axes[0, 2].axis("off")

axes[1, 0].imshow(img_usm_3, cmap="gray")
axes[1, 0].set_title("4. USM (σ=3.0, α=1.0)\n(Vùng nét dày hơn)")
axes[1, 0].axis("off")

axes[1, 1].imshow(img_lap, cmap="gray")
axes[1, 1].set_title("5. Laplacian Sharpen")
axes[1, 1].axis("off")

axes[1, 2].imshow(detail_vis, cmap="gray")
axes[1, 2].set_title("6. Bản đồ chi tiết (Detail Map)\nThành phần được cộng thêm")
axes[1, 2].axis("off")

plt.suptitle("So sánh các kỹ thuật Làm nét ảnh", fontsize=16)
plt.tight_layout()
os.makedirs("output", exist_ok=True)
plt.savefig("output/so_sanh_lam_net.png", dpi=150)
plt.show()