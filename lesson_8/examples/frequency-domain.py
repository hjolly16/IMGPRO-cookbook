import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def fourier_filter(img, filter_type="low_pass", radius=30):
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2  # Tâm của ảnh phổ

    # 1. Chuyển sang miền tần số
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)

    # 2. Tạo Mask (Đây chính là câu trả lời: thêm ảnh vào để làm bộ lọc!)
    mask = np.zeros((rows, cols), np.uint8)
    
    if filter_type == "low_pass":
        # Vẽ hình tròn trắng ở giữa để giữ lại tần số thấp
        cv2.circle(mask, (ccol, crow), radius, 1, -1)
    elif filter_type == "high_pass":
        # Vẽ hình tròn đen ở giữa, còn lại trắng để giữ lại tần số cao
        mask = np.ones((rows, cols), np.uint8)
        cv2.circle(mask, (ccol, crow), radius, 0, -1)

    # 3. Áp dụng Mask vào phổ
    fshift = dft_shift * mask

    # 4. Biến đổi ngược về miền không gian (Inverse DFT)
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    return img_back, mask

# === CHẠY THỬ NGHIỆM ===
img_path = "images/sample.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    img = np.zeros((300, 300), dtype=np.uint8)
    cv2.putText(img, "SHYN OS", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 5)

# Thực hiện lọc
radius_val = 40
low_pass_res, mask_lp = fourier_filter(img, "low_pass", radius_val)
high_pass_res, mask_hp = fourier_filter(img, "high_pass", radius_val)

# === HIỂN THỊ TRỰC QUAN ===
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Hàng 1: Xử lý tần số thấp (Làm mờ)
axes[0, 0].imshow(img, cmap="gray")
axes[0, 0].set_title("1. Ảnh gốc")
axes[0, 0].axis("off")

axes[0, 1].imshow(mask_lp, cmap="gray")
axes[0, 1].set_title(f"2. Mask Low-pass (Radius={radius_val})\n(Chỉ giữ lại các chấm ở giữa)")
axes[0, 1].axis("off")

axes[0, 2].imshow(low_pass_res, cmap="gray")
axes[0, 2].set_title("3. Kết quả: Ảnh bị MỜ\n(Vì mất tần số cao/cạnh sắc)")
axes[0, 2].axis("off")

# Hàng 2: Xử lý tần số cao (Trích xuất biên)
axes[1, 0].imshow(img, cmap="gray")
axes[1, 0].set_title("4. Ảnh gốc")
axes[1, 0].axis("off")

axes[1, 1].imshow(mask_hp, cmap="gray")
axes[1, 1].set_title("5. Mask High-pass\n(Đục lỗ ở giữa để lấy vùng biên phổ)")
axes[1, 1].axis("off")

axes[1, 2].imshow(high_pass_res, cmap="gray")
axes[1, 2].set_title("6. Kết quả: Chỉ còn BIÊN\n(Vì mất tần số thấp/vùng mịn)")
axes[1, 2].axis("off")

plt.suptitle("Sức mạnh của Fourier: Dùng 'Ảnh Mask' để điều khiển tần số", fontsize=16)
plt.tight_layout()
os.makedirs("output", exist_ok=True)
plt.savefig("output/fourier_filtering_intuitive.png", dpi=150)
plt.show()