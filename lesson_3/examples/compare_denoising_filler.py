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

# === BƯỚC 2: TẠO ẢNH CÓ NHIỄU (MÔ PHỎNG THỰC TẾ) ===
# 1. Nhiễu Gaussian (Thường gặp trong điều kiện thiếu sáng)
nhieu_gauss = np.random.normal(0, 25, img.shape).astype(np.float64)
img_gauss = np.clip(img.astype(np.float64) + nhieu_gauss, 0, 255).astype(np.uint8)

# 2. Nhiễu Muối Tiêu - Salt & Pepper (Do lỗi truyền dẫn tín hiệu)
img_sp = img.copy()
so_pixel_nhieu = int(img.size * 0.05) # 5% số lượng pixel bị nhiễu
# Muối (Trắng - 255)
coords_muoi = [np.random.randint(0, i - 1, so_pixel_nhieu) for i in img.shape]
img_sp[tuple(coords_muoi)] = 255
# Tiêu (Đen - 0)
coords_tieu = [np.random.randint(0, i - 1, so_pixel_nhieu) for i in img.shape]
img_sp[tuple(coords_tieu)] = 0

# === BƯỚC 3: ÁP DỤNG CÁC BỘ LỌC KHỬ NHIỄU ===
def ap_dung_bo_loc(image):
    return {
        "Mean": cv2.blur(image, (5, 5)),
        "Gaussian": cv2.GaussianBlur(image, (5, 5), 1.0),
        "Median": cv2.medianBlur(image, 5),
        "Bilateral": cv2.bilateralFilter(image, 9, 75, 75)
    }

kq_gauss = ap_dung_bo_loc(img_gauss)
kq_sp = ap_dung_bo_loc(img_sp)

# === BƯỚC 4: ĐÁNH GIÁ ĐỊNH LƯỢNG (PSNR) ===
def tinh_psnr(goc, ket_qua):
    mse = np.mean((goc.astype(np.float64) - ket_qua.astype(np.float64)) ** 2)
    if mse == 0: return float("inf")
    return 10 * np.log10(255.0 ** 2 / mse)

# === BƯỚC 5: HIỂN THỊ TRỰC QUAN ===
fig, axes = plt.subplots(2, 5, figsize=(22, 10))
cols = ["Ảnh Nhiễu", "Mean (Trung bình)", "Gaussian", "Median (Trung vị)", "Bilateral (Song phương)"]

# Hiển thị hàng nhiễu Gaussian
du_lieu_gauss = [img_gauss] + list(kq_gauss.values())
for idx, (anh, ten) in enumerate(zip(du_lieu_gauss, cols)):
    axes[0, idx].imshow(anh, cmap="gray")
    axes[0, idx].set_title(f"{ten}\nPSNR: {tinh_psnr(img, anh):.2f} dB")
    axes[0, idx].axis("off")

# Hiển thị hàng nhiễu Muối Tiêu
du_lieu_sp = [img_sp] + list(kq_sp.values())
for idx, (anh, ten) in enumerate(zip(du_lieu_sp, cols)):
    axes[1, idx].imshow(anh, cmap="gray")
    axes[1, idx].set_title(f"{ten}\nPSNR: {tinh_psnr(img, anh):.2f} dB")
    axes[1, idx].axis("off")

axes[0, 0].set_ylabel("Nhiễu Gaussian", fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel("Nhiễu Muối Tiêu", fontsize=14, fontweight='bold')

plt.suptitle("SO SÁNH CÁC BỘ LỌC KHỬ NHIỄU TRÊN TỪNG LOẠI NHIỄU", fontsize=16)
plt.tight_layout()
os.makedirs("output", exist_ok=True)
plt.savefig("output/so_sanh_denoising.png", dpi=150)
plt.show()