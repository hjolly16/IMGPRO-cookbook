import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def swap_phase_magnitude(img1, img2):
    # DFT cho cả 2 ảnh
    dft1 = np.fft.fft2(img1)
    dft2 = np.fft.fft2(img2)
    
    # Tách biên độ (mag) và pha (phase)
    mag1, phase1 = np.abs(dft1), np.angle(dft1)
    mag2, phase2 = np.abs(dft2), np.angle(dft2)
    
    # Hoán đổi: Mag1 + Phase2 và Mag2 + Phase1
    res1 = np.real(np.fft.ifft2(mag1 * np.exp(1j * phase2)))
    res2 = np.real(np.fft.ifft2(mag2 * np.exp(1j * phase1)))
    
    # [QUAN TRỌNG]: Chuẩn hóa về dải 0-255 để nhìn rõ
    res1 = cv2.normalize(res1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    res2 = cv2.normalize(res2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return res1, res2

# === TẠO ẢNH MẪU CỰC KỲ KHÁC BIỆT ===
size = 512
# Ảnh A: Một chữ "A" khổng lồ (Cấu trúc cứng)
img_a = np.zeros((size, size), dtype=np.float32)
cv2.putText(img_a, "A", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 15, 255, 20)

# Ảnh B: Một vòng tròn đồng tâm (Cấu trúc mềm)
img_b = np.zeros((size, size), dtype=np.float32)
for r in range(50, 250, 40):
    cv2.circle(img_b, (256, 256), r, 255, 5)

# Thực hiện hoán đổi
res_a_mag_b_phase, res_b_mag_a_phase = swap_phase_magnitude(img_a, img_b)

# === HIỂN THỊ ===
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(img_a, cmap="gray")
axes[0].set_title("Gốc A: Chữ 'A'")
axes[0].axis("off")

axes[1].imshow(img_b, cmap="gray")
axes[1].set_title("Gốc B: Vòng tròn")
axes[1].axis("off")

# Kết quả 1: Lấy năng lượng của A nhưng dán vào cấu trúc của B
axes[2].imshow(res_a_mag_b_phase, cmap="gray")
axes[2].set_title("Biên độ A + PHA B\n=> Hiện ra Vòng tròn!")
axes[2].axis("off")

# Kết quả 2: Lấy năng lượng của B nhưng dán vào cấu trúc của A
axes[3].imshow(res_b_mag_a_phase, cmap="gray")
axes[3].set_title("Biên độ B + PHA A\n=> Hiện ra chữ 'A'!")
axes[3].axis("off")

plt.suptitle("Minh chứng: Pha nắm giữ toàn bộ cấu trúc hình học", fontsize=16)
plt.tight_layout()
os.makedirs("output", exist_ok=True)
plt.savefig("output/phase_swap_intuitive.png", dpi=150)
plt.show()