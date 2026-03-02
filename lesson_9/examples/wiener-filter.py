import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os

def variance_of_laplacian(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def tao_motion_kernel(length, angle):
    """Tạo PSF (Point Spread Function) mô phỏng chuyển động thẳng."""
    kernel = np.zeros((length, length), dtype=np.float64)
    center = length // 2
    cos_a = np.cos(np.radians(angle))
    sin_a = np.sin(np.radians(angle))

    for i in range(length):
        offset = i - center
        x = int(center + offset * cos_a + 0.5)
        y = int(center + offset * sin_a + 0.5)
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1.0

    kernel /= kernel.sum()
    return kernel

def wiener_filter(img_blur, psf, K=0.01):
    """Khử mờ bằng thuật toán Wiener Deconvolution."""
    h, w = img_blur.shape
    # Padding PSF để khớp kích thước ảnh
    psf_padded = np.zeros((h, w), dtype=np.float64)
    kh, kw = psf.shape
    y0, x0 = h // 2 - kh // 2, w // 2 - kw // 2
    psf_padded[y0:y0 + kh, x0:x0 + kw] = psf

    # Chuyển sang miền tần số
    G = np.fft.fft2(img_blur.astype(np.float64))
    H = np.fft.fft2(psf_padded)

    # Áp dụng công thức Wiener
    H_conj = np.conj(H)
    H_sq = np.abs(H) ** 2
    F_hat = (H_conj / (H_sq + K)) * G

    # Biến đổi ngược và căn chỉnh lại tâm (fftshift)
    result = np.real(np.fft.ifft2(F_hat))
    result = np.fft.fftshift(result)
    return np.clip(result, 0, 255).astype(np.uint8)

# === BƯỚC 1: CHUẨN BỊ ẢNH ===
img = cv2.imread("images/sample.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    img = np.zeros((256, 256), dtype=np.uint8)
    cv2.putText(img, "SHYN OS", (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, 3)

# === BƯỚC 2: TẠO HIỆU ỨNG MỜ CHUYỂN ĐỘNG ===
length, angle = 21, 0 # Mờ ngang 21 pixel
psf = tao_motion_kernel(length, angle)
img_blur = cv2.filter2D(img, -1, psf)

# Thêm nhiễu Gaussian (nhiễu thực tế)
noise = np.random.normal(0, 2, img_blur.shape)
img_noisy = np.clip(img_blur.astype(np.float64) + noise, 0, 255).astype(np.uint8)

# === BƯỚC 3: PHỤC HỒI ===
K_values = [0.001, 0.01, 0.1]
results = [wiener_filter(img_noisy, psf, K) for K in K_values]

# === BƯỚC 4: HIỂN THỊ ===
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Hàng 1: Quy trình tạo mờ
axes[0, 0].imshow(img, cmap="gray"); axes[0, 0].set_title("1. Ảnh gốc"); axes[0, 0].axis("off")
axes[0, 1].imshow(psf, cmap="gray"); axes[0, 1].set_title(f"2. PSF (L={length}, θ={angle}°) ")
axes[0, 2].imshow(img_noisy, cmap="gray"); axes[0, 2].set_title(f"3. Motion Blur + Noise\nVoL={variance_of_laplacian(img_noisy):.1f}")
axes[0, 2].axis("off")

# So sánh với Unsharp Masking
blur_s = cv2.GaussianBlur(img_noisy, (0, 0), 3)
unsharp = cv2.addWeighted(img_noisy, 2.5, blur_s, -1.5, 0)
axes[0, 3].imshow(unsharp, cmap="gray"); axes[0, 3].set_title(f"4. Unsharp Masking\nVoL={variance_of_laplacian(unsharp):.1f}")
axes[0, 3].axis("off")

# Hàng 2: Kết quả Wiener
for i, K in enumerate(K_values):
    psnr = cv2.PSNR(img, results[i])
    axes[1, i].imshow(results[i], cmap="gray")
    axes[1, i].set_title(f"Wiener (K={K})\nPSNR={psnr:.1f}dB")
    axes[1, i].axis("off")

# Phổ Fourier của ảnh mờ
dft_blur = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(img_noisy.astype(np.float64)))))
axes[1, 3].imshow(dft_blur, cmap="gray"); axes[1, 3].set_title("Phổ ảnh mờ\n(Vạch tối vuông góc hướng mờ)")
axes[1, 3].axis("off")

plt.suptitle("Kỹ thuật Deconvolution: Khôi phục ảnh mờ chuyển động", fontsize=16)
plt.tight_layout()
os.makedirs("output", exist_ok=True)
plt.savefig("output/motion_deblur_analysis.png", dpi=150)
plt.show()