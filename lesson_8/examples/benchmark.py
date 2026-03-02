import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import os

def loc_khong_gian(img, kernel_size):
    """Lọc Gaussian truyền thống trong miền không gian."""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def loc_tan_so_fft(img, kernel_size):
    """Lọc Gaussian tương đương trong miền tần số."""
    rows, cols = img.shape
    opt_r = cv2.getOptimalDFTSize(rows)
    opt_c = cv2.getOptimalDFTSize(cols)

    padded = np.zeros((opt_r, opt_c), dtype=np.float64)
    padded[:rows, :cols] = img

    # Chuyển sang miền tần số
    dft = np.fft.fft2(padded)
    dft_shift = np.fft.fftshift(dft)

    # Tạo Gaussian LPF tương đương dựa trên kernel_size
    sigma_spatial = (kernel_size - 1) / 4  # Công thức xấp xỉ thực nghiệm
    # Tần số cắt D0 tỷ lệ nghịch với độ mờ trong miền không gian
    d0 = opt_r / (2 * np.pi * sigma_spatial) if sigma_spatial > 0 else opt_r
    
    crow, ccol = opt_r // 2, opt_c // 2
    u = np.arange(opt_r).reshape(-1, 1) - crow
    v = np.arange(opt_c).reshape(1, -1) - ccol
    D = np.sqrt(u**2 + v**2)
    H = np.exp(-(D**2) / (2 * (d0**2)))

    # Nhân ma trận (Lọc) và biến đổi ngược
    filtered = dft_shift * H
    result = np.real(np.fft.ifft2(np.fft.ifftshift(filtered)))
    
    return np.clip(result[:rows, :cols], 0, 255).astype(np.uint8)

# === THỰC HIỆN BENCHMARK ===
img = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
kernel_sizes = [3, 7, 15, 31, 51, 71, 101, 151, 201]
repeats = 15

t_spatial, t_fft = [], []

print("Đang chạy Benchmark... Vui lòng đợi.")
for ks in kernel_sizes:
    # 1. Đo miền không gian
    start = time.perf_counter()
    for _ in range(repeats): loc_khong_gian(img, ks)
    t_spatial.append((time.perf_counter() - start) / repeats * 1000)

    # 2. Đo miền tần số (FFT)
    start = time.perf_counter()
    for _ in range(repeats): loc_tan_so_fft(img, ks)
    t_fft.append((time.perf_counter() - start) / repeats * 1000)

# === TRỰC QUAN HÓA ===
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Đồ thị thời gian tuyệt đối
axes[0].plot(kernel_sizes, t_spatial, "bo-", linewidth=2, label="Spatial (OpenCV)")
axes[0].plot(kernel_sizes, t_fft, "rs-", linewidth=2, label="Frequency (FFT)")
axes[0].set_xlabel("Kích thước Kernel (K)")
axes[0].set_ylabel("Thời gian xử lý (ms)")
axes[0].set_title("So sánh tốc độ: Spatial vs. FFT\n(Ảnh 512x512)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Đồ thị tỉ lệ (Speedup)
ratio = [ts / tf for ts, tf in zip(t_spatial, t_fft)]
colors = ["#27ae60" if r > 1 else "#e74c3c" for r in ratio]
axes[1].bar([str(k) for k in kernel_sizes], ratio, color=colors)
axes[1].axhline(y=1, color="black", linestyle="--")
axes[1].set_title("Tỉ lệ hiệu năng (Spatial / FFT)\n> 1: FFT nhanh hơn | < 1: Spatial nhanh hơn")
axes[1].set_ylabel("Lần (Times)")

plt.tight_layout()
os.makedirs("output", exist_ok=True)
plt.savefig("output/fft_benchmark_final.png", dpi=150)
plt.show()

# In bảng kết quả
print(f"\n{'Kernel':>8} {'Spatial (ms)':>15} {'FFT (ms)':>12} {'Winner':>10}")
print("-" * 50)
for ks, ts, tf in zip(kernel_sizes, t_spatial, t_fft):
    winner = "FFT" if tf < ts else "Spatial"
    gain = ts / tf if tf < ts else tf / ts
    print(f"{ks:>8} {ts:>15.2f} {tf:>12.2f} {winner:>10} ({gain:.1f}x)")