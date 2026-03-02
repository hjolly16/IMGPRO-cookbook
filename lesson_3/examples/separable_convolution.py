import cv2
import numpy as np
import time
import os

# === BƯỚC 1: CHUẨN BỊ DỮ LIỆU LỚN ===
# Tạo ảnh ngẫu nhiên 6 Megapixels để thấy rõ sự khác biệt về hiệu năng
img = np.random.randint(0, 256, (2000, 3000), dtype=np.uint8)

# Các kích thước kernel để thử nghiệm (càng lớn sự khác biệt càng rõ)
kich_thuoc = [3, 7, 11, 21, 31, 51]

print(f"{'Kernel':>8} {'filter2D (ms)':>15} {'sepFilter2D (ms)':>18} {'Tăng tốc':>12}")
print("-" * 55)

# === BƯỚC 2: CHẠY BENCHMARK ===
for k in kich_thuoc:
    # 1. Tạo Kernel Gaussian 1D
    # Một bộ lọc Gaussian 2D có thể được tạo ra bằng cách nhân 2 bộ lọc 1D
    kernel_1d = cv2.getGaussianKernel(k, -1)
    
    # 2. Tạo Kernel Gaussian 2D (Ma trận k x k)
    kernel_2d = kernel_1d @ kernel_1d.T

    # --- Đo thời gian filter2D (Tích chập 2D thông thường) ---
    start = time.perf_counter()
    for _ in range(20):
        cv2.filter2D(img, -1, kernel_2d)
    t_2d = (time.perf_counter() - start) / 20 * 1000

    # --- Đo thời gian sepFilter2D (Tích chập tách được) ---
    # Thay vì tính k*k, nó tính theo chiều ngang rồi chiều dọc (k + k)
    start = time.perf_counter()
    for _ in range(20):
        cv2.sepFilter2D(img, -1, kernel_1d, kernel_1d)
    t_sep = (time.perf_counter() - start) / 20 * 1000

    tang_toc = t_2d / t_sep
    print(f"{k:>3} x {k:<3} {t_2d:>13.2f} {t_sep:>16.2f} {tang_toc:>11.1f}x")

print("\n✓ Nhận xét: Kernel càng lớn, sepFilter2D càng thể hiện sức mạnh vượt trội!")