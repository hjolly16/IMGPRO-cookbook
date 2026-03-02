import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Khởi tạo thư mục và đọc ảnh trước ---
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

img = cv2.imread("images/sample.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Lỗi: Không tìm thấy ảnh tại images/sample.jpg")
    exit()

# === PHẦN 1: XÂY DỰNG VÀ VẼ CÁC HÀM ÁNH XẠ (TRANSFORMATION FUNCTIONS) ===
# Tạo dải giá trị đầu vào r từ 0 đến 255
r = np.arange(256, dtype=np.float64)

# 1. Hàm đồng nhất (Identity): s = r
T_identity = r.copy()

# 2. Hiệu chỉnh Gamma: s = c * r^gamma
# Cần chuẩn hóa r về [0, 1] trước khi lấy mũ, biến đổi xong nhân lại 255
# Gamma < 1: Làm sáng vùng tối | Gamma > 1: Làm đậm vùng tối
T_gamma_05 = 255.0 * (r / 255.0)**0.5
T_gamma_20 = 255.0 * (r / 255.0)**2.0

# 3. Biến đổi Logarit: s = c * log(1 + r)
# Tính chất: Nâng rất nhanh ở giá trị r nhỏ, và cong cung nhẹ ở r lớn.
# Ứng dụng: Giúp nhìn thấy chi tiết ở vùng tối nhưng làm mờ bớt chi tiết ở vùng sáng. Nén dải động.
c_log = 255.0 / np.log(1 + 255)
T_log = c_log * np.log(1 + r)

# 4. Nghịch đảo (Negative): s = 255 - r
T_neg = 255 - r

# --- VẼ HÌNH 1: ĐỒ THỊ CÁC HÀM ÁNH XẠ ---
plt.figure(figsize=(10, 8))
plt.plot(r, T_identity, "k--", linewidth=2, label="Đồng nhất (s = r)")
plt.plot(r, T_gamma_05, "r-",  linewidth=2, label="Gamma = 0.5 (S = c * r^0.5)")
plt.plot(r, T_gamma_20, "b-",  linewidth=2, label="Gamma = 2.0 (S = c * r^2.0)")
plt.plot(r, T_log, "g-",       linewidth=2, label="Logarit (S = c * log(1+r))")
plt.plot(r, T_neg, "m-",       linewidth=2, label="Nghịch đảo (S = 255 - r)")

plt.xlabel("Giá trị pixel ban đầu (Input r)")
plt.ylabel("Giá trị pixel đầu ra (Output s)")
plt.title("Biểu đồ các hàm biến đổi cường độ sáng (Mapping Functions)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim([0, 255])
plt.ylim([0, 255])

plt.savefig(f"{output_dir}/1_ham_anh_xa.png", dpi=150)
plt.show()

# === PHẦN 2: TRỰC QUAN HÓA SỰ THAY ĐỔI TRÊN ẢNH THỰC TẾ ===
# Đây là cách hiệu quả nhất để người học hiểu các hàm trên sẽ làm gì với ảnh
# cv2.LUT (Look Up Table) cho phép áp dụng hàm ánh xạ r -> s vào toàn bộ ảnh cực kì nhanh

img_gamma_05 = cv2.LUT(img, T_gamma_05.astype(np.uint8))
img_gamma_20 = cv2.LUT(img, T_gamma_20.astype(np.uint8))
img_log      = cv2.LUT(img, T_log.astype(np.uint8))
img_neg      = cv2.LUT(img, T_neg.astype(np.uint8))

fig_img, axes_img = plt.subplots(2, 3, figsize=(16, 10))
axes_img = axes_img.ravel()

def plot_img_with_info(ax, image, title):
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis("off")

plot_img_with_info(axes_img[0], img, "1. Ảnh gốc (Đồng nhất)")
plot_img_with_info(axes_img[1], img_gamma_05, "2. Gamma 0.5\n(Đường đỏ đi lên -> Sáng ảnh)")
plot_img_with_info(axes_img[2], img_gamma_20, "3. Gamma 2.0\n(Đường xanh đi xuống -> Tối ảnh)")
plot_img_with_info(axes_img[3], img_log, "4. Logarit\n(Kéo sáng cực mạnh vùng tối)")
plot_img_with_info(axes_img[4], img_neg, "5. Nghịch đảo\n(Đảo ngược sáng tối âm bản)")

# Ẩn đồ thị số 6
axes_img[5].axis("off")

plt.tight_layout()
plt.savefig(f"{output_dir}/2_anh_xa_truc_quan_tren_anh.png", dpi=150)
plt.show()

# === PHẦN 3: ỨNG DỤNG LOGARIT ĐIỂN HÌNH - PHÂN TÍCH PHỔ FOURIER ===
# Phổ Fourier có một đặc điểm: Điểm trung tâm (f=0) có cường độ VÔ CÙNG LỚN (hàng triệu),
# trong khi các chi tiết tần số ở rìa xung quanh có cường độ CỰC NHỎ.
# Do đó, chênh lệch sáng/tối cực kỳ lớn (dải động lớn).
# Nếu không xài Log sẽ không thấy được chi tiết viền xung quanh.

# 1. Chuyển sang miền tần số bằng Fast Fourier Transform (FFT)
f_transform = np.fft.fft2(img.astype(np.float64))
# Dịch chuyển tần số thấp vào giữa ảnh để dễ quan sát
f_shift = np.fft.fftshift(f_transform)
# Tính độ lớn (Magnitude Spectrum)
magnitude = np.abs(f_shift)

# 2. Áp dụng biến đổi Logarit để nén dải động
magnitude_log = np.log(1 + magnitude)
# Chuẩn hóa về [0, 255] để hiển thị
magnitude_log_norm = (magnitude_log / magnitude_log.max() * 255).astype(np.uint8)

# VẼ HÌNH 3: SO SÁNH PHỔ FOURIER
fig_fourier, axes_f = plt.subplots(1, 3, figsize=(18, 6))

axes_f[0].imshow(img, cmap="gray")
axes_f[0].set_title("1. Ảnh gốc (Miền không gian)")
axes_f[0].axis("off")

# Hiển thị phổ không dùng Log
axes_f[1].imshow(magnitude, cmap="gray")
axes_f[1].set_title("2. Phổ Fourier (Chưa áp dụng Log)\nĐiểm tâm sáng chói, lấn át các chi tiết tần số khác.")
axes_f[1].axis("off")

# Hiển thị phổ có dùng Log
axes_f[2].imshow(magnitude_log_norm, cmap="gray")
axes_f[2].set_title("3. Phổ Fourier (Sau khi áp dụng Log)\nLog đã nén điểm tâm, kéo bật sáng các chi tiết nhỏ.")
axes_f[2].axis("off")

plt.tight_layout()
plt.savefig(f"{output_dir}/3_pho_fourier_log_nang_cao.png", dpi=150)
plt.show()

print("✓ Đã hoàn thành minh họa hàm ánh xạ trực quan và ứng dụng nén dải động trên phổ Fourier!")