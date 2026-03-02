import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Create output folder if it doesn't exist
os.makedirs("output", exist_ok=True)

def tich_chap_thu_cong(img, kernel):
    """Tích chập 2D thủ công (correlation, không lật kernel)."""
    kH, kW = kernel.shape
    pad_h, pad_w = kH // 2, kW // 2

    # Padding phản chiếu
    img_pad = cv2.copyMakeBorder(
        img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REFLECT_101
    )

    H, W = img.shape
    output = np.zeros_like(img, dtype=np.float64)

    for i in range(H):
        for j in range(W):
            vung = img_pad[i:i + kH, j:j + kW].astype(np.float64)
            output[i, j] = np.sum(vung * kernel)

    return output

# Đọc ảnh xám
img = cv2.imread("images/sample.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Không tìm thấy ảnh!")

# Chỉ dùng ảnh nhỏ để kiểm tra (tích chập thủ công rất chậm)
img_small = cv2.resize(img, (100, 100))

# === KERNEL TRUNG BÌNH 3×3 ===
kernel_mean = np.ones((3, 3), dtype=np.float64) / 9.0

# Tích chập thủ công
ket_qua_tc = tich_chap_thu_cong(img_small, kernel_mean)
ket_qua_tc = np.clip(ket_qua_tc, 0, 255).astype(np.uint8)

# OpenCV filter2D
ket_qua_cv = cv2.filter2D(img_small, -1, kernel_mean)

# So sánh
sai_khac = np.abs(ket_qua_tc.astype(np.int16) - ket_qua_cv.astype(np.int16))
print(f"Sai khác tối đa: {sai_khac.max()}")
print(f"Sai khác trung bình: {sai_khac.mean():.4f}")
# print(f"Giống nhau: {np.allclose(ket_qua_tc, ket_qua_cv, atol=1)}") # Removed error line

# --- Hiển thị trực quan so sánh thủ công và OpenCV ---
plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.imshow(img_small, cmap='gray')
plt.title("Ảnh gốc (100x100)")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(ket_qua_tc, cmap='gray')
plt.title("Tích chập thủ công")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(ket_qua_cv, cmap='gray')
plt.title("OpenCV (filter2D)")
plt.axis('off')

plt.tight_layout()
plt.savefig("output/manual_convolution_compare.png")
print("Đã lưu ảnh so sánh vào output/manual_convolution_compare.png")
plt.show()

# === THỬ CÁC KERNEL KHÁC ===
# Kernel Laplacian
kernel_lap = np.array([[0,  1, 0],
                       [1, -4, 1],
                       [0,  1, 0]], dtype=np.float64)

# Kernel làm nét
kernel_sharp = np.array([[ 0, -1,  0],
                         [-1,  5, -1],
                         [ 0, -1,  0]], dtype=np.float64)

ket_qua_dict = {}
for ten, k in [("Trung bình", kernel_mean), ("Laplacian", kernel_lap),
               ("Làm nét", kernel_sharp)]:
    # Để thấy rõ biên Laplacian có âm, dùng CV_64F, sau đó convert lại
    ket_qua = cv2.filter2D(img, cv2.CV_64F, k)
    if ten == "Laplacian":
        # Chuyển về mức 0-255 và hiển thị giá trị âm dưới dạng xám
        ket_qua = cv2.convertScaleAbs(ket_qua) 
    else:
        # Giới hạn 0-255 với các filter khác
        ket_qua = np.clip(ket_qua, 0, 255).astype(np.uint8)

    ket_qua_dict[ten] = ket_qua
    print(f"\n{ten}: shape={ket_qua.shape}, "
          f"min={ket_qua.min()}, max={ket_qua.max()}")

# --- Hiển thị trực quan hiệu ứng các kernel ---
plt.figure(figsize=(12, 4))
plt.subplot(1, 4, 1)
plt.imshow(img, cmap='gray')
plt.title("Ảnh gốc")
plt.axis('off')

for i, (ten, anh_kq) in enumerate(ket_qua_dict.items(), start=2):
    plt.subplot(1, 4, i)
    plt.imshow(anh_kq, cmap='gray')
    plt.title(ten)
    plt.axis('off')

plt.tight_layout()
plt.savefig("output/manual_convolution_kernels.png")
print("Đã lưu ảnh các kernel vào output/manual_convolution_kernels.png")
plt.show()
