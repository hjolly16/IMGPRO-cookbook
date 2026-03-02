import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def khop_luoc_do(img_nguon, img_tham_chieu):
    """
    Biến đổi img_nguon sao cho phân phối mức xám của nó 
    khớp với img_tham_chieu.
    """
    # 1. Tính CDF của ảnh nguồn
    hist_s = cv2.calcHist([img_nguon], [0], None, [256], [0, 256]).ravel()
    cdf_s = hist_s.cumsum()
    cdf_s = cdf_s / cdf_s[-1]  # Chuẩn hóa về [0, 1]

    # 2. Tính CDF của ảnh tham chiếu
    hist_r = cv2.calcHist([img_tham_chieu], [0], None, [256], [0, 256]).ravel()
    cdf_r = hist_r.cumsum()
    cdf_r = cdf_r / cdf_r[-1]  # Chuẩn hóa về [0, 1]

    # 3. Xây dựng bảng ánh xạ (LUT) để khớp hai CDF
    # Tìm r_k sao cho CDF_tham_chieu(r_k) gần với CDF_nguon(s_k) nhất
    lut = np.zeros(256, dtype=np.uint8)
    for s_k in range(256):
        # Tìm giá trị r_k có CDF gần nhất với CDF nguồn tại mức xám s_k
        khoang_cach = np.abs(cdf_r - cdf_s[s_k])
        lut[s_k] = np.argmin(khoang_cach)

    # 4. Áp dụng bảng tra để chuyển đổi ảnh
    return cv2.LUT(img_nguon, lut)

# === CHƯƠNG TRÌNH CHÍNH ===
img_path = "images/sample.jpg"
if not os.path.exists(img_path):
    print("Lỗi: Cần có ảnh 'images/sample.jpg' để làm mẫu.")
    exit()

# Đọc ảnh xám làm gốc
img_goc = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Tạo ảnh tham chiếu "giả" (ví dụ: một ảnh có độ tương phản cao, sáng hơn)
img_tham_chieu = cv2.convertScaleAbs(img_goc, alpha=1.5, beta=20)

# Tạo ảnh nguồn "giả" (ví dụ: một ảnh bị tối, mờ)
img_nguon_toi = cv2.convertScaleAbs(img_goc, alpha=0.5, beta=-10)

# Thực hiện khớp lược đồ
img_khop = khop_luoc_do(img_nguon_toi, img_tham_chieu)

# === HIỂN THỊ KẾT QUẢ ===
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Hiển thị ảnh
du_lieu_anh = [
    (img_nguon_toi, "1. Nguồn (Tối/Mờ)"),
    (img_tham_chieu, "2. Tham chiếu (Sáng/Rõ)"),
    (img_khop, "3. Kết quả Khớp lược đồ")
]

for i, (anh, tieu_de) in enumerate(du_lieu_anh):
    axes[0, i].imshow(anh, cmap="gray")
    axes[0, i].set_title(tieu_de)
    axes[0, i].axis("off")
    
    # Hiển thị Histogram tương ứng ở hàng dưới
    axes[1, i].hist(anh.ravel(), 256, [0, 256], color='teal', alpha=0.7)
    axes[1, i].set_xlim([0, 256])
    axes[1, i].set_title(f"Histogram - {i+1}")

plt.tight_layout()
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f"{output_dir}/histogram_matching.png", dpi=150)
plt.show()

print("✓ Đã hoàn thành khớp lược đồ. Hãy so sánh Histogram 2 và 3 để thấy sự tương đồng!")