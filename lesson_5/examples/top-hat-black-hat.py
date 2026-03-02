import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# === BƯỚC 1: TẠO ẢNH MÔ PHỎNG CHIẾU SÁNG KHÔNG ĐỀU ===
h, w = 400, 500

# 1. Tạo chữ đen trên nền trắng (Tài liệu lý tưởng)
text_img = np.ones((h, w), dtype=np.uint8) * 255
cv2.putText(text_img, "MORPHOLOGY", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 0, 4)
cv2.putText(text_img, "Top-Hat", (80, 220), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 0, 4)
cv2.putText(text_img, "Black-Hat", (60, 320), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 0, 4)

# 2. Tạo gradient chiếu sáng (Bên trái sáng - Bên phải tối)
gradient = np.zeros((h, w), dtype=np.float32)
for col in range(w):
    gradient[:, col] = 1.0 - 0.6 * (col / w)

# Thêm hiệu ứng tối dần ở góc dưới cùng bên phải
y_coords, x_coords = np.mgrid[0:h, 0:w]
dist = np.sqrt((x_coords - w) ** 2 + (y_coords - h) ** 2)
gradient *= (1.0 - 0.3 * dist / dist.max())

# 3. Áp gradient chiếu sáng lên ảnh chữ
img_uneven = (text_img.astype(np.float32) * gradient).astype(np.uint8)

# === BƯỚC 2: PHÂN TÍCH VỚI TOP-HAT VÀ BLACK-HAT ===
se_sizes = [15, 25, 45]
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

# Hàng 1: Ảnh gốc, ảnh lỗi ánh sáng, và sự thất bại của Otsu
axes[0, 0].imshow(text_img, cmap="gray")
axes[0, 0].set_title("1. Ảnh gốc lý tưởng")
axes[0, 0].axis("off")

axes[0, 1].imshow(img_uneven, cmap="gray")
axes[0, 1].set_title("2. Chiếu sáng không đều")
axes[0, 1].axis("off")

# Phân ngưỡng Otsu toàn cục - Sẽ thất bại vì ảnh bị tối một góc
_, thresh_global = cv2.threshold(img_uneven, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
axes[0, 2].imshow(thresh_global, cmap="gray")
axes[0, 2].set_title("3. Otsu (Thất bại)\nMất chữ ở vùng tối!")
axes[0, 2].axis("off")

# Hàng 2 & 3: Black-hat với các kích thước phần tử cấu trúc (SE) khác nhau
for idx, se_size in enumerate(se_sizes):
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size, se_size))

    # Black-hat: Trích xuất các chi tiết TỐI trên nền SÁNG (chữ đen)
    black_hat = cv2.morphologyEx(img_uneven, cv2.MORPH_BLACKHAT, se)

    # Phân ngưỡng lại trên kết quả Black-hat
    _, thresh_bh = cv2.threshold(black_hat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Đảo ngược màu (bitwise_not) để chữ thành đen, nền thành trắng cho giống ảnh gốc ban đầu
    thresh_bh_inv = cv2.bitwise_not(thresh_bh)

    axes[1, idx].imshow(black_hat, cmap="gray")
    axes[1, idx].set_title(f"Black-hat (SE={se_size})\nLấy chi tiết tối")
    axes[1, idx].axis("off")

    axes[2, idx].imshow(thresh_bh_inv, cmap="gray")
    axes[2, idx].set_title(f"Phân ngưỡng Black-hat\n(Chữ màu đen)")
    axes[2, idx].axis("off")

plt.suptitle("Sử dụng Black-Hat để trích xuất văn bản", fontsize=16)
plt.tight_layout()

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f"{output_dir}/morph_blackhat.png", dpi=150)
plt.show()

# === BƯỚC 3: CÔNG THỨC CHUẨN ĐỂ CÂN BẰNG CHIẾU SÁNG KHÔNG ĐỀU ===
# 1. Dùng phép CLOSING với SE lớn để ước lượng nền (xóa sạch chữ đen)
# Lựa chọn SE=45 vì nó đủ lớn để nuốt trọn nét chữ to nhất
se_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45)) 
background = cv2.morphologyEx(img_uneven, cv2.MORPH_CLOSE, se_big)

# 2. Trừ ảnh gốc khỏi nền để trích xuất phần chữ
# Lúc này phần nền chênh lệch ánh sáng sẽ tự triệt tiêu lẫn nhau, 
# chỉ để lại phần chữ (màu sáng) nổi bật trên một nền đen đồng nhất.
text_extracted = cv2.subtract(background, img_uneven)

# 3. Đảo ngược màu để trả về format "chữ đen nền trắng" quen thuộc
corrected = cv2.bitwise_not(text_extracted)

fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
axes2[0].imshow(img_uneven, cmap="gray")
axes2[0].set_title("1. Chiếu sáng không đều")
axes2[0].axis("off")

axes2[1].imshow(corrected, cmap="gray")
axes2[1].set_title("2. Sau khi cân bằng nền\n(Gốc được khử bằng Morph Close)")
axes2[1].axis("off")

# Nền bây giờ đã đồng nhất, Otsu toàn cục có thể hoạt động
_, thresh_corrected = cv2.threshold(corrected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
axes2[2].imshow(thresh_corrected, cmap="gray")
axes2[2].set_title("3. Otsu sau khi cân bằng\n(chuẩn)")
axes2[2].axis("off")

plt.tight_layout()
plt.savefig(f"{output_dir}/morph_illumination_correction.png", dpi=150)
plt.show()