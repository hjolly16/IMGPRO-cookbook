import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# === BƯỚC 1: TẠO ẢNH NHỊ PHÂN MẪU MÔ PHỎNG THỰC TẾ ===
# Tạo nền đen
img = np.zeros((300, 400), dtype=np.uint8)

# Vẽ đối tượng chính (Màu trắng)
cv2.rectangle(img, (50, 50), (200, 200), 255, -1) # Hình chữ nhật
cv2.circle(img, (300, 150), 60, 255, -1)          # Hình tròn

# 1. Thêm "nhiễu ngoài" (Salt noise - các đốm trắng li ti ngoài nền đen)
np.random.seed(42)
nhieu_ngoai = np.random.rand(300, 400) < 0.005
img[nhieu_ngoai] = 255

# 2. Thêm "nhiễu trong" (Pepper noise - các lỗ đen nhỏ bên trong vật thể)
for _ in range(15):
    cx = np.random.randint(60, 190)
    cy = np.random.randint(60, 190)
    cv2.circle(img, (cx, cy), 2, 0, -1)

# === BƯỚC 2: TẠO CÁC PHẦN TỬ CẤU TRÚC (STRUCTURING ELEMENTS - SE) ===
# SE giống như một chiếc "kính lúp" có hình dáng cụ thể để quét qua ảnh
se_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
se_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
se_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

print("--- MA TRẬN PHẦN TỬ CẤU TRÚC (5x5) ---")
print("1. Hình chữ nhật (RECT):\n", se_rect)
print("\n2. Hình chữ thập (CROSS):\n", se_cross)
print("\n3. Hình elip/tròn (ELLIPSE):\n", se_ellipse)

# === BƯỚC 3: THỰC HIỆN CO (EROSION) VÀ GIÃN (DILATION) ===
# Dùng SE hình Elip (thường cho kết quả tự nhiên nhất với vật thể cong)
eroded = cv2.erode(img, se_ellipse, iterations=1)
dilated = cv2.dilate(img, se_ellipse, iterations=1)

# Đếm pixel trắng để thấy sự thay đổi về diện tích
px_goc = np.sum(img > 0)
px_co = np.sum(eroded > 0)
px_gian = np.sum(dilated > 0)

print(f"\n--- DIỆN TÍCH VẬT THỂ (Số pixel trắng) ---")
print(f"Gốc : {px_goc:>8}")
print(f"Co  : {px_co:>8} (Mất đi {px_goc - px_co} pixel)")
print(f"Giãn: {px_gian:>8} (Tăng thêm {px_gian - px_goc} pixel)")

# === BƯỚC 4: HIỂN THỊ SO SÁNH ===
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(img, cmap="gray")
axes[0].set_title("1. Ảnh gốc\n(Nhiễu đốm trắng & Lỗ đen)")
axes[0].axis("off")

axes[1].imshow(eroded, cmap="gray")
axes[1].set_title("2. Phép CO (Erosion)\n✓ Xóa đốm trắng lấm tấm\n✗ Vật thể bị teo nhỏ & lỗ to ra")
axes[1].axis("off")

axes[2].imshow(dilated, cmap="gray")
axes[2].set_title("3. Phép GIÃN (Dilation)\n✓ Lấp đầy các lỗ đen\n✗ Đốm nhiễu phình to & vật thể béo lên")
axes[2].axis("off")

plt.tight_layout()
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f"{output_dir}/morph_co_gian_basic.png", dpi=150)
plt.show()

# So sánh 3 hình dạng SE khi áp dụng phép Co
fig2, axes2 = plt.subplots(1, 4, figsize=(20, 5))
axes2[0].imshow(img, cmap="gray")
axes2[0].set_title("Ảnh Gốc")
axes2[0].axis("off")

danh_sach_se = [
    (se_rect, "Chữ nhật (Góc cạnh)"),
    (se_cross, "Chữ thập (Nhọn)"),
    (se_ellipse, "Elip (Mềm mại)")
]

for idx, (se, ten) in enumerate(danh_sach_se):
    result = cv2.erode(img, se, iterations=1)
    axes2[idx + 1].imshow(result, cmap="gray")
    axes2[idx + 1].set_title(f"Co bằng SE {ten}")
    axes2[idx + 1].axis("off")

plt.suptitle("Sự ảnh hưởng của hình dáng Structuring Element (SE)", fontsize=16)
plt.tight_layout()
plt.savefig(f"{output_dir}/morph_se_comparison.png", dpi=150)
plt.show()