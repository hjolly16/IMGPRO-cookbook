import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# === BƯỚC 1: ĐẦU VÀO DỮ LIỆU ===
img = cv2.imread("images/sample.jpg")
if img is None:
    # Tạo ảnh mô phỏng nếu không tìm thấy file thực tế
    img = np.zeros((400, 500, 3), dtype=np.uint8)
    # Tạo nền gradient phức tạp
    for y in range(400):
        for x in range(500):
            img[y, x] = [80 + x // 5, 120 + y // 4, 60 + x // 6]
    # Vẽ đối tượng mô phỏng (hình elip)
    cv2.ellipse(img, (250, 200), (100, 130), 0, 0, 360, (50, 100, 200), -1)
    cv2.ellipse(img, (250, 200), (80, 110), 0, 0, 360, (60, 120, 220), -1)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]

# === BƯỚC 2: GRABCUT VỚI KHỞI TẠO HÌNH CHỮ NHẬT (BBOX) ===
# Định nghĩa vùng chứa đối tượng (x, y, w, h)
rect = (800, 700, 600, 640)

mask = np.zeros((h, w), dtype=np.uint8)
# Hai mảng tạm thời cần thiết cho thuật toán chạy nội bộ
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

# Lần chạy 1: Khởi tạo bằng Bbox (GC_INIT_WITH_RECT)
# Thuật toán sẽ tự coi vùng ngoài Bbox là nền, bên trong là "có thể là vật thể"
cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

# Lấy mask nhị phân: Chuyển vùng "Vật thể" và "Có thể là vật thể" thành 1, còn lại là 0
mask_bin1 = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
result1 = img_rgb * mask_bin1[:, :, np.newaxis]

# === BƯỚC 3: TINH CHỈNH KẾT QUẢ BẰNG MASK THỦ CÔNG ===
# Trong thực tế, đây là lúc người dùng dùng cọ vẽ thêm các nét "Chắc chắn"
mask_refined = mask.copy()

# Giả sử đánh dấu: Vùng trung tâm chắc chắn là Foreground (GC_FGD = 1)
cv2.circle(mask_refined, (250, 200), 40, 1, -1)

# Đánh dấu các góc chắc chắn là Background (GC_BGD = 0)
cv2.rectangle(mask_refined, (0, 0), (50, 50), 0, -1)
cv2.rectangle(mask_refined, (w - 50, 0), (w, 50), 0, -1)

# Lần chạy 2: Tinh chỉnh dựa trên thông tin vẽ thêm (GC_INIT_WITH_MASK)
cv2.grabCut(img, mask_refined, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)

# Tạo mask nhị phân cuối cùng
mask_bin2 = np.where((mask_refined == 2) | (mask_refined == 0), 0, 1).astype(np.uint8)

# === BƯỚC 4: HẬU XỬ LÝ (POST-PROCESSING) ===
# Sử dụng Morphology để lấp đầy các lỗ nhỏ và làm sạch biên
se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask_clean = cv2.morphologyEx(mask_bin2, cv2.MORPH_CLOSE, se, iterations=2)
mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, se, iterations=1)

# Làm mịn biên bằng GaussianBlur để tránh hiện tượng răng cưa
mask_smooth = cv2.GaussianBlur(mask_clean.astype(np.float32), (5, 5), 0)
_, mask_smooth = cv2.threshold((mask_smooth * 255).astype(np.uint8), 127, 1, cv2.THRESH_BINARY)

result_clean = img_rgb * mask_smooth[:, :, np.newaxis]

# === BƯỚC 5: HIỂN THỊ KẾT QUẢ ===
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Ảnh gốc và Bbox khởi tạo
axes[0, 0].imshow(img_rgb)
rx, ry, rw, rh = rect
rect_patch = plt.Rectangle((rx, ry), rw, rh, linewidth=2, edgecolor="lime", facecolor="none")
axes[0, 0].add_patch(rect_patch)
axes[0, 0].set_title("1. Ảnh gốc + Bbox khởi tạo")
axes[0, 0].axis("off")

# Mask thô sau lần chạy đầu tiên
axes[0, 1].imshow(mask_bin1, cmap="gray")
axes[0, 1].set_title("2. Mask nhị phân (Lần 1)")
axes[0, 1].axis("off")

# Kết quả tách nền lần 1
axes[0, 2].imshow(result1)
axes[0, 2].set_title("3. Kết quả tách nền (Lần 1)")
axes[0, 2].axis("off")

# Hiển thị các nhãn trạng thái trong Mask tinh chỉnh
axes[1, 0].imshow(mask_refined.astype(np.float32), cmap="nipy_spectral")
axes[1, 0].set_title("4. Bản đồ nhãn tinh chỉnh\n(0:BGD, 1:FGD, 2:PR_BGD, 3:PR_FGD)")
axes[1, 0].axis("off")

# Mask sạch sau hậu xử lý
axes[1, 1].imshow(mask_smooth, cmap="gray")
axes[1, 1].set_title("5. Mask cuối (Đã làm mịn biên)")
axes[1, 1].axis("off")

# Kết quả cuối cùng
axes[1, 2].imshow(result_clean)
axes[1, 2].set_title("6. Kết quả tách nền cuối cùng")
axes[1, 2].axis("off")

plt.suptitle("Thuật toán GrabCut: Phân đoạn đối tượng tương tác", fontsize=16)
plt.tight_layout()

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f"{output_dir}/grabcut_analysis.png", dpi=150)
plt.show()