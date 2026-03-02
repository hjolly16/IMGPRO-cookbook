import cv2
import numpy as np
import os

# === BƯỚC 1: ĐỌC ẢNH VÀ KIỂM TRA SIÊU DỮ LIỆU (METADATA) ===
img_path = "images/sample.jpg"
if not os.path.exists(img_path):
    print(f"Lỗi: Không tìm thấy file tại {img_path}")
    exit()

img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)

# Kiểm tra shape, dtype, strides
print("=== 1. THÔNG SỐ ẢNH GỐC (UINT8) ===")
print(f"  Shape (H, W, C) : {img_bgr.shape}")
print(f"  Dtype           : {img_bgr.dtype}") # Thường là uint8 (0-255)
# Strides: Số byte cần bước qua để sang hàng tiếp theo, pixel tiếp theo
print(f"  Strides         : {img_bgr.strides}") 
print(f"  Giá trị Min/Max : {img_bgr.min()}/{img_bgr.max()}")
print(f"  Dung lượng bộ nhớ: {img_bgr.nbytes:,} bytes ({img_bgr.nbytes/1024/1024:.2f} MB)")

# === BƯỚC 2: CHUYỂN ĐỔI SANG FLOAT32 (NORMALIZATION) ===
# Thường dùng trong Deep Learning để đưa giá trị về khoảng [0, 1]
img_float = img_bgr.astype(np.float32) / 255.0

print("\n=== 2. THÔNG SỐ ẢNH FLOAT32 (SAU CHUẨN HÓA) ===")
print(f"  Dtype           : {img_float.dtype}")
print(f"  Giá trị Min/Max : {img_float.min():.4f}/{img_float.max():.4f}")
# Float32 chiếm 4 bytes/phần tử, trong khi uint8 chỉ chiếm 1 byte
print(f"  Dung lượng bộ nhớ: {img_float.nbytes:,} bytes ({img_float.nbytes/1024/1024:.2f} MB)")
print(f"  => Tốn bộ nhớ gấp {img_float.nbytes / img_bgr.nbytes:.0f} lần so với uint8")

# === BƯỚC 3: CHUYỂN NGƯỢC FLOAT32 → UINT8 ===
# Quan trọng: Phải dùng np.clip để tránh lỗi tràn số khi giá trị ngoài [0, 255]
img_back = np.clip(img_float * 255.0, 0, 255).astype(np.uint8)

# Kiểm tra tính toàn vẹn của dữ liệu
sai_khac = np.sum(img_bgr != img_back)
print("\n=== 3. KIỂM TRA CHUYỂN ĐỔI NGƯỢC ===")
print(f"  Số pixel bị sai lệch: {sai_khac} (Phải bằng 0 nếu chuyển đổi chuẩn)")

# === BƯỚC 4: CÁC BIẾN THỂ KHÔNG GIAN MÀU KHÁC ===
# Ảnh xám (Grayscale) - Loại bỏ kênh màu, chỉ giữ lại cường độ sáng
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
print(f"\n=== 4. THÔNG TIN ẢNH XÁM ===")
print(f"  Shape           : {img_gray.shape}") # Chỉ còn (H, W), mất kênh C
print(f"  Kích thước bộ nhớ: {img_gray.nbytes:,} bytes")

# Ảnh RGBA (Thêm kênh Alpha - Độ trong suốt)
# Cách 1: Dùng OpenCV (Nhanh nhất)
img_rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)

# Cách 2: Thủ công (tốn thời gian hơn, chỉ để minh họa)
# b, g, r = cv2.split(img_bgr)
# alpha = np.ones(b.shape, dtype=np.uint8) * 255
# img_rgba_manual = cv2.merge([b, g, r, alpha])

print(f"\n=== 5. THÔNG TIN ẢNH RGBA (4 KÊNH) ===")
print(f"  Shape           : {img_rgba.shape}") # (H, W, 4)
print(f"  Dung lượng bộ nhớ: {img_rgba.nbytes:,} bytes")

# Lưu các phiên bản ảnh
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cv2.imwrite(f"{output_dir}/sample_gray.png", img_gray)
cv2.imwrite(f"{output_dir}/sample_rgba.png", img_rgba)
print(f"\n✓ Đã lưu ảnh xám và RGBA vào thư mục '{output_dir}'")