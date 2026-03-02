import cv2
import numpy as np
import time
import os

# === BƯỚC 1: CHUẨN BỊ DỮ LIỆU LỚN ===
# Tạo một ảnh ngẫu nhiên kích thước lớn (6 Megapixels) để thấy rõ sự khác biệt
# 2000x3000 pixel, kiểu uint8
img = np.random.randint(0, 256, (2000, 3000), dtype=np.uint8)
gamma = 0.5

print(f"--- THÔNG SỐ THỬ NGHIỆM ---")
print(f"Kích thước ảnh : {img.shape[1]}x{img.shape[0]}")
print(f"Dung lượng RAM : {img.nbytes/1024/1024:.1f} MB")
print(f"Số lượng pixel : {img.size:,} pixels\n")

# === CÁCH 1: TÍNH TOÁN TRỰC TIẾP (BRUTE FORCE) ===
# Với mỗi pixel, máy tính phải thực hiện: chia, lũy thừa, nhân, ép kiểu
start = time.perf_counter()
for _ in range(10):  # Chạy 10 lần để lấy trung bình
    img_float = img.astype(np.float64) / 255.0
    ket_qua_1 = np.clip(255 * (img_float ** gamma), 0, 255).astype(np.uint8)
t1 = (time.perf_counter() - start) / 10

# === CÁCH 2: SỬ DỤNG LUT (TỐI ƯU HÓA) ===
# Thay vì tính cho 6 triệu pixel, ta chỉ tính cho 256 giá trị đầu vào có thể có
lut = np.array([
    np.clip(255 * (i / 255.0)**gamma, 0, 255)
    for i in range(256)
], dtype=np.uint8)

start = time.perf_counter()
for _ in range(10):
    # Hàm cv2.LUT sẽ ánh xạ giá trị pixel dựa trên bảng tra
    ket_qua_2 = cv2.LUT(img, lut)
t2 = (time.perf_counter() - start) / 10

# === PHÂN TÍCH KẾT QUẢ ===
print(f"--- KẾT QUẢ ĐO LƯỜNG ---")
print(f"Tính trực tiếp (Direct) : {t1*1000:.2f} ms")
print(f"Sử dụng LUT (Optimized) : {t2*1000:.2f} ms")
print(f"Tốc độ tăng trưởng     : {t1/t2:.1f} lần!")

# Kiểm tra tính chính xác
if np.array_equal(ket_qua_1, ket_qua_2):
    print("\n✓ Xác nhận: Hai kết quả hoàn toàn giống nhau.")
else:
    print("\n❌ Cảnh báo: Kết quả có sự sai lệch.")