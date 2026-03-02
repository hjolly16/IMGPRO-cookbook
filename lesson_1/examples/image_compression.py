import cv2
import numpy as np
import os

# === BƯỚC 1: ĐẦU VÀO ===
img_path = "images/sample.jpg"
if not os.path.exists(img_path):
    print(f"Lỗi: Không tìm thấy ảnh tại {img_path}")
    exit()

img = cv2.imread(img_path, cv2.IMREAD_COLOR)

print(f"--- THÔNG SỐ ẢNH GỐC ---")
print(f"Kích thước: {img.shape}")
print(f"Dung lượng raw trong RAM: {img.nbytes:,} bytes ({img.nbytes/1024/1024:.2f} MB)\n")

# === BƯỚC 2: LƯU DƯỚI CÁC ĐỊNH DẠNG KHÁC NHAU ===
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# 1. PNG — Nén không mất dữ liệu (Lossless)
cv2.imwrite(f"{output_dir}/test.png", img)

# 2. JPEG — Nén có mất dữ liệu (Lossy) với các mức chất lượng khác nhau
# Mặc định của OpenCV cho JPEG thường là 95
for q in [95, 75, 50, 25, 10]:
    cv2.imwrite(f"{output_dir}/test_q{q}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])

# 3. BMP — Không nén (Raw)
cv2.imwrite(f"{output_dir}/test.bmp", img)

# === BƯỚC 3: SO SÁNH KÍCH THƯỚC TỆP TRÊN ĐĨA ===
print(f"{'Định dạng':<20} {'Kích thước':>12} {'Tỉ lệ nén':>12}")
print("-" * 46)

file_list = ["test.bmp", "test.png", "test_q95.jpg", "test_q75.jpg", "test_q50.jpg", "test_q25.jpg", "test_q10.jpg"]

for ten_file in file_list:
    duong_dan = f"{output_dir}/{ten_file}"
    kich_thuoc = os.path.getsize(duong_dan)
    ti_le = (kich_thuoc / img.nbytes) * 100
    print(f"{ten_file:<20} {kich_thuoc:>10,} B {ti_le:>11.1f}%")

# === BƯỚC 4: KIỂM TRA ĐỘ MẤT MÁT DỮ LIỆU (MSE/PSNR) ===
print(f"\n{'Chất lượng JPEG':<16} {'MSE':>10} {'PSNR (dB)':>12}")
print("-" * 40)

for q in [95, 75, 50, 25, 10]:
    img_jpeg = cv2.imread(f"{output_dir}/test_q{q}.jpg")
    
    # Tính sai số bình phương trung bình (MSE)
    mse = np.mean((img.astype(np.float64) - img_jpeg.astype(np.float64)) ** 2)
    
    # Tính tỷ số tín hiệu trên nhiễu (PSNR)
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 10 * np.log10(255.0 ** 2 / mse)
        
    print(f"Q = {q:<12} {mse:>10.2f} {psnr:>12.2f}")

# === BƯỚC 5: MINH HỌA SAI LẦM KHI LƯU MASK BẰNG JPEG ===
print("\n=== CẢNH BÁO: MẶT NẠ NHỊ PHÂN (PNG vs JPEG) ===")
mask = np.zeros((100, 100), dtype=np.uint8)
cv2.circle(mask, (50, 50), 30, 255, -1)  # Vẽ hình tròn trắng trên nền đen

# Lưu mặt nạ
cv2.imwrite(f"{output_dir}/mask.png", mask)
cv2.imwrite(f"{output_dir}/mask.jpg", mask, [cv2.IMWRITE_JPEG_QUALITY, 75])

# Đọc lại để kiểm tra
mask_png = cv2.imread(f"{output_dir}/mask.png", cv2.IMREAD_GRAYSCALE)
mask_jpg = cv2.imread(f"{output_dir}/mask.jpg", cv2.IMREAD_GRAYSCALE)

# Lấy các giá trị pixel duy nhất
val_png = np.unique(mask_png)
val_jpg = np.unique(mask_jpg)

print(f"  PNG - Các giá trị pixel: {val_png} (Giữ nguyên nhị phân)")
print(f"  JPG - Các giá trị pixel: {len(val_jpg)} giá trị khác nhau (Bị nhiễu hạt!)")
print(f"  => Lời khuyên: LUÔN lưu Ground Truth/Mask bằng định dạng .PNG!")