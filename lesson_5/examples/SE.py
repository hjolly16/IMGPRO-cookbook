import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# === BƯỚC 1: TẠO ẢNH GIẢ LẬP ĐA KÍCH THƯỚC ===
img = np.zeros((400, 500), dtype=np.uint8)

# 1. Đối tượng chính (Kích thước lớn)
cv2.rectangle(img, (30, 30), (230, 200), 255, -1)
cv2.circle(img, (370, 120), 80, 255, -1)

# 2. Đối tượng quan trọng nhưng nhỏ (Cần giữ lại)
cv2.rectangle(img, (50, 280), (90, 350), 255, -1)   # Kích thước 40x70 px
cv2.circle(img, (200, 320), 20, 255, -1)            # Bán kính 20 px

# Tổng cộng chúng ta có đúng 4 đối tượng thật sự cần quan tâm

# 3. Nhiễu (Các chấm rất nhỏ, bán kính từ 1 đến 3 px)
np.random.seed(42)  # Cố định seed để kết quả giống nhau mỗi lần chạy
for _ in range(80):
    cx = np.random.randint(0, 500)
    cy = np.random.randint(0, 400)
    r = np.random.randint(1, 4)
    cv2.circle(img, (cx, cy), r, 255, -1)

# === BƯỚC 2: KHẢO SÁT ẢNH HƯỞNG CỦA KÍCH THƯỚC SE ===
kich_thuoc_se = [3, 5, 7, 9, 11, 15]
fig, axes = plt.subplots(2, len(kich_thuoc_se) // 2 + 1, figsize=(18, 8))
axes_flat = axes.flatten()

axes_flat[0].imshow(img, cmap="gray")
axes_flat[0].set_title("1. Ảnh gốc\n(Nhiều nhiễu nhỏ)")
axes_flat[0].axis("off")

# Đếm số lượng thành phần liên thông trên ảnh gốc
# Trừ 1 vì connectedComponents đếm cả phần nền đen (background) là 1 thành phần
num_orig = cv2.connectedComponents(img)[0] - 1 

ket_qua = []

for idx, k in enumerate(kich_thuoc_se):
    # Tạo SE hình Elip với kích thước k x k
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    
    # Thực hiện phép Opening (Co trước, Giãn sau)
    result = cv2.morphologyEx(img, cv2.MORPH_OPEN, se)
    
    # Đếm số lượng vật thể còn lại sau khi Opening
    num_cc = cv2.connectedComponents(result)[0] - 1
    pixel_count = np.sum(result > 0)
    ket_qua.append({"k": k, "cc": num_cc, "pixels": pixel_count})

    axes_flat[idx + 1].imshow(result, cmap="gray")
    axes_flat[idx + 1].set_title(f"Opening SE = {k}x{k}\n{num_cc} đối tượng")
    axes_flat[idx + 1].axis("off")

# Ẩn các ô đồ thị thừa nếu có
for j in range(len(kich_thuoc_se) + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)

plt.suptitle("Sự ảnh hưởng của kích thước Phần tử cấu trúc (SE) lên phép Opening", fontsize=16)
plt.tight_layout()

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f"{output_dir}/morph_chon_se.png", dpi=150)
plt.show()

# === BƯỚC 3: IN BẢNG ĐÁNH GIÁ ===
print(f"{'SE (k)':>6} {'Số đối tượng':>15} {'Pixel trắng':>15}   {'Nhận xét'}")
print("-" * 75)
print(f"{'Gốc':>6} {num_orig:>15} {np.sum(img > 0):>15}   Còn lẫn rất nhiều nhiễu")

for r in ket_qua:
    # 4 là số lượng đối tượng thật sự (2 lớn + 2 nhỏ)
    if r["cc"] > 4:
        note = "❌ Còn nhiễu (SE quá nhỏ)"
    elif r["cc"] == 4:
        note = "✅ Vừa đủ (Xóa sạch nhiễu, giữ đủ 4 vật thể)"
    else:
        note = "⚠️ Mất đối tượng nhỏ! (SE quá to)"
    
    print(f"{r['k']:>6} {r['cc']:>15} {r['pixels']:>15}   {note}")

print(f"\n=> BÀI HỌC RÚT RA: Kích thước SE tối ưu là kích thước vừa đủ lớn để nuốt trọn hạt nhiễu to nhất, nhưng phải nhỏ hơn vật thể bé nhất cần giữ lại.")