import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def can_bang_trang_gray_world(img_bgr):
    """
    Cân bằng trắng theo giả thuyết Thế giới Xám (Gray World Hypothesis).
    Giả định rằng trung bình các kênh màu trong một bức ảnh chuẩn sẽ là màu xám trung tính.
    """
    # Tính giá trị trung bình mỗi kênh (B, G, R)
    mu_b = img_bgr[:, :, 0].mean()
    mu_g = img_bgr[:, :, 1].mean()
    mu_r = img_bgr[:, :, 2].mean()
    
    # Tính giá trị trung bình tổng thể của cả 3 kênh
    mu_tong = (mu_b + mu_g + mu_r) / 3.0

    # Tính hệ số điều chỉnh cho từng kênh (Scaling factors)
    k_b = mu_tong / mu_b
    k_g = mu_tong / mu_g
    k_r = mu_tong / mu_r

    print("=== Thông số thuật toán ===")
    print(f"  Trung bình gốc: B={mu_b:.1f}, G={mu_g:.1f}, R={mu_r:.1f}")
    print(f"  Hệ số điều chỉnh: k_B={k_b:.3f}, k_G={k_g:.3f}, k_R={k_r:.3f}")

    # Áp dụng nhân hệ số - Thực hiện trên float32 để giữ độ chính xác
    ket_qua = img_bgr.astype(np.float32)
    ket_qua[:, :, 0] *= k_b
    ket_qua[:, :, 1] *= k_g
    ket_qua[:, :, 2] *= k_r

    # Đưa giá trị về khoảng [0, 255] và chuyển về uint8
    ket_qua = np.clip(ket_qua, 0, 255).astype(np.uint8)
    return ket_qua

# === CHƯƠNG TRÌNH CHÍNH ===
img_path = "images/sample.jpg"
if not os.path.exists(img_path):
    print(f"Lỗi: Không tìm thấy ảnh tại {img_path}")
    exit()

img_bgr = cv2.imread(img_path)

# Áp dụng thuật toán Gray World
img_cb = can_bang_trang_gray_world(img_bgr)

# Hiển thị kết quả so sánh
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Chuyển sang RGB để Matplotlib hiển thị đúng
axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
axes[0].set_title("1. Trước cân bằng trắng (Bị ám màu)")
axes[0].axis("off")

axes[1].imshow(cv2.cvtColor(img_cb, cv2.COLOR_BGR2RGB))
axes[1].set_title("2. Sau cân bằng trắng (Gray World)")
axes[1].axis("off")

plt.tight_layout()

# Lưu kết quả
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
plt.savefig(f"{output_dir}/ket_qua_white_balance.png", dpi=150)

print(f"\n✓ Đã hoàn tất! Kết quả được lưu tại {output_dir}/ket_qua_white_balance.png")
plt.show()