import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def sap_xep_4_goc(pts):
    """
    Sắp xếp 4 tọa độ theo thứ tự chuẩn: 
    [trên-trái, trên-phải, dưới-phải, dưới-trái].
    Việc sắp xếp này giúp ma trận Transform luôn nhất quán.
    """
    pts = pts.reshape(4, 2)
    sorted_pts = np.zeros((4, 2), dtype=np.float32)

    # 1. Tổng x + y: Nhỏ nhất là trên-trái (Top-Left), lớn nhất là dưới-phải (Bottom-Right)
    s = pts.sum(axis=1)
    sorted_pts[0] = pts[np.argmin(s)]
    sorted_pts[2] = pts[np.argmax(s)]

    # 2. Hiệu y - x: Nhỏ nhất là trên-phải (Top-Right), lớn nhất là dưới-trái (Bottom-Left)
    d = np.diff(pts, axis=1).ravel()
    sorted_pts[1] = pts[np.argmin(d)]
    sorted_pts[3] = pts[np.argmax(d)]

    return sorted_pts

def quet_tai_lieu(img, pts_4goc):
    """Sử dụng Perspective Warp để đưa tài liệu về dạng phẳng chính diện."""
    pts = sap_xep_4_goc(pts_4goc)
    tl, tr, br, bl = pts

    # Tính toán kích thước ảnh đầu ra (dựa trên chiều dài tối đa các cạnh)
    # Tránh tình trạng ảnh sau khi nắn bị méo tỷ lệ
    w_top = np.linalg.norm(tr - tl)
    w_bot = np.linalg.norm(br - bl)
    w_out = int(max(w_top, w_bot))

    h_left = np.linalg.norm(bl - tl)
    h_right = np.linalg.norm(br - tr)
    h_out = int(max(h_left, h_right))

    # Định nghĩa các điểm đích tương ứng (hình chữ nhật phẳng)
    dst = np.float32([[0, 0], [w_out - 1, 0], 
                      [w_out - 1, h_out - 1], [0, h_out - 1]])

    # Tính toán ma trận phối cảnh (Perspective Matrix)
    M = cv2.getPerspectiveTransform(pts, dst)
    
    # Thực hiện Warp phối cảnh
    warped = cv2.warpPerspective(img, M, (w_out, h_out), flags=cv2.INTER_CUBIC)
    
    return warped, M

# === BƯỚC 1: TẠO DỮ LIỆU MÔ PHỎNG (ẢNH CHỤP NGHIÊNG) ===
# 1. Tạo một tài liệu phẳng giả lập
doc = np.ones((400, 300, 3), dtype=np.uint8) * 240
cv2.putText(doc, "IMGPRO", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (30, 30, 30), 3)
cv2.putText(doc, "Tutorial: Perspective", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
cv2.putText(doc, "Author: T&A Lab", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
cv2.rectangle(doc, (20, 20), (280, 380), (100, 100, 100), 2)

# 2. Giả lập chụp nghiêng bằng cách warp tài liệu vào một không gian 600x600
h_doc, w_doc = doc.shape[:2]
src_pts = np.float32([[0, 0], [w_doc, 0], [w_doc, h_doc], [0, h_doc]])
dst_warp = np.float32([[80, 60], [520, 30], [480, 550], [50, 500]])

M_warp = cv2.getPerspectiveTransform(src_pts, dst_warp)
canvas = np.ones((600, 600, 3), dtype=np.uint8) * 80 # Nền tối xung quanh
warped_doc = cv2.warpPerspective(doc, M_warp, (600, 600), 
                                 dst=canvas, 
                                 borderMode=cv2.BORDER_TRANSPARENT)

# === BƯỚC 2: THỰC HIỆN QUÉT VÀ NẮN PHẲNG ===
# Trong thực tế, 4 điểm dst_warp sẽ được tìm thấy bằng thuật toán tìm Contour
result, M_inv = quet_tai_lieu(warped_doc, dst_warp)

# === BƯỚC 3: HIỂN THỊ KẾT QUẢ ===
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(cv2.cvtColor(doc, cv2.COLOR_BGR2RGB))
axes[0].set_title("1. Tài liệu gốc (Lý tưởng)")
axes[0].axis("off")

# Vẽ 4 góc đã đánh dấu lên ảnh chụp nghiêng
warped_draw = warped_doc.copy()
for i, pt in enumerate(dst_warp.astype(int)):
    cv2.circle(warped_draw, tuple(pt), 8, (0, 0, 255), -1)
    cv2.putText(warped_draw, str(i), (pt[0] + 15, pt[1] - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

axes[1].imshow(cv2.cvtColor(warped_draw, cv2.COLOR_BGR2RGB))
axes[1].set_title("2. Ảnh chụp nghiêng\n(Đã xác định 4 góc)")
axes[1].axis("off")

axes[2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
axes[2].set_title("3. Kết quả quét\n(Đã nắn phẳng chính diện)")
axes[2].axis("off")

plt.suptitle("Kỹ thuật Digital Scan bằng Perspective Transform", fontsize=16)
plt.tight_layout()

# Lưu file đầu ra
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f"{output_dir}/perspective_scan_final.png", dpi=150)
plt.show()

print(f"Kích thước gốc: {w_doc}x{h_doc}")
print(f"Kích thước sau khi quét: {result.shape[1]}x{result.shape[0]}")