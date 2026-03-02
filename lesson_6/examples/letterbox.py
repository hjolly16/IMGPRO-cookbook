import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def letterbox(img, target_size=(640, 640), color=(114, 114, 114)):
    """
    Thay đổi kích thước ảnh (Resize) nhưng vẫn giữ nguyên tỷ lệ khung hình (Aspect Ratio).
    Phần không gian thừa sẽ được đệm (pad) bằng một màu cố định.
    Kỹ thuật này là tiêu chuẩn tiền xử lý cho các mô hình như YOLO.

    Tham số:
        img: Ảnh gốc đầu vào.
        target_size: Kích thước mục tiêu (width, height).
        color: Màu đệm viền (Mặc định là xám 114).

    Trả về:
        img_lb: Ảnh sau khi đã Letterbox.
        scale: Tỷ lệ đã dùng để thu/phóng ảnh.
        pad: Tuple chứa giá trị padding (pad_left, pad_top).
    """
    h, w = img.shape[:2]
    tw, th = target_size

    # 1. Tính toán tỷ lệ thu/phóng sao cho ảnh không bị méo
    scale = min(tw / w, th / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    # 2. Thay đổi kích thước ảnh với tỷ lệ đã tính
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 3. Tính toán phần viền cần đệm (Padding) để đạt đúng target_size
    pad_left = (tw - new_w) // 2
    pad_top = (th - new_h) // 2
    pad_right = tw - new_w - pad_left
    pad_bottom = th - new_h - pad_top

    # 4. Thêm viền đệm bằng hàm copyMakeBorder
    img_lb = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=color
    )

    return img_lb, scale, (pad_left, pad_top)

def letterbox_to_original(x_lb, y_lb, scale, pad):
    """
    Chuyển đổi ngược tọa độ (x, y) từ ảnh Letterbox về lại tọa độ trên ảnh gốc.
    Rất cần thiết khi mô hình AI dự đoán Bounding Box trên ảnh Letterbox và ta cần vẽ nó lên ảnh thật.
    """
    x_orig = (x_lb - pad[0]) / scale
    y_orig = (y_lb - pad[1]) / scale
    return x_orig, y_orig

# === BƯỚC 1: TẢI ẢNH ===
img_path = "images/sample.jpg"
if not os.path.exists(img_path):
    print(f"Lỗi: Không tìm thấy ảnh tại {img_path}")
    exit()

img = cv2.imread(img_path)
if img is None:
    raise ValueError("Không thể giải mã ảnh!")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]

# === BƯỚC 2: TIỀN XỬ LÝ ẢNH ===
# 1. Kỹ thuật Letterbox (Giữ tỷ lệ, đệm viền)
img_lb, scale, pad = letterbox(img, (640, 640))
img_lb_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)

# 2. Kỹ thuật Stretch (Bóp méo ảnh để lấp đầy 640x640)
img_stretch = cv2.resize(img, (640, 640))
img_stretch_rgb = cv2.cvtColor(img_stretch, cv2.COLOR_BGR2RGB)

# === BƯỚC 3: MÔ PHỎNG LẬP BẢN ĐỒ BOUNDING BOX (BBOX) ===
# Giả sử mô hình AI (như YOLO) phát hiện được một Bbox trên ảnh Letterbox 640x640
bbox_lb = (200, 150, 400, 350)  # (x1, y1, x2, y2)

# Khôi phục tọa độ Bbox này về hệ quy chiếu của ảnh gốc ban đầu
x1_orig, y1_orig = letterbox_to_original(bbox_lb[0], bbox_lb[1], scale, pad)
x2_orig, y2_orig = letterbox_to_original(bbox_lb[2], bbox_lb[3], scale, pad)

# Đảm bảo tọa độ khôi phục không vượt quá giới hạn khung ảnh gốc (Clamp)
x1_orig = max(0, int(x1_orig))
y1_orig = max(0, int(y1_orig))
x2_orig = min(w, int(x2_orig))
y2_orig = min(h, int(y2_orig))

# === BƯỚC 4: HIỂN THỊ VÀ SO SÁNH ===
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Hình 1: Ảnh gốc và Bbox đã được khôi phục tọa độ chuẩn xác
img_orig_draw = img_rgb.copy()
cv2.rectangle(img_orig_draw, (x1_orig, y1_orig), (x2_orig, y2_orig), (0, 255, 0), 3)
axes[0].imshow(img_orig_draw)
axes[0].set_title(f"Gốc + Bbox khôi phục\n({w}x{h} - Tỷ lệ: {w/h:.2f})")
axes[0].axis("off")

# Hình 2: Ảnh Letterbox (Tiêu chuẩn AI)
img_lb_draw = img_lb_rgb.copy()
cv2.rectangle(img_lb_draw, (bbox_lb[0], bbox_lb[1]), (bbox_lb[2], bbox_lb[3]), (0, 255, 0), 3)
axes[1].imshow(img_lb_draw)
axes[1].set_title(f"Letterbox (640x640)\nScale={scale:.3f}, Pad={pad}")
axes[1].axis("off")

# Hình 3: Ảnh Stretch (Lỗi méo hình)
axes[2].imshow(img_stretch_rgb)
axes[2].set_title("Stretch (640x640)\n(Sai lệch tỷ lệ - Bị méo!)")
axes[2].axis("off")

plt.suptitle("Tiền xử lý AI: Letterbox vs Stretch", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.93])  

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f"{output_dir}/letterbox_vs_stretch.png", dpi=150)
plt.show()

# === BƯỚC 5: XUẤT THÔNG SỐ TOÁN HỌC ===
print("=== THÔNG SỐ CHUYỂN ĐỔI LETTERBOX ===")
print(f"- Ảnh gốc: {w}x{h} (Tỷ lệ {w/h:.2f})")
print(f"- Hệ số tỷ lệ (Scale): {scale:.4f}")
print(f"- Đệm viền (Padding): Left={pad[0]}, Top={pad[1]}")
print(f"- Tọa độ Bbox trên Letterbox: {bbox_lb}")
print(f"- Tọa độ Bbox thực tế (Gốc):  ({x1_orig}, {y1_orig}, {x2_orig}, {y2_orig})")