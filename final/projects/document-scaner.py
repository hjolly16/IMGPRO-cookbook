import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Đường dẫn gốc tính từ vị trí file này
_BASE = Path(__file__).parent.parent
IMG_DIR = _BASE / "images"
OUT_DIR = _BASE / "output"

# Tạo thư mục output nếu chưa có
OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

# ===================================================================
# A.3.1 — Tiền xử lý
# ===================================================================

def resize_for_detection(img, max_dim=800):
    """Resize để tăng tốc + giữ tỉ lệ."""
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1.0:
        img_small = cv2.resize(img, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
    else:
        img_small = img.copy()
        scale = 1.0
    return img_small, scale


# ===================================================================
# A.3.2 — Phát hiện mép tài liệu
# ===================================================================

def sort_four_points(pts):
    """Sắp xếp 4 điểm theo thứ tự: top-left, top-right,
    bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()

    rect[0] = pts[np.argmin(s)]   # top-left: x+y nhỏ nhất
    rect[2] = pts[np.argmax(s)]   # bottom-right: x+y lớn nhất
    rect[1] = pts[np.argmin(d)]   # top-right: x-y nhỏ nhất
    rect[3] = pts[np.argmax(d)]   # bottom-left: x-y lớn nhất
    return rect


def find_document_contour(img):
    """Tìm contour 4 đỉnh lớn nhất — mép tài liệu."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge
    edges = cv2.Canny(blur, 50, 150)

    # Dilate để nối biên đứt
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Tìm contour
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Sắp xếp theo diện tích giảm dần
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    doc_contour = None
    for cnt in contours[:5]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4 and cv2.contourArea(approx) > 1000:
            doc_contour = approx.reshape(4, 2).astype(np.float32)
            break

    if doc_contour is None:
        print("Cảnh báo: không tìm thấy mép tài liệu — dùng toàn ảnh.")
        h, w = img.shape[:2]
        doc_contour = np.array([
            [0, 0], [w, 0], [w, h], [0, h]
        ], dtype=np.float32)

    return sort_four_points(doc_contour)


# ===================================================================
# A.3.3 — Perspective warp
# ===================================================================

def perspective_warp(img, pts_src):
    """Biến đổi perspective: 4 điểm nguồn → hình chữ nhật."""
    tl, tr, br, bl = pts_src

    # Tính kích thước output
    width_top = np.linalg.norm(tr - tl)
    width_bot = np.linalg.norm(br - bl)
    width = int(max(width_top, width_bot))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    height = int(max(height_left, height_right))

    pts_dst = np.array([
        [0, 0], [width - 1, 0],
        [width - 1, height - 1], [0, height - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(img, M, (width, height))
    return warped


# ===================================================================
# A.3.4 — Deskew
# ===================================================================

def compute_skew_angle(img_gray):
    """Tính góc nghiêng bằng Hough lines."""
    edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=100, minLineLength=100, maxLineGap=10
    )

    if lines is None or len(lines) == 0:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Chỉ lấy góc gần ngang (±30°)
        if abs(angle) < 30:
            angles.append(angle)

    if len(angles) == 0:
        return 0.0

    return np.median(angles)


def deskew(img):
    """Xoay ảnh để chữ nằm ngang."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    angle = compute_skew_angle(gray)

    if abs(angle) < 0.5:
        return img  # Không cần xoay

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Tính kích thước mới để không cắt mất góc
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    rotated = cv2.warpAffine(img, M, (new_w, new_h),
                             borderValue=(255, 255, 255))
    return rotated


# ===================================================================
# A.3.5 — Binarize + Enhance
# ===================================================================

def binarize_enhance(img):
    """Nhị phân hoá adaptive + tăng cường."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Denoise nhẹ
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # Adaptive threshold
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=21, C=10
    )

    # Morphology: loại nhiễu nhỏ
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return binary


# ===================================================================
# A.3.6 — Pipeline hoàn chỉnh
# ===================================================================

def create_test_image(out_path: Path):
    """Tạo ảnh tài liệu tổng hợp để chạy thử khi không có ảnh thật."""
    h, w = 600, 420
    canvas = np.ones((800, 600, 3), dtype=np.uint8) * 180  # nền xám

    # Document (trắng, hơi nghiêng)
    pts = np.array([
        [90, 80], [510, 60],
        [530, 720], [70, 740]
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [pts], (255, 255, 255))

    # Vài dòng chữ giả lập
    for i, y in enumerate(range(140, 680, 55)):
        bar_w = np.random.randint(280, 380)
        cv2.rectangle(canvas, (110, y), (110 + bar_w, y + 18), (60, 60, 60), -1)

    cv2.imwrite(str(out_path), canvas)
    print(f"Đã tạo ảnh test tại: {out_path}")


def document_scanner(img_path, output_path=None):
    """Pipeline quét tài liệu hoàn chỉnh."""
    img_path = Path(img_path)
    if not img_path.exists():
        print(f"Không tìm thấy ảnh, tạo ảnh test tại: {img_path}")
        create_test_image(img_path)

    img_orig = cv2.imread(str(img_path))
    if img_orig is None:
        print(f"Lỗi: không đọc được {img_path}")
        return None

    print(f"Ảnh gốc: {img_orig.shape[1]}x{img_orig.shape[0]}")

    # 1. Tiền xử lý (resize để phát hiện mép nhanh)
    img_small, scale = resize_for_detection(img_orig)

    # 2. Phát hiện mép trên ảnh nhỏ
    pts_small = find_document_contour(img_small)

    # Chuyển tọa độ về ảnh gốc
    pts_orig = pts_small / scale

    # 3. Perspective warp trên ảnh gốc (giữ chất lượng)
    warped = perspective_warp(img_orig, pts_orig)
    print(f"Sau warp: {warped.shape[1]}x{warped.shape[0]}")

    # 4. Deskew
    deskewed = deskew(warped)

    # 5. Binarize + Enhance
    result = binarize_enhance(deskewed)

    # Hiển thị
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))

    axes[0].imshow(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
    axes[0].set_title("1. Ảnh gốc")
    axes[0].axis("off")

    # Vẽ contour phát hiện
    img_contour = img_small.copy()
    pts_int = pts_small.astype(int).reshape(-1, 1, 2)  # (4,1,2) cho cv2.polylines
    cv2.polylines(img_contour, [pts_int], True, (0, 255, 0), 3)
    axes[1].imshow(cv2.cvtColor(img_contour, cv2.COLOR_BGR2RGB))
    axes[1].set_title("2. Phát hiện mép")
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    axes[2].set_title("3. Perspective warp")
    axes[2].axis("off")

    axes[3].imshow(result, cmap="gray")
    axes[3].set_title("4. Binarize + Enhance")
    axes[3].axis("off")

    plt.suptitle("Document Scanner Pipeline", fontsize=14)
    plt.tight_layout()
    vis_path = OUT_DIR / "doc_scanner_result.png"
    plt.savefig(str(vis_path), dpi=150)
    print(f"Đã lưu visualization: {vis_path}")
    plt.show()

    if output_path:
        cv2.imwrite(str(output_path), result)
        print(f"Đã lưu kết quả: {output_path}")

    return result


# === CHẠY ===
if __name__ == "__main__":
    result = document_scanner(
        IMG_DIR / "document_photo.jpg",
        OUT_DIR / "document_scanned.png",
    )
