import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import qrcode

# Duong dan goc tinh tu vi tri file nay
_BASE = Path(__file__).parent.parent
IMG_DIR = _BASE / "images"
OUT_DIR = _BASE / "output"

OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

# ===================================================================
# C.3.1 — Các bước tiền xử lý
# ===================================================================

def barcode_denoise(gray):
    """Khử nhiễu giữ biên sắc."""
    return cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)


def barcode_sharpen(gray):
    """Tăng nét — critical cho barcode mờ."""
    # Unsharp masking
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    sharpened = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    return sharpened


def barcode_normalize(gray, block=101):
    """Chuẩn hoá ánh sáng cục bộ."""
    bg = cv2.GaussianBlur(gray, (block, block), 0).astype(np.float64)
    bg[bg == 0] = 1
    norm = gray.astype(np.float64) / bg * 128
    return np.clip(norm, 0, 255).astype(np.uint8)


def barcode_threshold(gray, method="adaptive"):
    """Nhị phân hoá barcode."""
    if method == "adaptive":
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, blockSize=31, C=10
        )
    elif method == "otsu":
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    elif method == "sauvola":
        # Sauvola thresholding — tốt cho barcode
        mean = cv2.GaussianBlur(gray.astype(np.float64), (31, 31), 0)
        sq_mean = cv2.GaussianBlur(
            (gray.astype(np.float64)) ** 2, (31, 31), 0
        )
        std = np.sqrt(np.maximum(sq_mean - mean ** 2, 0))
        k = 0.2
        R = 128
        threshold = mean * (1 + k * (std / R - 1))
        binary = ((gray > threshold) * 255).astype(np.uint8)
    else:
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    return binary


def barcode_clahe(gray, clip=2.0, tile=8):
    """CLAHE — cân bằng histogram cục bộ, rất hiệu quả cho QR bị chênh sáng."""
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return clahe.apply(gray)


def barcode_gamma(gray, gamma=0.5):
    """Gamma correction — kéo sáng vùng quá tối do gradient ánh sáng."""
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8
    )
    return cv2.LUT(gray, table)


def barcode_morphology(binary, code_type="barcode"):
    """Morphology cho barcode vs QR."""
    if code_type == "barcode":
        # Barcode: closing ngang để nối các bar bị đứt
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
        # Opening nhỏ để loại nhiễu
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel_open)

    elif code_type == "qr":
        # Chỉ opening 3x3 rất nhẹ để loại đốm đơn pixel,
        # KHÔNG dùng kernel lớn hơn vì sẽ làm hỏng finder pattern
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    else:
        result = binary

    return result


def _try_decode_qr(img_gray):
    """Thử decode QR bằng nhiều detector, nhiều scale, cả ảnh đảo."""
    detectors = [cv2.QRCodeDetector()]
    try:
        detectors.append(cv2.QRCodeDetectorAruco())
    except AttributeError:
        pass  # OpenCV cũ không có QRCodeDetectorAruco

    for scale in [1, 2, 4]:
        if scale == 1:
            scaled = img_gray
        else:
            scaled = cv2.resize(
                img_gray, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_CUBIC
            )
        for det in detectors:
            for img_try in [scaled, cv2.bitwise_not(scaled)]:
                try:
                    data, _, _ = det.detectAndDecode(img_try)
                    if data:
                        return data
                except Exception:
                    pass
    return ""


# ===================================================================
# C.3.0 — Tao anh test tong hop
# ===================================================================

def create_barcode_test_image(out_path: Path):
    """Tạo ảnh barcode 1-D tổng hợp để chạy thử."""
    h, w = 200, 600
    canvas = np.ones((h, w), dtype=np.uint8) * 240

    np.random.seed(7)
    bits = np.random.choice([0, 1], size=80, p=[0.45, 0.55])
    x = 60
    for bit in bits:
        bar_w = np.random.choice([2, 4, 6])
        if bit == 0:
            canvas[:, x:x + bar_w] = 0
        x += bar_w
        if x >= w - 60:
            break

    noise = np.random.normal(0, 8, (h, w))
    canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    grad = np.linspace(0.75, 1.0, h)[:, None]
    canvas = np.clip(canvas * grad, 0, 255).astype(np.uint8)

    img_bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(out_path), img_bgr)
    print(f"Da tao anh barcode test tai: {out_path}")


def create_qr_test_image(out_path: Path):
    """Tạo ảnh QR thật có chứa đường dẫn, sau đó làm hỏng ảnh cực mạnh."""
    # 1. Tạo QR code thật
    qr = qrcode.QRCode(version=1, box_size=10, border=3)
    qr.add_data("https://talab.s4h.edu.vn/")
    qr.make(fit=True)
    img_pil = qr.make_image(fill_color="black", back_color="white")
    canvas = np.array(img_pil).astype(np.uint8) * 255
    h, w = canvas.shape

    # 2. Cố tình làm hỏng ảnh: blur nhẹ, gradient sáng chéo, noise, giảm contrast
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)

    grad = np.outer(np.linspace(0.1, 1.0, h), np.linspace(0.1, 1.0, w))
    canvas = np.clip(canvas.astype(np.float32) * grad, 0, 255)

    noise = np.random.normal(0, 5, canvas.shape)
    canvas = np.clip(canvas + noise, 0, 255).astype(np.uint8)

    canvas = cv2.addWeighted(canvas, 0.7, np.ones_like(canvas) * 125, 0.3, 0)

    img_bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(out_path), img_bgr)
    print(f"Da tao anh QR test KHÓ TÀN BẠO nát bét tai: {out_path}")


# ===================================================================
# C.3.2 — Pipeline voi nhieu cau hinh thu
# ===================================================================

def barcode_preprocess_pipeline(img_path, code_type="qr"):
    """Pipeline tien xu ly barcode/QR — thu nhieu cau hinh."""
    img_path = Path(img_path)
    if not img_path.exists():
        print(f"Khong tim thay anh, tao anh test tai: {img_path}")
        if code_type == "qr":
            create_qr_test_image(img_path)
        else:
            create_barcode_test_image(img_path)

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Loi doc anh: {img_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Thử decode trước khi tiền xử lý
    if code_type == "qr":
        data = _try_decode_qr(gray)
        if data:
            print(f"Decode thành công KHÔNG CẦN tiền xử lý: {data}")
            return gray

    # ── Cấu hình pipeline theo loại code ──────────────────────────────────────
    if code_type == "qr":
        configs = [
            {
                "gamma": None,  "clahe": False, "sharpen": False,
                "normalize": False, "threshold": "otsu",  "scale": 1,
                "name": "Otsu cơ bản",
            },
            {
                "gamma": 0.5,  "clahe": True,  "sharpen": False,
                "normalize": False, "threshold": "otsu",  "scale": 1,
                "name": "Gamma + CLAHE + Otsu",
            },
            {
                "gamma": 0.5,  "clahe": True,  "sharpen": True,
                "normalize": False, "threshold": "adaptive", "scale": 1,
                "name": "Gamma + CLAHE + Adaptive",
            },
            {
                "gamma": 0.5,  "clahe": True,  "sharpen": True,
                "normalize": False, "threshold": "sauvola", "scale": 2,
                "name": "Gamma+CLAHE+Sauvola ×2",
            },
        ]
    else:
        configs = [
            {"gamma": None, "clahe": False, "sharpen": False, "normalize": False,
             "threshold": "otsu",     "scale": 1, "name": "Otsu đơn giản"},
            {"gamma": None, "clahe": False, "sharpen": True,  "normalize": True,
             "threshold": "adaptive", "scale": 1, "name": "Sharpen + Normalize + Adaptive"},
            {"gamma": None, "clahe": False, "sharpen": True,  "normalize": True,
             "threshold": "sauvola",  "scale": 1, "name": "Sharpen + Normalize + Sauvola"},
            {"gamma": None, "clahe": False, "sharpen": False, "normalize": True,
             "threshold": "adaptive", "scale": 1, "name": "Normalize + Adaptive"},
        ]

    results = []

    # --- Bước trung gian dùng cho visualize ---
    _s0 = gray.copy()
    _s1 = barcode_denoise(_s0)
    if code_type == "qr":
        _s2 = barcode_gamma(_s1, gamma=0.5)
        _s3 = barcode_clahe(_s2)
        _s4 = barcode_sharpen(_s3)
        _s5 = barcode_threshold(_s4, method="adaptive")
        pipeline_steps = [
            (_s0, "① Ảnh gốc\n(noise + gradient sáng)"),
            (_s1, "② Bilateral Denoise\n(khử noise, giữ biên)"),
            (_s2, "③ Gamma 0.5\n(kéo sáng vùng tối)"),
            (_s3, "④ CLAHE\n(cân bằng sáng cục bộ)"),
            (_s4, "⑤ Unsharp Sharpen\n(tăng nét module)"),
            (_s5, "⑥ Adaptive Threshold\n(nhị phân hóa)"),
        ]
    else:
        _s2 = barcode_sharpen(_s1)
        _s3 = barcode_normalize(_s2)
        _s4 = barcode_threshold(_s3, method="adaptive")
        _s5 = barcode_morphology(_s4, code_type)
        pipeline_steps = [
            (_s0, "① Ảnh gốc\n(noise + gradient sáng)"),
            (_s1, "② Bilateral Denoise\n(khử noise, giữ biên)"),
            (_s2, "③ Unsharp Sharpen\n(tăng nét bar mờ)"),
            (_s3, "④ Division Normalize\n(cân bằng ánh sáng)"),
            (_s4, "⑤ Adaptive Threshold\n(nhị phân hoá)"),
            (_s5, "⑥ Morphology Close\n(nối bar đứt)"),
        ]

    for cfg in configs:
        processed = gray.copy()
        # Upscale nếu yêu cầu (giúp detector nhận diện QR module nhỏ)
        scale = cfg.get("scale", 1)
        if scale > 1:
            processed = cv2.resize(
                processed, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_CUBIC
            )
        processed = barcode_denoise(processed)
        if cfg.get("gamma"):
            processed = barcode_gamma(processed, gamma=cfg["gamma"])
        if cfg.get("clahe"):
            processed = barcode_clahe(processed)
        if cfg.get("sharpen"):
            processed = barcode_sharpen(processed)
        if cfg.get("normalize"):
            processed = barcode_normalize(processed)
        binary = barcode_threshold(processed, method=cfg["threshold"])
        binary = barcode_morphology(binary, code_type)
        results.append({"name": cfg["name"], "image": binary})
        if code_type == "qr":
            # Thử decode trên binary + ảnh đảo + scale khác nhau
            data = _try_decode_qr(binary)
            if data:
                print(f"✅ Decode thành công với [{cfg['name']}]: {data}")
                results[-1]["decoded"] = True
                results[-1]["data"]   = data
            else:
                print(f"❌ Decode THẤT BẠI với [{cfg['name']}]")
                results[-1]["decoded"] = False

    # ── Vẽ figure 2 hàng ──────────────────────────────────────────
    n_steps = len(pipeline_steps)
    n_cfgs  = len(results)

    fig = plt.figure(figsize=(n_steps * 3.2, 10))
    fig.patch.set_facecolor("#f8f9fa")

    # ── Hàng 1: từng bước pipeline ────────────────────────────────
    for i, (img_step, title) in enumerate(pipeline_steps):
        ax = fig.add_subplot(2, n_steps, i + 1)
        ax.imshow(img_step, cmap="gray", aspect="auto")
        ax.set_title(title, fontsize=8.5, fontweight="bold", pad=4)
        ax.axis("off")
        # Mũi tên nối các bước
        if i < n_steps - 1:
            fig.text(
                (i + 1) / n_steps - 0.005, 0.72,
                "→", fontsize=16, ha="center", va="center", color="#555"
            )

    # Nhãn hàng 1
    fig.text(0.01, 0.75, "PIPELINE\nTỪNG BƯỚC",
             fontsize=9, fontweight="bold", va="center",
             rotation=90, color="#2c3e50")

    # ── Hàng 2: so sánh 4 config + intensity profile ──────────────
    # 4 config chiếm n_cfgs cột, cột cuối là intensity profile
    n_cols_row2 = n_cfgs + 1

    # Tính profile trên dòng giữa ảnh gốc
    mid_row = gray.shape[0] // 2
    profile_orig = gray[mid_row, :].astype(np.float32)

    for i, r in enumerate(results):
        left  = (i / n_cols_row2) * 0.88 + 0.06
        ax = fig.add_axes([left, 0.06, 0.88 / n_cols_row2 - 0.01, 0.32])

        # Resize ảnh kết quả về kích thước gốc nếu đã upscale
        disp_img = r["image"]
        if disp_img.shape != gray.shape:
            disp_img = cv2.resize(
                disp_img, (gray.shape[1], gray.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        ax.imshow(disp_img, cmap="gray", aspect="auto")
        decoded = r.get("decoded")
        if decoded:
            data_str = r.get("data", "")
            short = (data_str[:22] + "…") if len(data_str) > 25 else data_str
            status_sym = f"✓ {short}"
        else:
            status_sym = "✗ Không decode được"
        status_col = "#27ae60" if decoded else "#e74c3c"
        border_col = "#27ae60" if decoded else "#e74c3c"
        for spine in ax.spines.values():
            spine.set_edgecolor(border_col)
            spine.set_linewidth(2.5)
        ax.set_title(f"{r['name']}", fontsize=8, fontweight="bold", pad=3)
        ax.set_xlabel(status_sym, fontsize=7.5, color=status_col, fontweight="bold")
        ax.tick_params(left=False, bottom=False,
                       labelleft=False, labelbottom=False)

    # Intensity profile cột cuối
    ax_prof = fig.add_axes([0.06 + 0.88 * n_cfgs / n_cols_row2 + 0.005,
                             0.06, 0.88 / n_cols_row2 - 0.015, 0.32])
    ax_prof.plot(profile_orig, color="#3498db", lw=1.2, label="Gốc (gray)")
    # Profile sau bước cân bằng sáng (CLAHE hoặc Normalize)
    mid_label = "Sau Gamma+CLAHE" if code_type == "qr" else "Sau Normalize"
    ax_prof.plot(_s3[mid_row, :].astype(np.float32), color="#e67e22", lw=1.2, label=mid_label)
    # Profile binary cuối
    last_binary = _s5 if code_type == "qr" else _s4
    prof_bin = last_binary[mid_row, :].astype(np.float32)
    ax_prof.plot(prof_bin, color="#2ecc71", lw=1.0, alpha=0.7, label="Binary cuối")
    ax_prof.axhline(128, color="gray", lw=0.7, ls="--", label="Ngưỡng 128")
    ax_prof.set_title("Intensity Profile\n(dòng giữa ảnh)",
                       fontsize=8.5, fontweight="bold", pad=3)
    ax_prof.set_xlabel("pixel X", fontsize=7.5)
    ax_prof.set_ylabel("giá trị", fontsize=7.5)
    ax_prof.legend(fontsize=6.5, loc="upper right")
    ax_prof.tick_params(labelsize=7)
    ax_prof.set_facecolor("#f0f0f0")

    # Nhãn hàng 2
    fig.text(0.01, 0.22, f"SO SÁNH\n{n_cfgs} CONFIG",
             fontsize=9, fontweight="bold", va="center",
             rotation=90, color="#2c3e50")

    # Đường kẻ phân cách 2 hàng
    fig.add_artist(plt.Line2D([0.03, 0.97], [0.44, 0.44],
                               color="#bdc3c7", lw=1.2, transform=fig.transFigure))

    plt.suptitle("Barcode/QR — Preprocessing Pipeline", fontsize=13,
                 fontweight="bold", y=0.98)

    out_path = OUT_DIR / f"{code_type}_preprocess.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Da luu visualization: {out_path}")
    plt.show()

    return results


# === CHAY ===
if __name__ == "__main__":
    results = barcode_preprocess_pipeline(
        IMG_DIR / "qr.jpg", code_type="qr"
    )
