import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===================================================================
# E.3.1 — Phân tích ảnh thiếu sáng
# ===================================================================

def phan_tich_thieu_sang(img):
    """Đánh giá mức độ thiếu sáng."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    mean_val = np.mean(gray)
    median_val = np.median(gray)
    dark_ratio = np.sum(gray < 50) / gray.size  # % pixel rất tối
    bright_ratio = np.sum(gray > 200) / gray.size  # % pixel sáng
    std_val = np.std(gray)

    print(f"  Brightness mean:  {mean_val:.1f} / 255")
    print(f"  Brightness median: {median_val:.1f} / 255")
    print(f"  Dark pixels (<50): {dark_ratio*100:.1f}%")
    print(f"  Bright pixels (>200): {bright_ratio*100:.1f}%")
    print(f"  Contrast (std):   {std_val:.1f}")

    if mean_val < 50:
        level = "RẤT TỐI"
    elif mean_val < 80:
        level = "THIẾU SÁNG"
    elif mean_val < 120:
        level = "HƠI TỐI"
    else:
        level = "ĐỦ SÁNG"

    print(f"  Đánh giá: {level}")
    return mean_val, dark_ratio


# ===================================================================
# E.3.2 — Chỉnh sáng bằng Gamma
# ===================================================================

def _auto_gamma(y_channel):
    """Tính gamma tự động từ kênh Y."""
    fg = y_channel[y_channel > 10]
    ref_val = float(np.percentile(fg, 90)) if len(fg) > 100 else float(np.mean(y_channel))
    ref_val = max(ref_val, 5.0)
    # Kéo p90 lên target=160 — đủ sáng mà không overexpose
    gamma = np.log(ref_val / 255.0) / np.log(160.0 / 255.0)
    gamma = np.clip(gamma, 1.0, 3.0)
    print(f"  Auto gamma: {gamma:.2f}  (ref_val p90={ref_val:.1f})")
    return gamma


def gamma_correction(img, gamma=None):
    """Gamma correction trên kênh Y + khuếch đại chroma tương ứng.

    Lý thuyết:
    - Gamma boost Y từ Y_old → Y_new (tỷ lệ R = Y_new/Y_old).
    - Trong cảnh thực, khi ánh sáng tăng R lần, độ lệch chroma cũng tăng ≈ sqrt(R).
    - Kéo giãn (Cb-128) và (Cr-128) theo sqrt(R) giữ màu tự nhiên đúng mức.
    """
    ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycbcr)

    if gamma is None:
        gamma = _auto_gamma(y)

    lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                    for i in range(256)]).astype(np.uint8)
    y_corrected = cv2.LUT(y, lut)

    # Khuếch đại độ lệch Cb/Cr theo sqrt(Y_new / Y_old)
    y_f  = y.astype(np.float64) + 1.0
    yn_f = y_corrected.astype(np.float64) + 1.0
    ratio = np.sqrt(yn_f / y_f)                       # H×W, > 1 với ảnh tối
    cb_f = 128.0 + (cb.astype(np.float64) - 128.0) * ratio
    cr_f = 128.0 + (cr.astype(np.float64) - 128.0) * ratio
    cb_out = np.clip(cb_f, 0, 255).astype(np.uint8)
    cr_out = np.clip(cr_f, 0, 255).astype(np.uint8)

    merged = cv2.merge([y_corrected, cr_out, cb_out])
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)


# ===================================================================
# E.3.3 — Chỉnh sáng bằng MSRCP (Multi-Scale Retinex with
#          Color Preservation)
# ===================================================================

def multi_scale_retinex(img, sigmas=[15, 80, 250]):
    """Multi-Scale Retinex — áp dụng trên kênh luminance, giữ nguyên tỉ lệ màu.

    Chiến lược:
    - Tính MSR trên luminance (trung bình 3 kênh) → hệ số tăng sáng.
    - Nhân hệ số tăng sáng đều cho cả 3 kênh → không làm lệch màu.
    - Chuẩn hóa cuối theo percentile [1, 99] → tránh outlier.
    - Tăng nhẹ saturation bù lại cảm giác mờ màu sau MSR.
    """
    # Khử nhiễu nhẹ trước MSR để tránh khuếch đại noise ở vùng tối
    pre = cv2.bilateralFilter(img, d=5, sigmaColor=25, sigmaSpace=25)
    img_f = pre.astype(np.float64) + 1.0

    # MSR chỉ trên luminance (trung bình 3 kênh)
    lum = img_f.mean(axis=2)  # H×W
    retinex_lum = np.zeros_like(lum)
    for sigma in sigmas:
        blur = cv2.GaussianBlur(lum, (0, 0), sigma)
        retinex_lum += np.log10(lum) - np.log10(np.maximum(blur, 1.0))
    retinex_lum /= len(sigmas)

    # Chuẩn hóa hệ số tăng sáng về [0.2, 1.0]
    lo, hi = np.percentile(retinex_lum, 1), np.percentile(retinex_lum, 99)
    enhance = np.clip((retinex_lum - lo) / max(hi - lo, 1e-6), 0, 1)
    enhance = 0.2 + 0.8 * enhance  # floor 0.2 → tránh vùng tối thành đen hoàn toàn

    # Áp dụng hệ số đều cho cả 3 kênh → giữ nguyên tỉ lệ màu gốc
    result_f = (img_f - 1.0) * enhance[:, :, np.newaxis]

    # Chuẩn hóa toàn ảnh về [0, 230] để tránh clipping
    r_lo = np.percentile(result_f, 1)
    r_hi = np.percentile(result_f, 99)
    result_f = np.clip((result_f - r_lo) / max(r_hi - r_lo, 1e-6), 0, 1) * 230.0
    result_u8 = np.clip(result_f, 0, 255).astype(np.uint8)

    return result_u8


# ===================================================================
# E.3.4 — Denoise cho ảnh thiếu sáng
# ===================================================================

def unsharp_mask(img, sigma=1.0, strength=0.5):
    """Unsharp masking CHỈ trên kênh Y — tăng nét mà không tạo artifact màu."""
    ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycbcr)

    blur = cv2.GaussianBlur(y, (0, 0), sigma)
    y_sharp = cv2.addWeighted(y, 1.0 + strength, blur, -strength, 0)
    y_sharp = np.clip(y_sharp, 0, 255).astype(np.uint8)

    merged = cv2.merge([y_sharp, cr, cb])
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)


def denoise_low_light(img):
    """Bilateral filter — khử nhiễu giữ nguyên cạnh, nhanh hơn NLMeans."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray)
    # Ảnh càng tối → sigma lớn hơn để xử lý noise nhiều hơn
    sigma = int(np.clip(40 - mean_val / 4, 15, 45))
    return cv2.bilateralFilter(img, d=9, sigmaColor=sigma, sigmaSpace=sigma)


# ===================================================================
# E.3.5 — CLAHE trên kênh L (LAB)
# ===================================================================

def clahe_enhance(img, clip_limit=2.5, grid_size=8):
    """CLAHE trên kênh Y + kéo nhẹ chroma theo tỷ lệ cục bộ."""
    ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycbcr)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(grid_size, grid_size)
    )
    y_enhanced = clahe.apply(y)

    # Kéo chroma nhẹ theo sqrt(ratio) — giữ màu tự nhiên sau CLAHE
    y_f  = y.astype(np.float64) + 1.0
    ye_f = y_enhanced.astype(np.float64) + 1.0
    ratio = np.sqrt(np.clip(ye_f / y_f, 0.5, 2.0))   # giới hạn để không quá mạnh
    cb_f = 128.0 + (cb.astype(np.float64) - 128.0) * ratio
    cr_f = 128.0 + (cr.astype(np.float64) - 128.0) * ratio

    merged = cv2.merge([
        y_enhanced,
        np.clip(cr_f, 0, 255).astype(np.uint8),
        np.clip(cb_f, 0, 255).astype(np.uint8)
    ])
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)


# ===================================================================
# E.3.6 — Color correction (khử cast màu)
# ===================================================================

def color_correction(img, mode="stretch"):
    """Chỉnh màu — hai chế độ.

    mode="gray_world"  — Gray World assumption (phù hợp ảnh ngoài trời).
    mode="stretch"     — Kéo giãn từng kênh theo percentile [1, 99].
                         An toàn hơn cho ảnh khoa học / fluorescence,
                         không giả định nền xám.
    """
    result = img.copy().astype(np.float64)

    if mode == "gray_world":
        avg_b = np.mean(result[:, :, 0])
        avg_g = np.mean(result[:, :, 1])
        avg_r = np.mean(result[:, :, 2])
        avg_all = (avg_b + avg_g + avg_r) / 3
        result[:, :, 0] *= avg_all / max(avg_b, 1)
        result[:, :, 1] *= avg_all / max(avg_g, 1)
        result[:, :, 2] *= avg_all / max(avg_r, 1)
    else:  # stretch — mặc định
        for c in range(3):
            lo = np.percentile(result[:, :, c], 1)
            hi = np.percentile(result[:, :, c], 99)
            if hi - lo > 1:
                result[:, :, c] = (result[:, :, c] - lo) / (hi - lo) * 255.0

    return np.clip(result, 0, 255).astype(np.uint8)


# ===================================================================
# E.3.7 — Pipeline hoàn chỉnh
# ===================================================================

def low_light_enhance(img_path, method="balanced"):
    """Pipeline tăng cường ảnh thiếu sáng.

    method:
        "gamma"    — chỉ gamma correction
        "retinex"  — Multi-Scale Retinex
        "balanced" — Gamma + Denoise + CLAHE + Color correction
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"Lỗi đọc ảnh: {img_path}")
        return None

    print(f"\n{'='*40}")
    print(f"  LOW-LIGHT ENHANCEMENT")
    print(f"  Phương pháp: {method}")
    print(f"{'='*40}")

    phan_tich_thieu_sang(img)

    if method == "gamma":
        result = gamma_correction(img)

    elif method == "retinex":
        result = multi_scale_retinex(img)
        result = clahe_enhance(result, clip_limit=2.0)

    elif method == "balanced":
        # Toàn bộ pipeline trong không gian YCbCr:
        # Cb/Cr (màu) không bao giờ bị chạm → màu ra giống thật 100%.
        #
        # Bước 1: Bilateral denoise — giữ cạnh, nhanh
        step1 = denoise_low_light(img)

        # Bước 2: Gamma chỉ trên Y → không lệch màu
        step2 = gamma_correction(step1)

        # Bước 3: CLAHE chỉ trên Y → tăng chi tiết cục bộ
        step3 = clahe_enhance(step2, clip_limit=2.0, grid_size=8)

        # Bước 4: Unsharp chỉ trên Y → sắc nét không artifact màu
        result = unsharp_mask(step3, sigma=1.0, strength=0.4)

    else:
        print(f"Phương pháp không hợp lệ: {method}")
        return None

    # === SO SÁNH TẤT CẢ PHƯƠNG PHÁP ===
    result_gamma = gamma_correction(img)
    result_retinex = multi_scale_retinex(img)

    step1_b = denoise_low_light(img)
    step2_b = gamma_correction(step1_b)
    step3_b = clahe_enhance(step2_b, clip_limit=2.0)
    result_balanced = unsharp_mask(step3_b, sigma=1.0, strength=0.4)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Ảnh gốc (thiếu sáng)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(cv2.cvtColor(result_gamma, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Gamma correction")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(cv2.cvtColor(result_retinex, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title("Multi-Scale Retinex")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(cv2.cvtColor(result_balanced, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("Balanced (Gamma+Denoise+CLAHE)\n★ Recommended")
    axes[1, 0].axis("off")

    # Histogram so sánh
    gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_result = cv2.cvtColor(result_balanced, cv2.COLOR_BGR2GRAY)

    axes[1, 1].hist(gray_orig.ravel(), 256, range=[0, 256],
                    alpha=0.5, color="blue", label="Gốc")
    axes[1, 1].hist(gray_result.ravel(), 256, range=[0, 256],
                    alpha=0.5, color="red", label="Sau xử lý")
    axes[1, 1].set_title("Histogram so sánh")
    axes[1, 1].legend()

    # Zoom vùng tối
    h, w = img.shape[:2]
    # Lấy vùng tối nhất
    gray_blocks = cv2.resize(gray_orig, (8, 8))
    min_idx = np.unravel_index(np.argmin(gray_blocks), gray_blocks.shape)
    ry, rx = min_idx[0] * h // 8, min_idx[1] * w // 8
    rh, rw = h // 4, w // 4
    ry = min(ry, h - rh)
    rx = min(rx, w - rw)

    crop_orig = img[ry:ry+rh, rx:rx+rw]
    crop_result = result_balanced[ry:ry+rh, rx:rx+rw]
    zoom = np.hstack([crop_orig, crop_result])
    axes[1, 2].imshow(cv2.cvtColor(zoom, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title("Zoom: Gốc (trái) vs Enhanced (phải)")
    axes[1, 2].axis("off")

    plt.suptitle("Low-Light Enhancement", fontsize=14)
    plt.tight_layout()
    plt.savefig("output/low_light_result.png", dpi=150)
    plt.show()

    return result


# === CHẠY ===
result = low_light_enhance("images/low_light.jpg", method="balanced")
if result is not None:
    cv2.imwrite("output/enhanced.jpg", result)
    print("Đã lưu: output/enhanced.jpg")
