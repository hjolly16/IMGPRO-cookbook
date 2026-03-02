import cv2
import numpy as np
import os
import matplotlib

# Thiết lập backend TkAgg để hỗ trợ hiển thị cửa sổ trên hệ thống Linux
try:
    matplotlib.use('TkAgg')
except:
    pass
import matplotlib.pyplot as plt

def evaluate_visual_restoration(original_image, damaged_image, restored_image, title="Restoration Quality"):
    """
    Dashboard so sánh trực quan quy trình phục hồi ảnh:
    Hàng 1: Trước (Hỏng) vs Sau (Phục hồi)
    Hàng 2: Đối chứng (Gốc) vs Bản đồ sai số
    """
    # 1. Tính toán chỉ số PSNR
    psnr_val = cv2.PSNR(original_image, restored_image)
    
    # 2. Tạo bản đồ sai lệch
    gray_orig = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY) if original_image.ndim == 3 else original_image
    gray_rest = cv2.cvtColor(restored_image, cv2.COLOR_BGR2GRAY) if restored_image.ndim == 3 else restored_image
    diff_map = cv2.absdiff(gray_orig, gray_rest)
    heatmap = cv2.applyColorMap(diff_map, cv2.COLORMAP_JET)

    # 3. Tạo Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    axes[0, 0].imshow(cv2.cvtColor(damaged_image, cv2.COLOR_BGR2RGB) if damaged_image.ndim == 3 else damaged_image, cmap='gray')
    axes[0, 0].set_title("1. TRƯỚC: ẢNH BỊ HỎNG (Input)", fontsize=14, fontweight='bold', color='red', pad=15)
    
    axes[0, 1].imshow(cv2.cvtColor(restored_image, cv2.COLOR_BGR2RGB) if restored_image.ndim == 3 else restored_image, cmap='gray')
    axes[0, 1].set_title(f"2. SAU: ẢNH PHỤC HỒI (Output)\nPSNR: {psnr_val:.2f} dB",
                         fontsize=14, fontweight='bold', color='green', pad=15)
    
    axes[1, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB) if original_image.ndim == 3 else original_image, cmap='gray')
    axes[1, 0].set_title("3. ĐỐI CHỨNG: ẢNH GỐC (Ground Truth)", fontsize=14, fontweight='bold', pad=15)
    
    axes[1, 1].imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("4. BẢN ĐỒ SAI SỐ (Đỏ = Vùng lỗi)", fontsize=14, fontweight='bold', pad=15)

    for ax in axes.ravel():
        ax.axis("off")

    # Tiêu đề tổng
    plt.suptitle(f"Hệ thống Đánh giá Chất lượng Phục hồi: {title}",
                 fontsize=18, fontweight='bold', y=0.98)

    plt.subplots_adjust(
        top=0.92,      # khoảng cách với suptitle
        bottom=0.05,   # khoảng trống phía dưới
        hspace=0.25,   # khoảng cách giữa hàng
        wspace=0.15    # khoảng cách giữa cột
    )


    os.makedirs("output", exist_ok=True)
    plt.savefig("output/quality_check_dashboard.png", dpi=150, bbox_inches="tight")
    plt.show()

# --- CHẠY DEMO THỰC TẾ ---
if __name__ == "__main__":
    # 1. Tạo ảnh gốc với cấu trúc hình học rõ nét
    h, w = 300, 400
    original = np.zeros((h, w, 3), dtype=np.uint8)
    # Nền caro xám
    for i in range(0, h, 40):
        cv2.line(original, (0, i), (w, i), (100, 100, 100), 1)
    # Vật thể màu
    cv2.circle(original, (200, 150), 70, (50, 50, 200), -1)
    cv2.rectangle(original, (50, 50), (150, 150), (50, 200, 50), -1)

    # 2. Tạo vết hỏng (Chữ trắng đè lên cấu trúc)
    damaged = original.copy()
    mask = np.zeros((h, w), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(damaged, "DAMAGED", (80, 170), font, 1.5, (255, 255, 255), 5, cv2.LINE_AA)
    cv2.putText(mask, "DAMAGED", (80, 170), font, 1.5, 255, 5, cv2.LINE_AA)

    # 3. Thực hiện phục hồi bằng thuật toán Telea
    restored = cv2.inpaint(damaged, mask, 3, cv2.INPAINT_TELEA)

    # 4. Hiển thị Dashboard hoàn chỉnh
    evaluate_visual_restoration(original, damaged, restored, "Khôi phục chữ đè (Watermark Removal)")