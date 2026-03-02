import cv2
import numpy as np
import os
import matplotlib

# Thiết lập backend tương tác
try:
    matplotlib.use('TkAgg')
except:
    pass
import matplotlib.pyplot as plt

def create_shadow_simulation(height, width):
    """Tạo bề mặt vật liệu giả lập có khuyết tật và bóng đổ."""
    # Tạo bề mặt đồng nhất
    surface = np.ones((height, width), dtype=np.float64) * 200

    # Thêm các khuyết tật (Defects): chấm tối và vết xước
    cv2.circle(surface, (150, 120), 15, 80, -1)
    cv2.circle(surface, (400, 250), 10, 90, -1)
    cv2.line(surface, (250, 50), (280, 350), 70, 3)

    # Mô phỏng bóng đổ (Gradient tối ở góc dưới-phải)
    y, x = np.mgrid[0:height, 0:width]
    shadow_field = 1.0 - 0.5 * (x / width) * (y / height)
    
    # Ảnh thu được bao gồm bóng và nhiễu
    shadowed_image = (surface * shadow_field).astype(np.float64)
    shadowed_image += np.random.normal(0, 3, (height, width))
    shadowed_image = np.clip(shadowed_image, 0, 255).astype(np.uint8)

    return surface, shadowed_image

def process_shadow_removal(image, kernel_size=101):
    """Loại bỏ bóng đổ bằng phép chia chuẩn hóa (Division Normalization)."""
    # Ước lượng trường bóng (Shading estimation)
    shading_estimate = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0).astype(np.float64)
    shading_estimate[shading_estimate == 0] = 1 # Tránh chia cho 0
    
    mean_intensity = np.mean(shading_estimate)
    normalized = (image.astype(np.float64) / shading_estimate) * mean_intensity
    return np.clip(normalized, 0, 255).astype(np.uint8)

def detect_defects(image, method="adaptive"):
    """Phân đoạn để tìm kiếm khuyết tật."""
    if method == "otsu":
        # Otsu truyền thống (Global Threshold)
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        # Ngưỡng thích nghi (Adaptive Threshold)
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, blockSize=31, C=10
        )
        # Cleanup morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return binary

def visualize_inspection():
    h, w = 400, 600
    ground_truth_surface, shadowed_img = create_shadow_simulation(h, w)

    # Thực hiện xử lý
    normalized_img = process_shadow_removal(shadowed_img)
    
    # So sánh các kết quả phân đoạn
    binary_truth = detect_defects(ground_truth_surface.astype(np.uint8), method="otsu")
    binary_before = detect_defects(shadowed_img, method="otsu")
    binary_after = detect_defects(normalized_img, method="adaptive")

    # Hiển thị
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    axes[0, 0].imshow(ground_truth_surface, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Original Surface (Clean)")
    
    axes[0, 1].imshow(shadowed_img, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("Image with Shadow")
    
    axes[0, 2].imshow(normalized_img, cmap="gray", vmin=0, vmax=255)
    axes[0, 2].set_title("Normalized (Shadow Removed)")
    
    axes[1, 0].imshow(binary_truth, cmap="gray")
    axes[1, 0].set_title("Ground Truth Segmentation")
    
    axes[1, 1].imshow(binary_before, cmap="gray")
    axes[1, 1].set_title("Otsu Before Correction\n(False Positives in Shadow)")
    
    axes[1, 2].imshow(binary_after, cmap="gray")
    axes[1, 2].set_title("Adaptive After Correction\n(Accurate Defect Detection)")

    for ax in axes.ravel(): ax.axis("off")
    
    plt.suptitle("Shadow Removal for Industrial Quality Control", fontsize=16)
    plt.tight_layout()
    
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/shadow_inspection_analysis.png", dpi=150)
    
    # Tính toán định lượng
    fp_before = np.sum(binary_before[ground_truth_surface > 150] == 255)
    fp_after = np.sum(binary_after[ground_truth_surface > 150] == 255)
    reduction = (1 - fp_after / max(fp_before, 1)) * 100

    print(f"False Positives (Pixel count):")
    print(f"  - Before: {fp_before}")
    print(f"  - After:  {fp_after}")
    print(f"  - Error Reduction: {reduction:.2f}%")
    
    plt.show()

if __name__ == "__main__":
    visualize_inspection()