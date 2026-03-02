import cv2
import numpy as np
import os
import matplotlib

# Thiết lập backend tương tác cho Linux
try:
    matplotlib.use('TkAgg')
except:
    pass
import matplotlib.pyplot as plt

def create_simulated_data(height, width):
    """Tạo dữ liệu mô phỏng với hiệu ứng Vignetting và nhiễu."""
    # Tạo ảnh lý tưởng với các đối tượng có độ sáng khác nhau
    ideal_image = np.zeros((height, width), dtype=np.float64)
    cv2.rectangle(ideal_image, (50, 50), (200, 150), 180, -1)
    cv2.rectangle(ideal_image, (300, 100), (500, 300), 200, -1)
    cv2.circle(ideal_image, (400, 250), 60, 160, -1)
    cv2.circle(ideal_image, (150, 300), 40, 220, -1)

    # Tạo mẫu chiếu sáng không đều (Vignetting pattern)
    y, x = np.mgrid[0:height, 0:width]
    center_y, center_x = height / 2, width / 2
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_distance = np.sqrt(center_x**2 + center_y**2)
    illumination_pattern = 1.0 - 0.6 * (distance / max_distance)**2

    # Tạo ảnh thực tế bằng cách nhân với mẫu chiếu sáng và thêm nhiễu
    raw_image = (ideal_image * illumination_pattern).astype(np.float64)
    raw_image += np.random.normal(0, 5, (height, width))
    raw_image = np.clip(raw_image, 0, 255).astype(np.uint8)

    return ideal_image, illumination_pattern, raw_image

def flat_field_correction(image, illumination_pattern):
    """
    Hiệu chỉnh ánh sáng bằng phương pháp Flat-field.
    Yêu cầu biết trước hoặc có ảnh tham chiếu của mẫu chiếu sáng.
    """
    # Mô phỏng ảnh flat thu được từ thực tế
    flat_frame = (255 * illumination_pattern).astype(np.float64)
    flat_frame += np.random.normal(0, 2, flat_frame.shape)
    flat_frame = np.clip(flat_frame, 1, 255)

    mean_value = np.mean(flat_frame)
    corrected = (image.astype(np.float64) / flat_frame) * mean_value
    return np.clip(corrected, 0, 255).astype(np.uint8)

def division_normalization(image, kernel_size=101):
    """
    Hiệu chỉnh ánh sáng bằng phương pháp chia chuẩn hóa (Division Normalization).
    Tự ước lượng nền bằng bộ lọc thông thấp (Gaussian Blur).
    """
    background_estimate = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0).astype(np.float64)
    background_estimate[background_estimate == 0] = 1 # Tránh lỗi chia cho 0
    
    mean_background = np.mean(background_estimate)
    corrected = (image.astype(np.float64) / background_estimate) * mean_background
    return np.clip(corrected, 0, 255).astype(np.uint8)

def top_hat_correction(image, kernel_size=51):
    """
    Sử dụng phép biến đổi Top-hat để trích xuất đối tượng sáng trên nền tối không đều.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

def visualize_results():
    h, w = 400, 600
    ideal, illumination, raw = create_simulated_data(h, w)

    # Thực hiện các phương pháp hiệu chỉnh
    result_ff = flat_field_correction(raw, illumination)
    result_div = division_normalization(raw)
    result_th = top_hat_correction(raw)

    # Hiển thị so sánh các phương pháp
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    data = [
        (ideal, "Ideal Image", "gray"),
        (illumination, "Illumination Pattern", "hot"),
        (raw, "Raw Image (Uneven)", "gray"),
        (result_ff, "Flat-field Correction", "gray"),
        (result_div, "Division Normalization", "gray"),
        (result_th, "Top-hat Transform", "gray")
    ]

    for ax, (img, title, cmap) in zip(axes.ravel(), data):
        ax.imshow(img, cmap=cmap, vmin=0, vmax=255 if cmap=="gray" else None)
        ax.set_title(title)
        ax.axis("off")

    plt.suptitle("Illumination Correction Techniques", fontsize=16)
    plt.tight_layout()
    
    # Lưu kết quả
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/illumination_analysis.png", dpi=150)
    
    # Hiển thị ảnh hưởng lên Thresholding
    _, thresh_raw = cv2.threshold(raw, 100, 255, cv2.THRESH_BINARY)
    _, thresh_corrected = cv2.threshold(result_div, 100, 255, cv2.THRESH_BINARY)
    
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    axes2[0].imshow(thresh_raw, cmap="gray")
    axes2[0].set_title("Thresholding: Raw (Incorrect at corners)")
    axes2[1].imshow(thresh_corrected, cmap="gray")
    axes2[1].set_title("Thresholding: Corrected (Accurate)")
    
    for ax in axes2: ax.axis("off")
    plt.savefig("output/threshold_comparison.png", dpi=150)
    
    plt.show()

if __name__ == "__main__":
    visualize_results()