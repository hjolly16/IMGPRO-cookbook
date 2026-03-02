import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def danh_gia_bien(edges):
    """
    Đánh giá chất lượng biên dựa trên đặc điểm hình học:
    Mật độ, Độ liên tục và Độ mỏng.
    """
    # Đảm bảo ảnh ở dạng nhị phân (0 hoặc 1)
    binary_edges = (edges > 0).astype(np.uint8)
    so_pixel_bien = np.sum(binary_edges)
    tong_pixel = binary_edges.size

    # 1. Mật độ biên (Density)
    mat_do = (so_pixel_bien / tong_pixel) * 100

    # 2. Tính toán lân cận (Sử dụng kernel 3x3 để đếm số hàng xóm của mỗi pixel biên)
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    # neighbor_count[i,j] sẽ chứa số lượng pixel biên xung quanh pixel (i,j)
    neighbor_count = cv2.filter2D(binary_edges, -1, kernel)

    # 3. Độ liên tục (Continuity)
    # Tỉ lệ pixel biên có ít nhất 1 pixel biên khác đứng cạnh
    pixel_co_lan_can = np.sum((binary_edges == 1) & (neighbor_count > 0))
    do_lien_tuc = (pixel_co_lan_can / max(so_pixel_bien, 1)) * 100

    # 4. Độ mỏng (Thinness)
    # Canny chuẩn sẽ có độ mỏng cao (pixel biên thường chỉ có tối đa 2 lân cận: trước và sau)
    pixel_mong = np.sum((binary_edges == 1) & (neighbor_count <= 2))
    do_mong = (pixel_mong / max(so_pixel_bien, 1)) * 100

    return {
        "mat_do": mat_do,
        "do_lien_tuc": do_lien_tuc,
        "do_mong": do_mong,
        "so_pixel_bien": so_pixel_bien,
    }

# === CHƯƠNG TRÌNH CHÍNH ===
img_path = "images/sample.jpg"
if not os.path.exists(img_path):
    print("Lỗi: Không tìm thấy ảnh mẫu.")
    exit()

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img_blur = cv2.GaussianBlur(img, (5, 5), 1.0)

# Mảng lưu ảnh và tiêu đề để trực quan hóa
images_to_show = [img]
titles_to_show = ["Ảnh Gốc"]

# In tiêu đề bảng so sánh
print(f"{'Phương pháp':<25}{'Mật độ':>10}{'Liên tục':>12}{'Độ mỏng':>10}")
print("-" * 60)

# Thử nghiệm với Canny (các ngưỡng khác nhau)
for t_low, t_high in [(30, 60), (50, 150), (100, 200)]:
    edges = cv2.Canny(img_blur, t_low, t_high)
    res = danh_gia_bien(edges)
    label = f"Canny ({t_low}, {t_high})"
    print(f"{label:<25}{res['mat_do']:>9.2f}%{res['do_lien_tuc']:>11.1f}%{res['do_mong']:>9.1f}%")
    images_to_show.append(edges)
    titles_to_show.append(label)

# Thử nghiệm với Sobel (để thấy sự khác biệt về độ mỏng)
sobel_x = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)
magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
edges_sobel = (magnitude > 50).astype(np.uint8) * 255

res_s = danh_gia_bien(edges_sobel)
print(f"{'Sobel (T=50)':<25}{res_s['mat_do']:>9.2f}%{res_s['do_lien_tuc']:>11.1f}%{res_s['do_mong']:>9.1f}%")

images_to_show.append(edges_sobel)
titles_to_show.append("Sobel (T=50)")

print("\n> Ghi chú Tutorial:")
print("  - Mật độ quá cao (>20%): Ảnh bị nhiễu.")
print("  - Độ mỏng thấp (<80%): Biên bị dày, không chuẩn (như Sobel).")
print("  - Liên tục thấp (<70%): Biên bị đứt gãy quá nhiều.")

# === TRỰC QUAN HÓA VÀ LƯU KẾT QUẢ ===
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

fig, axes = plt.subplots(1, len(images_to_show), figsize=(18, 4))
for ax, image, title in zip(axes, images_to_show, titles_to_show):
    ax.imshow(image, cmap="gray")
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
out_path = os.path.join(output_dir, "edge_quality_assessment_result.jpg")
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"\nĐã lưu ảnh trực quan hóa tại: {out_path}")
plt.show()