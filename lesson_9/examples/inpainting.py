import cv2
import numpy as np
import os
import matplotlib

# Thiết lập backend TkAgg để hỗ trợ hiển thị cửa sổ trên Linux
try:
    matplotlib.use('TkAgg')
except:
    pass
import matplotlib.pyplot as plt

def create_base_texture(height, width):
    """Tạo ảnh nền cấu trúc để thử nghiệm khả năng nối nét của thuật toán."""
    image = np.full((height, width, 3), 240, dtype=np.uint8) # Nền xám nhạt
    
    # Vẽ các sọc màu ngang để kiểm tra tính liên tục của cấu trúc
    for i in range(0, height, 60):
        cv2.rectangle(image, (0, i), (width, i+25), (180, 120, 100), -1)
        
    # Vẽ đối tượng hình học để kiểm tra phục hồi đường cong
    cv2.circle(image, (width//2, height//2), 65, (60, 140, 60), -1)
    return image

def apply_damage_and_mask(image):
    """Giả lập các vết hỏng và tạo mặt nạ (Mask) tương ứng."""
    damaged = image.copy()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # 1. Vết xước dọc (cắt ngang các sọc ngang)
    cv2.line(damaged, (110, 20), (130, 280), (255, 255, 255), 6)
    cv2.line(mask, (110, 20), (130, 280), 255, 6)
    
    # 2. Chữ trắng (Watermark) đè lên vật thể trung tâm
    # Sửa lỗi: Sử dụng font chuẩn và tăng thickness để tạo độ đậm
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_pos = (130, 160)
    cv2.putText(damaged, "INPAINT", text_pos, font, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(mask, "INPAINT", text_pos, font, 1.2, 255, 3, cv2.LINE_AA)
    
    return damaged, mask

def run_inpainting_tutorial():
    h, w = 300, 400
    original = create_base_texture(h, w)
    damaged, raw_mask = apply_damage_and_mask(original)
    
    # Giãn nở Mask để đảm bảo thuật toán lấy đủ pixel biên sạch
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_final = cv2.dilate(raw_mask, kernel, iterations=1)
    
    # Thực hiện 2 kỹ thuật phục hồi phổ biến trong OpenCV
    # 1. Telea: Dựa trên phương pháp Fast Marching
    result_telea = cv2.inpaint(damaged, mask_final, 3, cv2.INPAINT_TELEA)
    # 2. NS: Dựa trên phương trình Navier-Stokes
    result_ns = cv2.inpaint(damaged, mask_final, 3, cv2.INPAINT_NS)
    
    # === TRỰC QUAN HÓA THEO LUỒNG XỬ LÝ ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    
    # HÀNG 1: TRẠNG THÁI ĐẦU VÀO
    axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("1. ẢNH GỐC (TARGET)", fontsize=13, fontweight='bold')
    
    axes[0, 1].imshow(cv2.cvtColor(damaged, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("2. ẢNH BỊ HỎNG (SOURCE)", fontsize=13, fontweight='bold', color='red')
    
    axes[0, 2].imshow(mask_final, cmap="gray")
    axes[0, 2].set_title("3. MẶT NẠ PHỤC HỒI (MASK)", fontsize=13, fontweight='bold')
    
    # HÀNG 2: KẾT QUẢ VÀ ĐÁNH GIÁ
    axes[1, 0].imshow(cv2.cvtColor(result_telea, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("4. KẾT QUẢ: THUẬT TOÁN TELEA", fontsize=13, fontweight='bold', color='green')
    
    axes[1, 1].imshow(cv2.cvtColor(result_ns, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("5. KẾT QUẢ: NAVIER-STOKES", fontsize=13, fontweight='bold', color='green')
    
    # Bản đồ sai số: Cho thấy vùng nào thuật toán 'đoán' chưa chuẩn
    error_map = cv2.absdiff(original, result_telea)
    axes[1, 2].imshow(cv2.cvtColor(error_map, cv2.COLOR_BGR2GRAY), cmap="hot")
    axes[1, 2].set_title("6. BẢN ĐỒ SAI SỐ (ERROR MAP)", fontsize=13, fontweight='bold')
    
    for ax in axes.ravel():
        ax.axis("off")
        
    plt.suptitle("PHÂN TÍCH SO SÁNH CÁC KỸ THUẬT PHỤC HỒI ẢNH", fontsize=18, y=0.98)
    plt.tight_layout()
    
    # Lưu kết quả
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/inpainting_tutorial_final.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    run_inpainting_tutorial()