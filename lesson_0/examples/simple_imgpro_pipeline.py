import cv2
import numpy as np
import os

# === CÁC HÀM XỬ LÝ (PIPELINE) ===

def doc_anh(duong_dan):
    """Bước 1: Đọc ảnh và kiểm tra lỗi đường dẫn."""
    if not os.path.exists(duong_dan):
        raise FileNotFoundError(f"Không tìm thấy ảnh tại: {duong_dan}")
        
    img = cv2.imread(duong_dan, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Lỗi: Không thể giải mã ảnh tại {duong_dan}")
        
    print(f"[Đầu vào] Kích thước: {img.shape}, Kiểu dữ liệu: {img.dtype}")
    return img

def xu_ly_xam(img):
    """Bước 2: Chuyển đổi sang ảnh xám (Grayscale)."""
    ket_qua = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"[Xử lý]   Đã chuyển sang ảnh xám. Kích thước mới: {ket_qua.shape}")
    return ket_qua

def do_luong_chat_luong(img_goc, img_xu_ly):
    """
    Bước 3: Đánh giá sai số giữa ảnh gốc và ảnh đã xử lý.
    Sử dụng MSE (Mean Squared Error) và PSNR (Peak Signal-to-Noise Ratio).
    """
    # Nếu ảnh gốc là màu, chuyển về xám để cùng số kênh (channel) với ảnh xử lý
    if len(img_goc.shape) == 3:
        goc_compare = cv2.cvtColor(img_goc, cv2.COLOR_BGR2GRAY)
    else:
        goc_compare = img_goc

    # Chuyển sang float64 để tính toán chính xác, tránh tràn số khi bình phương
    mse = np.mean((goc_compare.astype(np.float64) - img_xu_ly.astype(np.float64)) ** 2)

    # Tính PSNR (Đơn vị: dB). Giá trị càng cao, ảnh càng ít biến đổi.
    if mse == 0:
        psnr = float("inf")
    else:
        max_pixel = 255.0
        psnr = 10 * np.log10((max_pixel ** 2) / mse)

    print(f"[Đo lường] MSE: {mse:.4f}, PSNR: {psnr:.2f} dB")
    return {"mse": mse, "psnr": psnr}

def xuat_ket_qua(img, duong_dan):
    """Bước 4: Lưu ảnh đầu ra và đảm bảo thư mục tồn tại."""
    thu_muc = os.path.dirname(duong_dan)
    if thu_muc and not os.path.exists(thu_muc):
        os.makedirs(thu_muc)
        
    success = cv2.imwrite(duong_dan, img)
    if success:
        print(f"[Đầu ra]  Đã lưu ảnh thành công vào: {duong_dan}")
    else:
        print(f"[Lỗi]     Không thể ghi ảnh vào: {duong_dan}")

# === CHẠY QUY TRÌNH (MAIN FLOW) ===
if __name__ == "__main__":
    try:
        # Đường dẫn tệp
        input_path = "images/sample.jpg"
        output_path = "output/anh_ket_qua.png"

        # Thực thi pipeline
        anh_goc = doc_anh(input_path)
        anh_xam = xu_ly_xam(anh_goc)
        
        # Đo lường (Ở đây MSE sẽ = 0 vì phép chuyển đổi xám của OpenCV là nhất quán)
        chi_so = do_luong_chat_luong(anh_goc, anh_xam)

        # Lưu kết quả
        xuat_ket_qua(anh_xam, output_path)

        print("\n✓ Quy trình hoàn tất thành công!")
        
    except Exception as e:
        print(f"\n❌ Quy trình thất bại: {e}")