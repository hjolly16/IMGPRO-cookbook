# IMGPRO Cookbook - Sổ tay Xử lý Ảnh

Chào mừng đến với **IMGPRO Cookbook**, một bộ các ví dụ, kỹ thuật và bài học thực hành về Xử lý Ảnh (Image Processing) sử dụng Python và OpenCV của cuốn sách cùng tên.

Dự án này được chia thành các bài học theo chủ đề từ cơ bản đến nâng cao, đi kèm với một dự án cuối khóa tổng hợp kiến thức.

## 📦 Yêu cầu

Dự án yêu cầu Python (>= 3.14) và các thư viện sau:
- `opencv-python`
- `numpy`
- `matplotlib`

Nên sử dụng uv cho toàn bộ dự án vì dự án này được build trên python thông qua uv

## 🚀 Cấu trúc nội dung

### [Lesson 0: Cơ bản / Nhập môn](lesson_0/)
Làm quen với việc đọc, hiển thị và lưu ảnh, các vấn đề về kiểu dữ liệu và pipeline xử lý cơ bản.
- `read_display_save_image.py`: Đọc và hiển thị ảnh.
- `datatype_overflow.py`: Xử lý tràn số trong ảnh.
- `simple_imgpro_pipeline.py`: Quy trình xử lý ảnh đơn giản.

### [Lesson 1: Biểu diễn ảnh & Màu sắc](lesson_1/)
Hiểu về cấu trúc ảnh số, không gian màu và các kỹ thuật cân bằng sáng.
- `color_spaces.py`: Các không gian màu (RGB, HSV, LAB...).
- `CLAHE.py`: Cân bằng histogram thích ứng (Contrast Limited Adaptive Histogram Equalization).
- `white_balance.py`: Cân bằng trắng.
- `image_compression.py`: Nén ảnh.

### [Lesson 2: Các phép toán trên điểm ảnh & Histogram](lesson_2/)
Các kỹ thuật biến đổi cường độ sáng, histogram và phân ngưỡng.
- `thresholding.py` & `adaptive_thresholding.py`: Phân ngưỡng nhị phân.
- `histogram_CDF.py`, `histogram_matching.py`: Phân tích và khớp histogram.
- `linear_tranform_gamma.py`, `logarit_tranform_mapping_function.py`: Biến đổi tuyến tính/phi tuyến, Gamma correction.
- `LUT.py`: Bảng tra (Look-Up Table).

### [Lesson 3: Lọc & Khử nhiễu (Filtering & Denoising)](lesson_3/)
Các kỹ thuật làm mịn ảnh, làm sắc nét và khử nhiễu.
- `manual_convolution.py`: Tích chập thủ công.
- `image_sharpening.py`: Làm sắc nét ảnh.
- `NLM.py`: Khử nhiễu Non-Local Means.
- `padding_compare.py`: Các kỹ thuật đệm (padding).

### [Lesson 4: Phát hiện biên & Đặc trưng (Edge & Feature Detection)](lesson_4/)
Tìm kiếm các cạnh và điểm đặc trưng trong ảnh.
- `canny.py`, `sobel_scharr.py`: Các thuật toán phát hiện biên.
- `corner_detection.py`, `FAST.py`: Phát hiện góc và điểm đặc trưng nhanh.
- `edge_quality_assessment.py`: Đánh giá chất lượng biên.

### [Lesson 5: Các phép toán hình thái học (Morphology)](lesson_5/)
Xử lý hình thái để lọc nhiễu, tách tách đối tượng.
- `erosion-dilation.py`: Co và giãn ảnh.
- `opening-closing.py`: Phép mở và đóng.
- `top-hat-black-hat.py`: Lọc chi tiết sáng/tối nhỏ.
- `morphological-gradient.py`: Gradient hình thái học.

### [Lesson 6: Biến đổi hình học (Geometric Transformations)](lesson_6/)
Thay đổi góc nhìn, kích thước và vị trí của ảnh.
- `affine-tranformation.py`: Biến đổi Affine.
- `perspective-tranfomation.py` & `homography.py`: Biến đổi phối cảnh.
- `interpolation.py`: Các kỹ thuật nội suy khi resize.

### [Lesson 7: Phân đoạn ảnh (Segmentation)](lesson_7/)
Tách đối tượng ra khỏi nền và phân vùng ảnh.
- `contour.py`: Tìm và vẽ đường bao.
- `watershed.py`: Thuật toán phân vùng Watershed.
- `grabcut.py`: Tách nền tương tác GrabCut.

### [Lesson 8: Miền tần số (Frequency Domain)](lesson_8/)
Xử lý ảnh dựa trên biến đổi Fourier.
- `frequency-domain.py`: Chuyển đổi sang miền tần số.
- `LPF.py`, `HPF-HFE.py`: Lọc thông thấp (làm mờ) và thông cao (làm nét).
- `notch.py`: Lọc nhiễu tuần hoàn.

### [Lesson 9: Khôi phục & Nâng cao (Restoration)](lesson_9/)
Các kỹ thuật xử lý ảnh nâng cao để khôi phục chất lượng.
- `inpainting.py`: Tái tạo vùng ảnh bị mất.
- `shadow-removal.py`: Loại bỏ bóng đổ.
- `wiener-filter.py`: Lọc Wiener để khôi phục ảnh mờ.
- `blur-map.py`: Bản đồ độ mờ.

### [Final Project: Ứng dụng thực tế](final/)
Tổng hợp kiến thức để xây dựng ứng dụng hoàn chỉnh.
- `document-scaner.py`: Máy quét tài liệu.
- `barcode-qr.py`: Đọc mã vạch và QR code.
- `enhanced.py`: Tăng cường chất lượng ảnh.

## 🛠 Cài đặt & Chạy thử

1.  **Clone repository:**
    ```bash
    git clone https://github.com/hjolly16/IMGPRO-cookbook.git
    cd IMGPRO_cookbook
    ```

2.  **Cài đặt môi trường:**
    Mỗi bài học có file `pyproject.toml` riêng, nhưng bạn có thể cài đặt các thư viện chung:
    ```bash
    cd lesson_0
    uv sync
    ```

3.  **Chạy ví dụ:**
    Di chuyển vào thư mục bài học và chạy script Python tương ứng. Ví dụ:
    ```bash
    cd lesson_0
    uv run examples/datatype_overflow.py
    ```

## 📝 Ghi chú
- Hãy đảm bảo bạn đã đặt các ảnh đầu vào vào thư mục `images/` tương ứng trong mỗi bài học nếu script yêu cầu.
- Thư mục `output/` sẽ chứa kết quả sau khi chạy script.

---
*Dự án này được biên soạn cho mục đích học tập và tham khảo.*
