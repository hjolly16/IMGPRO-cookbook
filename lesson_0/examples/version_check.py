import cv2
import numpy as np
import matplotlib
import sys

# Kiểm tra phiên bản
print(f"Python:     {sys.version}")
print(f"OpenCV:     {cv2.__version__}")
print(f"NumPy:      {np.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")

# Kiểm tra OpenCV hoạt động
# Tạo ảnh đen 100x100 pixel
test_img = np.zeros((100, 100, 3), dtype=np.uint8)
print("\nẢnh thử nghiệm:")
print(f"  Kích thước: {test_img.shape}")
print(f"  Kiểu dữ liệu: {test_img.dtype}")
print(f"  Giá trị min/max: {test_img.min()}/{test_img.max()}")
print("\n✓ Tất cả thư viện hoạt động bình thường!")
