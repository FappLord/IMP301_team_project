# ĐỀ XUẤT ĐỒ ÁN CUỐI KỲ MÔN XỬ LÝ ẢNH & VIDEO (IMP302)

## 1. Tên Đề Tài (Project Title)
* **Tiếng Việt:** Nghiên cứu và Xây dựng Hệ thống Khôi phục, Phân đoạn và Nhận diện Biển số xe trong điều kiện suy giảm chất lượng ảnh (Image Degradation Restoration, Segmentation & Recognition Pipeline for License Plates).
* **Tiếng Anh:** *End-to-End Classical Image Processing Pipeline for Degraded License Plate Restoration, Segmentation, and Pattern Recognition.*

---

## 2. Lý do chọn đề tài & Điểm mạnh khi Bảo vệ (Why this topic?)
1. **Bám sát 100% giáo trình IMP302 (Không bị "lạc đề" sang thuần Deep Learning):**
   * Nhiều nhóm thường mắc lỗi chỉ lấy một model có sẵn (như YOLO/EasyOCR) về chạy, dẫn đến việc khi bảo vệ giảng viên hỏi về kiến thức nền tảng của môn học (biến đổi miền tần số, lọc khôi phục ảnh, phân đoạn ngưỡng, hình thái học, trích xuất đặc trưng cổ điển) thì không có trong bài làm.
   * Đề tài này kết nối **8/12 chương quan trọng nhất** của môn học thành một chuỗi xử lý (Pipeline) khép kín, có đầu vào – đầu ra và số liệu chứng minh ở từng bước.
2. **Tận dụng tối đa source code đã thực hành trên lớp:**
   * Đã có sẵn nền tảng code từ các bài lab trước: **Thuật toán Otsu & Region Labeling** (`slide_1`), **Khôi phục suy giảm ảnh & Biến đổi Wavelet DWT** (`slide_3`), **Nhận diện mẫu bằng biến đổi Walsh-Hadamard WHT / Hartley** (`slide_4`).
3. **Dễ chia việc cho các thành viên trong Team:**
   * Hệ thống được chia làm **4 Modules độc lập**, mỗi thành viên có thể phụ trách 1 Module rõ ràng (cả code lẫn viết phần báo cáo tương ứng).
4. **Chạy nhẹ, không cần GPU hay Dataset nặng hàng chục GB:**
   * Toàn bộ quá trình chạy thực nghiệm và huấn luyện bộ phân loại chỉ mất vài chục giây trên CPU thông thường, rất thuận tiện để chạy live demo khi bảo vệ.

---

## 3. Kiến trúc Hệ thống (4-Stage Pipeline)

Hệ thống nhận đầu vào là ảnh chụp phương tiện giao thông bị suy giảm chất lượng (mờ do xe di chuyển, nhiễu hạt do thiếu sáng, độ tương phản thấp, góc chụp nghiêng) và trả về chuỗi ký tự biển số xe cùng các chỉ số đánh giá.

```text
[Ảnh đầu vào bị mờ / nhiễu / tối]
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ MODULE 1: KHÔI PHỤC ẢNH & TĂNG CƯỜNG CHẤT LƯỢNG (Slide 1, 2, 3, 8)      │
│ - Khử mờ chuyển động (Motion Deblurring): Wiener Deconvolution (FFT)    │
│ - Khử nhiễu (Denoising): Wavelet DWT Thresholding, Median / Bilateral   │
│ - Tăng tương phản: CLAHE & Cân bằng Histogram                           │
│ => Đánh giá: Đo lường sự cải thiện qua MSE, PSNR, SSIM                  │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ MODULE 2: PHÁT HIỆN VÙNG BIỂN SỐ & HIỆU CHỈNH HÌNH HỌC (Slide 2, 7, 9, 11)│
│ - Dò biên dọc (Sobel-X / Canny) khai thác mật độ cạnh cao của biển số   │
│ - Phép toán hình thái học (Morphological Top-Hat, Black-Hat, Closing)   │
│ - Lọc vùng ứng viên theo đặc trưng hình học (Aspect Ratio, Area, Extent)│
│ - Căn chỉnh góc nghiêng (Affine / Perspective Deskewing)                │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ MODULE 3: PHÂN ĐOẠN & TÁCH KÝ TỰ (Slide 1, 9)                           │
│ - Nhị phân hóa: So sánh Global Otsu vs. Adaptive Thresholding           │
│ - Lọc nhiễu hình thái học (Opening/Closing xóa ốc vít và viền khung)    │
│ - Đánh nhãn vùng liên thông: Connected Component Labeling (8-connected) │
│ - Lọc Bounding Box ký tự & sắp xếp thứ tự dòng trên / dòng dưới         │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ MODULE 4: TRÍCH XUẤT ĐẶC TRƯNG & PHÂN LOẠI KÝ TỰ (Slide 4, 10, 12)      │
│ - Miền biến đổi: Hệ số tần số thấp Walsh-Hadamard Transform (WHT)       │
│ - Đặc trưng biên & hình thái: HOG, Hu Moments, Số lỗ Euler (Topology)   │
│ - Bộ phân loại: So sánh Minimum Distance, k-NN, và SVM / Random Forest  │
│ => Đánh giá: Accuracy, Precision, Recall, F1-Score, Confusion Matrix    │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
[Kết quả: Biển số "51A-123.45" + Báo cáo trực quan từng bước]
```

---

## 4. Chi tiết Công việc & Mapping Kiến thức theo từng Module

### Module 1: Khôi phục ảnh suy giảm & Tăng cường độ tương phản
* **File Notebook:** `01_image_restoration_enhancement.ipynb`
* **Kiến thức môn học áp dụng:**
  * *Slide 1 & 2 (Basic Gray-Level & Image Processing):* Biến đổi mức xám, Histogram Equalization, CLAHE, lọc không gian (Gaussian, Median, Bilateral).
  * *Slide 3 (Image Degradation, Restoration and Reconstruction):* Mô hình suy giảm ảnh $g(x,y) = h(x,y)*f(x,y) + \eta(x,y)$, khôi phục bằng lọc **Wiener** trên miền tần số, khử nhiễu đa phân giải bằng **Discrete Wavelet Transform (DWT)**.
  * *Slide 8 (Image Rendering and Assessment):* Đánh giá chất lượng ảnh khách quan bằng **MSE, PSNR, SSIM**.
* **Đầu ra cần có cho Report:**
  * Hình ảnh trực quan: Ảnh gốc $\rightarrow$ Ảnh bị suy giảm (Nhiễu + Mờ) $\rightarrow$ Ảnh sau khi khôi phục bằng từng phương pháp.
  * Biểu đồ cột / Bảng số liệu so sánh PSNR và SSIM giữa các bộ lọc.

### Module 2: Phát hiện vùng Biển số & Căn chỉnh góc nghiêng
* **File Notebook:** `02_license_plate_detection.ipynb`
* **Kiến thức môn học áp dụng:**
  * *Slide 7 (Image Acquisition & Geometric Transforms):* Biến đổi hình học Affine / Perspective để xoay thẳng biển số bị nghiêng.
  * *Slide 9 & 11 (Segmentation & Object Detection):* Phát hiện biên Sobel/Canny, phép toán hình thái học (Top-Hat làm nổi vùng sáng, Closing nối liền các nét chữ thành khối chữ nhật), lọc đối tượng theo tỷ lệ khung hình (Aspect Ratio) và độ đặc (Solidity).
* **Đầu ra cần có cho Report:**
  * Hình ảnh từng bước trung gian: Ảnh xám $\rightarrow$ Ảnh biên dọc Sobel $\rightarrow$ Ảnh sau Morphology Closing $\rightarrow$ Các Bounding Box ứng viên $\rightarrow$ Khung biển số đúng được cắt ra và làm thẳng.

### Module 3: Nhị phân hóa & Phân đoạn từng Ký tự
* **File Notebook:** `03_character_segmentation.ipynb`
* **Kiến thức môn học áp dụng:**
  * *Slide 1 & 9 (Thresholding & Region Labeling):* Thuật toán tìm ngưỡng tự động **Otsu** (tối đa hóa phương sai giữa 2 lớp $\sigma_B^2$), ngưỡng thích nghi cục bộ (**Adaptive Thresholding**) xử lý bóng đổ, và thuật toán **Connected Component Labeling (Region Labeling)** 8 láng giềng.
* **Đầu ra cần có cho Report:**
  * So sánh kết quả tách chữ giữa Otsu toàn cục và Adaptive Thresholding khi biển số bị chói sáng/bóng râm.
  * Bản đồ màu (Color-coded Label Map) của các vùng liên thông và kết quả cắt rời từng chữ cái chuẩn hóa về kích thước $32 \times 32$.

### Module 4: Trích xuất Đặc trưng & Nhận diện Ký tự (Pattern Classification)
* **File Notebook:** `04_feature_extraction_recognition.ipynb`
* **Kiến thức môn học áp dụng:**
  * *Slide 4 (Image Analysis & Transforms):* Biến đổi **Walsh-Hadamard (WHT)** trích xuất ma trận đặc trưng tần số thấp của ký tự.
  * *Slide 10 (Feature Extraction):* Đặc trưng **HOG (Histogram of Oriented Gradients)**, **7 bất biến Hu Moments**, và đặc trưng hình thái học/tô-pô (Euler Number - đếm số lỗ kín trong ký tự như `8`, `B` có 2 lỗ; `0`, `A`, `D`, `P` có 1 lỗ; `1`, `2`, `3`, `7` có 0 lỗ).
  * *Slide 12 (Image Pattern Classification):* So sánh các mô hình phân loại mẫu: **Minimum Distance Classifier (Template Matching)**, **k-NN**, và **SVM**.
* **Đầu ra cần có cho Report:**
  * Bảng so sánh độ chính xác (Accuracy, F1-score) và thời gian chạy giữa các bộ đặc trưng (Chỉ WHT vs. Chỉ HOG vs. Kết hợp WHT + HOG + Hình học).
  * Ma trận nhầm lẫn (**Confusion Matrix**) phân tích các cặp ký tự dễ nhầm (như `0` và `D`, `8` và `B`, `1` và `I`).

---

## 5. Các Thí nghiệm Đắt giá trong Báo cáo (Ablation Studies)
Để bài báo cáo đạt điểm tối đa, nhóm sẽ trình bày **3 bảng thực nghiệm chứng minh**:
1. **Tác động của Khôi phục ảnh (Module 1) lên toàn hệ thống:** Chứng minh khi ảnh bị mờ/nhiễu nặng, nếu **không có Module 1** thì tỷ lệ tách và nhận diện đúng cả biển số chỉ đạt mức thấp, nhưng **khi bật Module 1 (Wiener + DWT + CLAHE)** thì cả PSNR lẫn độ chính xác nhận diện đều tăng vọt.
2. **Đánh giá phương pháp Nhị phân hóa (Module 3):** So sánh Global Otsu vs. Adaptive Thresholding trong điều kiện ánh sáng đều và ánh sáng không đều.
3. **Đánh giá Đặc trưng & Bộ phân loại (Module 4):** So sánh hiệu quả của WHT (Chương 4) vs. HOG/Hu Moments (Chương 10) trên các thuật toán k-NN, Minimum Distance và SVM (Chương 12).

---

## 6. Cấu trúc Thư mục Dự kiến (`project/`)

```text
project/
├── README.md                                  # Tài liệu mô tả đề tài & hướng dẫn chạy
├── data/                                      # Tập ảnh biển số thực tế & tập mẫu ký tự (A-Z, 0-9)
├── src/                                       # Các hàm xử lý dùng chung (pipeline modules)
│   ├── restoration.py                         # Wiener, Wavelet DWT, CLAHE, PSNR/SSIM
│   ├── detection.py                           # Sobel, Morphology, ROI extraction, Deskew
│   ├── segmentation.py                        # Otsu, Adaptive Threshold, Region Labeling
│   └── recognition.py                         # WHT, HOG, Hu Moments, k-NN / SVM classifiers
├── notebooks/                                 # 4 Notebooks thực nghiệm & trực quan hóa
│   ├── 01_image_restoration_enhancement.ipynb
│   ├── 02_license_plate_detection.ipynb
│   ├── 03_character_segmentation.ipynb
│   ├── 04_feature_extraction_recognition.ipynb
│   └── 05_end_to_end_demo.ipynb               # Notebook chạy tổng hợp toàn bộ Pipeline
├── outputs/                                   # Hình ảnh minh họa & bảng biểu xuất ra cho Report
└── report/                                    # Nội dung bài báo cáo & slide thuyết trình
```

---

## 7. Đề xuất Phân công Công việc (Teamwork Allocation)

| Thành viên | Phụ trách chính | Công việc cụ thể (Code + Viết Report) |
| :--- | :--- | :--- |
| **Thành viên 1** | **Module 1 (Chương 1, 2, 3, 8)** | Xây dựng mô hình gây nhiễu/mờ, khôi phục ảnh bằng Wiener Filter, Wavelet DWT, CLAHE; tính PSNR/SSIM; viết Chương Cơ sở lý thuyết & Thực nghiệm Khôi phục ảnh. |
| **Thành viên 2** | **Module 2 (Chương 2, 7, 9, 11)** | Phát hiện vùng biển số bằng Sobel + Hình thái học (Morphology) + lọc hình học; căn chỉnh góc nghiêng; viết phần Phát hiện đối tượng (Object Detection). |
| **Thành viên 3** | **Module 3 (Chương 1, 9)** | Nhị phân hóa Otsu / Adaptive Thresholding, lọc nhiễu nhị phân, thuật toán Region Labeling (8-connectivity) tách từng ký tự; viết phần Phân đoạn ảnh (Segmentation). |
| **Thành viên 4** | **Module 4 (Chương 4, 10, 12) & Tích hợp** | Trích xuất đặc trưng WHT, HOG, Hu Moments/Euler; huấn luyện & đánh giá k-NN, SVM; ghép nối pipeline End-to-End và tổng hợp kết quả. |

*(Lưu ý: Nếu nhóm có 3 hoặc 5 người, có thể gộp Module 2 & 3 hoặc tách riêng phần Thực nghiệm End-to-End & Viết Report/Slide cho thành viên thứ 5).*

