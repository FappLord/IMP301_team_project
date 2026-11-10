# Đồ Án Cuối Kỳ Môn Xử Lý Ảnh và Video (IMP302)
## Khôi Phục Ảnh Biển Số Xe Suy Giảm Đa Tác Nhân Kết Hợp Phân Đoạn và Nhận Dạng Ký Tự Bằng Xử Lý Ảnh Cổ Điển

Dự án này cài đặt hoàn chỉnh hệ thống **Khôi phục, Phát hiện, Phân đoạn và Nhận dạng Biển số Xe máy Việt Nam** bằng **100% các thuật toán xử lý ảnh cổ điển từ nguyên lý toán học gốc (*Classical Image Processing from First Principles*)**, bám sát toàn bộ 12 chương bài giảng trong thư mục [`Doc/`](file:///c:/PhatVo_Important/Codes/My_Codes/Academy/IMP302/project/Doc) (`1..12.pdf`) và đề cương [`PROJECT_PROPOSAL_TEAM.md`](file:///c:/PhatVo_Important/Codes/My_Codes/Academy/IMP302/project/PROJECT_PROPOSAL_TEAM.md).

---

## Điểm Nổi Bật Của Đồ Án

1. **Chủ đề Trọng tâm — Khôi phục Ảnh Biển số Suy giảm (Chapter 3 & Chapter 8 trong `Doc/`):**
   - Mô phỏng đầy đủ **6 mô hình suy giảm vật lý của Chapter 3**: Nhiễu trắng Gaussian (`awgn`), Nhiễu xung Muối Tiêu (`salt_pepper`), Nhiễu lệch thiếu sáng Rayleigh/Gamma (`rayleigh`), Nhiễu sọc giao thoa tuần hoàn (`periodic`), và Nhòe chuyển động tuyến tính kèm nhiễu (`motion_blur`).
   - Triển khai đầy đủ các bộ lọc khôi phục không gian & tần số: **Arithmetic / Geometric / Harmonic / Contraharmonic Mean Filters** (Ch. 3.3), **Center-Weighted Median Filter** (Ch. 2.2), **Butterworth Notch Reject Filter** (Ch. 3.3), **2D Haar Wavelet Shrinkage** (Ch. 3.5 & Ch. 4.6), **Inverse Filter** (Ch. 3.8), **Wiener Deconvolution** (Ch. 3.6), và **Constrained Least Squares (CLS) Filtering** với toán tử Laplacian $\mathbf{P}(u,v)$ (Ch. 3.7).
   - Đánh giá khách quan bằng đầy đủ 4 chỉ số của **Chapter 8**: **MSE, SNR, PSNR, SSIM** và độ chính xác nhận dạng đầu cuối.

2. **Chia Tách Tập Dữ Liệu Chuẩn (Train / Validation / Test) & Mở Rộng Đa Suy Giảm Gấp $6\times$:**
   - Chia tách phân tầng ở **cấp độ biển số (Zero Character Leakage)** theo tỷ lệ **70% Train (80 biển) / 15% Validation (17 biển) / 15% Test (18 biển)** tại [`data/splits/dataset_splits.json`](file:///c:/PhatVo_Important/Codes/My_Codes/Academy/IMP302/project/data/splits/dataset_splits.json).
   - Mở rộng gấp **$6\times$** qua 6 kịch bản suy giảm của Chapter 3 thành **$690$ ảnh phương tiện** ($480$ Train / $102$ Val / $108$ Test) và **$5.643$ mẫu ký tự $32\times 32$** ($3.973$ Train / $767$ Val / $903$ Test).

3. **Đối Sánh Toàn Diện Các Thuật Toán Phân Đoạn, Biến Đổi, Đặc Trưng & Phân Loại Mẫu (`Doc/1, 4, 9, 10, 11, 12.pdf`):**
   - **Phân đoạn (Ch. 1.2 & Ch. 9.1):** Cài đặt từ công thức gốc và đối sánh **Global Otsu's Method**, **K-Means Clustering ($K=2$)**, và **Local Adaptive Gaussian Thresholding** + **8-Connected Region Labeling** (đạt **95.65%** phân đoạn đúng tuyệt đối số ký tự).
   - **Biến đổi & Đặc trưng (Ch. 4 & Ch. 10):** Đối sánh **2D WHT (Sequency-ordered)**, **2D DCT**, **2D DHT (Hartley)**, **Boundary Fourier Descriptors**, **Covariance Matrix Eccentricity**, **7 Hu Moment Invariants**, **Euler Number $E=C-H$**, **HOG**, và giảm chiều **PCA**.
   - **Phân loại mẫu (Ch. 11 & Ch. 12):** Đối sánh 7 bộ phân loại (**Minimum-Distance**, **Correlation Coefficient**, **Statistical Gaussian Bayes**, **Multi-Layer Perceptron MLP**, **k-NN**, **Random Forest**, **RBF-SVM**), đạt **96.99%** (5-Fold CV với Combined+PCA+MLP) và **95.24%** trên tập kiểm thử mở rộng độc lập (`Held-Out Test 6x`).

---

## Hướng Dẫn Chạy Nhanh

```bash
# 1. Chạy live demo trên tập Held-Out Test Split (kiểm thử cả ảnh sạch và khôi phục đa suy giảm Chapter 3):
python demo.py

# 2. Chạy toàn bộ thực nghiệm, chia tập Train/Val/Test, mở rộng 6x và xuất 10 biểu đồ vào outputs/:
python generate_all_experiments.py

# 3. Đóng gói lại 5 Jupyter Notebooks thực thi hoàn chỉnh trong notebooks/:
python build_notebooks.py
```

---

## Danh Mục Tài Liệu & Báo Cáo Đầu Ra

- **Báo cáo khoa học đầy đủ (Markdown):** [`report/FINAL_REPORT.md`](file:///c:/PhatVo_Important/Codes/My_Codes/Academy/IMP302/project/report/FINAL_REPORT.md)
- **Báo cáo khoa học định dạng Word (.docx):** [`report/FINAL_REPORT_IMP302.docx`](file:///c:/PhatVo_Important/Codes/My_Codes/Academy/IMP302/project/report/FINAL_REPORT_IMP302.docx)
- **Slide thuyết trình bảo vệ đồ án:** [`report/PRESENTATION_SLIDES.md`](file:///c:/PhatVo_Important/Codes/My_Codes/Academy/IMP302/project/report/PRESENTATION_SLIDES.md)
- **Số liệu thực nghiệm tổng hợp:** [`outputs/metrics_summary.json`](file:///c:/PhatVo_Important/Codes/My_Codes/Academy/IMP302/project/outputs/metrics_summary.json)
