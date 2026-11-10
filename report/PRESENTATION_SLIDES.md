# SLIDE TRÌNH CHIẾU BẢO VỆ ĐỒ ÁN IMP302 (IMAGE & VIDEO PROCESSING)

---

## SLIDE 1: TIÊU ĐỀ ĐỒ ÁN
- **Học phần:** Xử lý Ảnh và Video (`IMP302`)
- **Đề tài:** **Khôi Phục Ảnh Biển Số Xe Suy Giảm Đa Tác Nhân Kết Hợp Phân Đoạn và Nhận Dạng Ký Tự Bằng Phương Pháp Xử Lý Ảnh Cổ Điển**
- **Định hướng học thuật:**
  - **Chủ đề trọng tâm:** **Khôi phục & Tái tạo Ảnh Suy giảm (Chapter 3 & Chapter 8 trong `Doc/`)**
  - **Phương pháp:** 100% Xử lý ảnh cổ điển từ nguyên lý gốc (*First Principles*), bám sát 12 chương bài giảng `Doc/1..12.pdf`, không dùng pre-trained Deep Learning OCR.

---

## SLIDE 2: ĐẶT VẤN ĐỀ & TẠI SAO KHÔI PHỤC ẢNH LÀ CHỦ ĐỀ TRỌNG TÂM?
- Trong thực tế giám sát giao thông, biển số xe máy trong ảnh $472\times 303$ chỉ chiếm vùng rất nhỏ ($\approx 72\times 63\text{ px}$), khoảng hở giữa hai chữ số liền kề chỉ rộng **$3\sim 4\text{ pixels}$**.
- Khi gặp **nhòe chuyển động tuyến tính** ($L=7\text{ px}, \theta=12^\circ$) + **nhiễu cảm biến AWGN / Muối tiêu / Giao thoa tuần hoàn**:
  - Vệt nhòe làm các ký tự dính liền vào nhau và dính vào khung viền biển số.
  - Nếu đưa trực tiếp vào bộ phân đoạn & nhận dạng: **Tỷ lệ phân đoạn đúng sụp đổ từ 95.65% xuống 1.74% (0.00% nhận dạng đúng biển số)!**
- $\Rightarrow$ **Module 1 (Chapter 3 Image Restoration)** đóng vai trò quyết định sự sống còn của toàn bộ hệ thống.

---

## SLIDE 3: BẢNG ĐỐI CHIẾU KIẾN THỨC VỚI 12 CHƯƠNG BÀI GIẢNG (`Doc/1..12.pdf`)
- **Chapter 1 & 2 (`Doc/1, 2.pdf`):** Full-scale Histogram Stretch, Log Transform, Weighted Median Filter ($3\times 3$), Unsharp Masking / Highboost, Morphological Top-Hat / Black-Hat / Opening / Closing, Otsu's Method, 8-Connected Region Labeling, Bicubic Interpolation.
- **Chapter 3 (`Doc/3.pdf` — Trọng tâm):** Mô hình $g=h*f+\eta$, 6 hàm mật độ nhiễu PDFs (Gaussian, Rayleigh, Gamma, Uniform, Salt & Pepper, Periodic), Arithmetic/Geometric/Harmonic/Contraharmonic Mean, **Butterworth Notch Reject Filter**, **2D Wavelet Denoising**, **Inverse Filter**, **Wiener Filter**, **Constrained Least Squares (CLS) Filter**.
- **Chapter 4 (`Doc/4.pdf`):** **2D DHT** ($\text{cas}$), **2D DCT**, **Sequency-Ordered 2D WHT**, và **2D Haar DWT**.
- **Chapter 7, 8, 9 (`Doc/7, 8, 9.pdf`):** Nội suy lấy mẫu lại, chỉ số **MSE / SNR / PSNR / SSIM**, và phân cụm **K-Means ($K=2$)**.
- **Chapter 10, 11, 12 (`Doc/10, 11, 12.pdf`):** **Fourier Boundary Descriptors**, **Covariance Matrix Eccentricity**, **7 Hu Moments**, **Euler Number**, **HOG**, **PCA**, **IoU**, và 7 bộ phân loại (**Min-Dist, Correlation, Gaussian Bayes, MLP, k-NN, RF, SVM**).

---

## SLIDE 4: CHIA TÁCH TRAIN / VAL / TEST & MỞ RỘNG DỮ LIỆU GẤP $6\times$ (`data/splits/`)
- **Chia tách ở cấp độ biển số (Zero Character Leakage):**
  - **Train (70%):** 80 biển số gốc $\rightarrow$ **480 ảnh biển số mở rộng** ($3.973$ mẫu ký tự)
  - **Validation (15%):** 17 biển số gốc $\rightarrow$ **102 ảnh biển số mở rộng** ($767$ mẫu ký tự)
  - **Test (15%):** 18 biển số gốc $\rightarrow$ **108 ảnh biển số mở rộng** ($903$ mẫu ký tự)
- **6 Kịch bản mở rộng dữ liệu từ Chapter 3 (`115 × 6 = 690` ảnh):**
  1. `clean` (Ảnh thực tế gốc)
  2. `awgn` ($\sigma=10.0$, Ch. 3.2 P7)
  3. `salt_pepper` ($P=4\%$, Ch. 3.2 P12)
  4. `rayleigh` (Nhiễu lệch thiếu sáng, Ch. 3.2 P8)
  5. `periodic` (Sọc giao thoa hình sin $u_0=24, v_0=18$, Ch. 3.3 P18–22)
  6. `motion_blur` (Nhòe chuyển động $L=7, \theta=12^\circ + \text{AWGN }\sigma=7.0$, Ch. 3.4 & 3.6–3.7)

![Dataset Splits and Expansion](../outputs/fig10_dataset_splits_and_expansion.png)

---

## SLIDE 5: MODULE 1 (TRỌNG TÂM) — KHÔI PHỤC ĐA NHIỄU & GIẢI CHẬP WIENER / CLS
- **Khử nhiễu xung Muối Tiêu:** **Center-Weighted Median Filter** ($\mathbf{W}=\begin{bmatrix}1&2&1\\2&3&2\\1&2&1\end{bmatrix}$) nâng PSNR từ **19.22 dB $\rightarrow$ 37.20 dB** (SSIM **0.9875**), vượt xa Median chuẩn ($35.31\text{ dB}$).
- **Khử nhiễu Tuần hoàn:** **Butterworth Notch Reject Filter** trên miền 2D FFT triệt tiêu cặp đỉnh xung liên hợp $(\pm u_0, \pm v_0)$, nâng PSNR từ **23.75 dB $\rightarrow$ 38.42 dB** (+14.67 dB).
- **Khử Nhòe Chuyển động + AWGN:** Kết hợp **2D Haar Wavelet Shrinkage** + **Hybrid Wiener-CLS Deconvolution** ($\hat{F}(u,v) = \frac{(1+K)H^*(u,v)}{|H(u,v)|^2 + K + \gamma|P(u,v)|^2}G(u,v)$), nâng PSNR từ **25.42 dB $\rightarrow$ 28.34 dB** và tránh hoàn toàn hiện tượng nổ nhiễu của Inverse Filter ($21.69\text{ dB}$).

![Module 1 Restoration](../outputs/fig01_module1_restoration.png)

---

## SLIDE 6: MODULE 2 & 3 — PHÁT HIỆN BIỂN SỐ, ĐỐI SÁNH OTSU / K-MEANS / ADAPTIVE & CCL 8 HƯỚNG
- **Module 2 (Chapters 1, 2, 7, 11):** Black-Hat ($25\times 25$) + Sobel dọc $|G_x|$ + Closing/Opening + Xoay Affine **Bicubic** $\rightarrow$ **100% (115/115)** phát hiện đúng biển số.
- **Module 3 (Chapters 1.2 & 9.1):**
  - Cài đặt từ công thức gốc **Otsu's Method** ($\sigma_B^2(t)$), **K-Means Clustering ($K=2$)**, và **Local Adaptive Gaussian Thresholding**.
  - Thuật toán cắt cầu nối khung viền + **8-Connected Region Labeling** đạt **95.65% (110/115)** phân đoạn đúng tuyệt đối số ký tự (dưới bóng đổ mạnh: Adaptive đạt **94.78%**, K-Means **92.17%**, Otsu **91.30%**).

![Module 3 Segmentation](../outputs/fig03_module3_segmentation.png)

---

## SLIDE 7: MODULE 4 — ĐỐI SÁNH TRANSFORM (CH. 4), ĐẶC TRƯNG (CH. 10) & PHÂN LOẠI (CH. 12)
- **Đối sánh 3 phép biến đổi Chapter 4 (100 chiều, SVM 5-Fold CV):**
  - **2D WHT (Sequency-ordered, Ch. 4.4):** **95.33%** (nhanh nhất, chỉ dùng cộng/trừ)
  - **2D DCT (Ch. 4.3):** **95.23%**
  - **2D DHT (Hartley, Ch. 4.2):** **92.84%**
- **Kết hợp Đặc trưng (WHT + HOG + Hu/Covariance Eccentricity/Fourier Boundary/Euler) & PCA (Ch. 10.3):**
  - **Combined + PCA (80 chiều) + MLP (Ch. 12.4):** **96.99%** (5-Fold CV)
  - **Combined (480 chiều) + SVM:** **96.37%** (5-Fold CV) | **Gaussian Bayes (Ch. 12.3):** **94.40%**
- **Hiệu quả huấn luyện trên tập mở rộng $6\times$ (Kiểm định độc lập trên Held-Out Test Split):**
  - Huấn luyện SVM trên **480 biển mở rộng ($3.973$ ký tự)** giúp tăng độ chính xác trên tập Test mở rộng từ **92.36% lên 95.24% (+2.88%)**!

![Module 4 Evaluation](../outputs/fig05_module4_evaluation.png)

---

## SLIDE 8: KẾT QUẢ TOÀN TUYẾN & DEMO TRÊN TẬP KIỂM THỬ ĐỘC LẬP (`TEST SPLIT`)
- Trên tập kiểm thử độc lập (**Held-Out Test Split**), việc bật **Module 1 (Chapter 3 Restoration)** giúp khôi phục tỷ lệ nhận dạng toàn biển số:
  - **Nhòe chuyển động ($L=7, \sigma=7$):** từ **0.00% $\rightarrow$ 61.11%** (nhận dạng ký tự từ **3.85% $\rightarrow$ 93.59%**)
  - **Nhiễu Muối Tiêu ($4\%$):** từ **55.56% $\rightarrow$ 94.44%**
  - **Nhiễu Tuần hoàn (Sinusoidal):** từ **72.22% $\rightarrow$ 94.44%**
  - **Nhiễu Gaussian / Rayleigh:** từ **94.44% $\rightarrow$ 100.00%**

![End-to-End Demo](../outputs/fig07_end_to_end_demo.png)
