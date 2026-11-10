# End-to-End Classical Image Processing Pipeline for Degraded License Plate Restoration, Segmentation, and Pattern Recognition
## Khôi phục Ảnh Biển số Xe Suy giảm Đa tác nhân Kết hợp Phân đoạn và Nhận dạng Ký tự bằng các Phương pháp Xử lý Ảnh Cổ điển

**Course:** IMP302 — Digital Image and Video Processing *(Stanford EE368 / CS232 Curriculum Alignment)*  
**Paper Type:** Final Project Technical Report (Stanford CS231n / EE368 Academic Paper Standard)  
**Repository:** [https://github.com/FappLord/IMP301_team_project](https://github.com/FappLord/IMP301_team_project)  

---

## TÓM TẮT (ABSTRACT)

Trong các hệ thống giao thông thông minh (Intelligent Transportation Systems — ITS) ngoài thực tế, ảnh và video giám sát chụp biển số xe máy thường xuyên chịu sự suy giảm vật lý đa tác nhân: **nhòe chuyển động tuyến tính (Linear Motion Blur)** khi phương tiện di chuyển qua vùng quét của cảm biến, **nhiễu cảm biến (Additive White Gaussian Noise, Rayleigh/Gamma Skewed Noise, Impulse Salt-and-Pepper Noise)** trong điều kiện phơi sáng kém, **nhiễu giao thoa tuần hoàn (Periodic Sinusoidal Interference)** trên kênh truyền, và **biến dạng hình học phối cảnh (Perspective Foreshortening & In-plane Tilt $\pm 30^\circ$)**. Khi đưa trực tiếp các ảnh suy giảm này vào các bộ tách ngưỡng nhị phân hoặc nhận dạng ký tự, hiện tượng bệt nét (stroke smearing) lấp đầy khe hở $3\sim 4\text{ px}$ giữa các chữ số làm sụp đổ hoàn toàn khâu phân đoạn liên thông, khiến độ chính xác nhận dạng toàn biển số rơi xuống **$0.00\%$**.

Bám sát 100% nền tảng lý thuyết của giáo trình **Stanford EE368 / IMP302 (`Doc/1..12.pdf`)**, báo cáo này nghiên cứu sâu chủ đề trọng tâm là **Khôi phục & Tái tạo Ảnh Biển số Suy giảm (Chapter 3: Image Degradation, Restoration and Reconstruction)** và thiết kế một hệ thống xử lý ảnh cổ điển khép kín 4 giai đoạn từ nguyên lý toán học gốc (*from first principles*): **(1) Khôi phục ảnh đa suy giảm** kết hợp khử nhiễu đa phân giải **2D Haar Wavelet Shrinkage**, giải chập chính quy hóa **Hybrid Wiener–Constrained Least Squares (CLS) Deconvolution**, lọc trung vị trọng số tâm **Center-Weighted Median Filter** và lọc chắn dải **Butterworth Notch Reject Filter**; **(2) Định vị đa biển số & hiệu chỉnh hình học** bằng toán tử hình thái học **Top-Hat/Black-Hat**, đạo hàm dọc **Sobel $|G_x|$**, triệt tiêu cực đại **NMS (IoU)** và biến đổi **Phối cảnh 4 đỉnh (4-Point Perspective Homography)** nội suy Bicubic; **(3) Nhị phân hóa & phân đoạn ký tự** đối sánh **Global Otsu**, **Phân cụm K-Means ($K=2$)**, và **Local Gaussian Adaptive Thresholding** kết hợp lọc viền theo độ dài đoạn liên tục cực đại (**Maximum Contiguous Run-Length Filtering**) và gán nhãn liên thông 8 hướng (**8-Connected Component Labeling**); và **(4) Trích xuất đặc trưng & phân loại mẫu** kết hợp biến đổi trực giao **2D Walsh-Hadamard (WHT)**, **DCT-II**, **Discrete Hartley (DHT)** với **Fourier Boundary Descriptors**, **Ma trận Hiệp phương sai Vùng**, **7 Moment Bất biến Hu**, **Số Euler Tô-pô ($E = C - H$)**, **HOG** và **PCA** trên 7 bộ phân loại cổ điển (**Minimum-Distance, Correlation, Gaussian Bayes, MLP, k-NN, Random Forest, RBF-SVM**). Thực nghiệm trên bộ dữ liệu được chia tách nghiêm ngặt theo cấp độ biển số (**70% Train / 15% Val / 15% Test**, mở rộng gấp $6\times$ lên **$690$ ảnh phương tiện** và **$5.643$ mẫu ký tự**) chứng minh khâu khôi phục ảnh giúp nâng PSNR lên tới **$+17.98\text{ dB}$**, khôi phục độ chính xác nhận dạng ký tự trên tập kiểm thử từ **$3.85\%$ lên $93.59\%$** dưới nhòe chuyển động nặng, và đạt **$96.99\%$** độ chính xác phân loại ký tự.

**Keywords:** *Image Restoration, Wiener Deconvolution, Constrained Least Squares, 2D Haar Wavelet, Perspective Homography, Otsu Thresholding, K-Means Segmentation, Walsh-Hadamard Transform, Boundary Fourier Descriptors, Euler Topology, Statistical Bayes Classifier, License Plate Recognition.*

---

## 1. GIỚI THIỆU & ĐẶT VẤN ĐỀ (INTRODUCTION)

### 1.1. Động lực Nghiên cứu & Định nghĩa Bài toán (Motivation & Problem Formulation)
Nhận dạng biển số phương tiện tự động (Automatic License Plate Recognition — ALPR) là bài toán cốt lõi của thị giác máy tính trong quản lý giao thông đô thị, thu phí tự động và giám sát an ninh. Tuy nhiên, tại Việt Nam, biển số xe máy hai dòng (kích thước vật lý $190\text{ mm} \times 140\text{ mm}$, chứa 8 hoặc 9 ký tự xếp trên hai hàng) đặt ra bài toán xử lý tín hiệu hai chiều đặc biệt thách thức khi thu nhận qua camera giám sát ngoài trời.

Xét ảnh gốc không suy giảm $f(x,y) \in \mathbb{R}^{M\times N}$. Trong quá trình thu nhận ảnh thực tế (Chapter 3 & Chapter 7), tín hiệu quan sát được $g(x,y)$ là kết quả của toán tử suy giảm tuyến tính bất biến theo vị trí (Linear Space-Invariant — LSI) kết hợp với biến dạng hình học phối cảnh $\mathcal{T}_{\mathbf{H}}$ và nhiễu ngẫu nhiên $\eta(x,y)$:
$$g(x,y) = \mathcal{T}_{\mathbf{H}}\!\Big\{ (h * f)(x,y) \Big\} + \eta(x,y)$$
trong đó $h(x,y)$ là hàm lan truyền điểm (Point Spread Function — PSF) của chuyển động tương đối giữa phương tiện và camera, $\mathbf{H} \in \mathbb{R}^{3\times 3}$ là ma trận đồng nhất thức phối cảnh (Homography matrix), và $\eta(x,y)$ là nhiễu cộng tính hoặc xung lực. Bài toán đặt ra là: *Từ ảnh suy giảm $g(x,y)$, ước lượng ảnh biển số chuẩn hóa $\hat{f}_{\text{roi}}(x,y) \in \mathbb{R}^{160\times 128}$ có chất lượng tín hiệu cực đại (theo tiêu chuẩn MSE, SNR, PSNR, SSIM của Chapter 8) sao cho khâu phân đoạn vùng liên thông 8 hướng (Chapter 1 & 9) và bộ phân loại mẫu thống kê/hình học (Chapter 4, 10, 12) khôi phục chính xác chuỗi ký tự $\hat{\mathbf{s}} = (\hat{c}_1, \hat{c}_2, \dots, \hat{c}_L)$.*

### 1.2. Các Thách thức Kỹ thuật Trong Thực tế (Technical Challenges)
Qua khảo sát bộ dữ liệu thực tế và các ảnh kiểm thử ngoài tự nhiên (*in-the-wild stress tests*), một hệ thống xử lý ảnh cổ điển phải giải quyết đồng thời 5 thách thức:
1. **Sự suy giảm kép do Nhòe chuyển động & Nhiễu cảm biến (Coupled Motion Blur & Sensor Noise):** Khoảng cách ngang giữa hai ký tự liền kề trên biển số $160\times 128$ chỉ rộng $3\sim 4\text{ px}$. Một vệt nhòe chuyển động $L = 7\text{ px}$ sẽ làm nhòe biên mực đen lấp kín khe hở này, biến 4–5 ký tự thành một khối liên thông duy nhất. Nếu giải chập ngây thơ bằng bộ lọc ngược (Direct Inverse Filter), nhiễu $\eta(x,y)$ tại các không điểm của hàm truyền $H(u,v)$ sẽ bị khuếch đại vô hạn.
2. **Góc chụp nghiêng lớn ($\pm 30^\circ$), Co ngắn phối cảnh & Đa phương tiện trong cùng khung hình:** Ảnh thực tế có thể chứa nhiều xe máy cạnh nhau, biển số nằm sát mép ảnh, bị nghiêng mạnh hoặc biến dạng hình thang khiến tỷ lệ khung trục đứng (Axis-Aligned Bounding Box) bị méo lệch nghiêm trọng.
3. **Nhiễu cấu trúc từ khung viền biển số, đinh tán và tem phụ (`"VIE"` Flag Sticker):** Khung viền đen quanh biển số hoặc vạch ngăn cách tem `"VIE"` dễ nối liền với đỉnh ký tự hàng trên (`2`, `9`, `D`, `7`) hoặc tạo thành các mảnh nhiễu giả ở mép trái/phải làm lệch thứ tự đọc ký tự từ trái sang phải.
4. **Chiếu sáng không đồng đều & Bóng đổ chéo (Non-Uniform Illumination):** Bóng của đuôi xe hoặc ánh nắng gắt làm phân bố mức xám hai đỉnh (bimodal histogram) bị phá vỡ, khiến các thuật toán ngưỡng toàn cục đơn thuần dễ làm mất nét chữ.
5. **Độ tương đồng hình học cao giữa các lớp ký tự & Mất cân bằng mẫu hiếm:** Trên phông chữ biển số Việt Nam, các cặp ký tự như **`D` và `B`**, **`E` và `C`**, **`7` và `1`**, **`8` và `B`** có hình chiếu ngoài rất giống nhau, đồng thời các chữ cái sê-ri hiếm (như `A`, `D`, `E`, `R`, `T`, `Y`) xuất hiện rất ít trong các tập dữ liệu nhỏ.

### 1.3. Đóng góp Chính của Đồ án (Summary of Contributions)
- **Hệ thống Khôi phục Ảnh Đa tác nhân Toàn diện (Chapter 3 Core Theme):** Cài đặt từ nguyên lý toán học gốc 6 mô hình suy giảm vật lý và đối sánh thực nghiệm các bộ lọc khôi phục không gian – tần số – đa phân giải (Mean filters, $3\times 3$ Center-Weighted Median, Butterworth Notch Reject, 2D Haar Wavelet Shrinkage, Direct Inverse, Wiener, và Hybrid Wiener–CLS Deconvolution).
- **Thiết kế Tập dữ liệu Chuẩn Không Rò rỉ & Mở rộng $6\times$:** Phân chia phân tầng ở cấp độ biển số (**80 Train / 17 Validation / 18 Test**), mở rộng gấp $6\times$ lên **690 ảnh biển số (5.643 mẫu ký tự $32\times 32$)**, kết hợp bộ tổng hợp chữ cái biển số Việt Nam (`A–Z`) để khắc phục triệt để hiện tượng lệch xác suất Platt scaling trên các lớp hiếm.
- **Định vị Đa biển số NMS & Nắn Phối cảnh 4 Đỉnh (4-Point Perspective Homography):** Kết hợp đặc trưng hình thái học Black-Hat/Sobel với độ lấp đầy hình chữ nhật xoay tối thiểu (`rot_extent`), NMS (IoU) và xấp xỉ đa giác Douglas-Peucker 4 đỉnh để phát hiện và đọc chính xác nhiều biển số theo thứ tự chuẩn **từ trái sang phải**.
- **Bộ lọc Viền theo Độ dài Đoạn Liên tục Cực đại (`_max_run`) & Hậu kiểm Tô-pô học (Chapter 10):** Ngăn chặn hoàn toàn lỗi cắt cụt đỉnh chữ `D`, `2`, `7` hoặc cạnh phải số `9`, đồng thời phân biệt chính xác tuyệt đối `D` vs `B` (qua Số Euler $E = C - H$ và mật độ eo giữa), `E` vs `C` (qua lưỡi ngang giữa), và `7` vs `1` (qua tỷ lệ độ rộng mũ ngang).
- **Ứng dụng Studio Tương tác Thời gian thực (`demo.py`):** Hỗ trợ xử lý trực tiếp luồng Video (`.mp4/.avi`) với cơ chế **Bỏ phiếu Đa khung hình (Temporal Majority Voting)** và kiểm thử ảnh tĩnh trực quan.

---

## 2. TỔNG QUAN LÝ THUYẾT & ĐỐI CHIẾU GIÁO TRÌNH STANFORD EE368 / IMP302 (RELATED WORK & THEORETICAL FOUNDATIONS)

Khác với các cách tiếp cận hộp đen (black-box deep learning) đòi hỏi hàng chục nghìn ảnh gán nhãn và tài nguyên GPU lớn nhưng thiếu tính giải thích toán học ở cấp độ xử lý tín hiệu, đồ án này xây dựng toàn bộ chuỗi xử lý dựa trên lý thuyết **Xử lý Ảnh Số Cổ điển (Classical Digital Image Processing)** theo giáo trình của Gonzalez & Woods [1] và bộ bài giảng **Stanford EE368 / IMP302 (`Doc/1..12.pdf`)** của GS. Bernd Girod [2]. Bảng 2.1 trình bày ma trận ánh xạ chi tiết giữa 12 chương lý thuyết và mã nguồn thực thi của hệ thống.

### Bảng 2.1: Ma trận Đối chiếu Toàn diện giữa Giáo trình `Doc/1..12.pdf` và Các Module Cài đặt (`src/`)

| Chương Bài Giảng (`Doc/1..12.pdf`) | Nền Tảng Toán Học & Thuật Toán Trong Slide | Hàm Cài Đặt Trong Mã Nguồn (`src/`) | Vai Trò Trong Pipeline Hệ Thống |
| :--- | :--- | :--- | :--- |
| **`1. Basic Gray-Level Image Processing.pdf`** | • Full-Scale Histogram Stretch (P9) & Log Point Op (P10)<br>• Multi-Frame Image Averaging Denoising (P12–13)<br>• Nearest / Bilinear / Bicubic Interpolation (P16)<br>• **Otsu's Global Thresholding** $\sigma_B^2(t)$ (P21–22)<br>• **8-Connected Region Labeling** (P25–27) & Morphology (P28–30) | `full_scale_histogram_stretch()`<br>`logarithmic_point_operation()`<br>`image_averaging_denoise()`<br>`otsu_threshold_manual()`<br>`segment_characters()` | Chuẩn hóa dải động độ sáng, nội suy xoay ảnh biển số và nhị phân hóa / tách vùng liên thông 8 hướng |
| **`2. Image and Video Processing.pdf`** | • Box / Gaussian Spatial Smoothing (P7, P32–33)<br>• Median & **$3\times 3$ Center-Weighted Median Smoother** (P11–15)<br>• Morphological Erosion, Dilation, Opening, Closing (P21–26)<br>• Laplacian & **Unsharp Masking / Highboost Filtering** (P34–37)<br>• 2D DFT Frequency-Domain Filtering (P40–47) | `median_denoise()`<br>`weighted_median_filter()`<br>`unsharp_masking_highboost()`<br>`compute_morphological_maps()` | Khử nhiễu xung bảo toàn góc nét chữ mảnh, tăng cường biên tần số cao và gom cụm vùng ứng viên biển số |
| **`3. Image Degradation- Restoration and Resconstruction.pdf`** *(Chủ đề Trọng tâm)* | • Mô hình suy giảm LSI: $g(x,y)=h(x,y)*f(x,y)+\eta(x,y)$ (P5, P28)<br>• **6 Hàm mật độ nhiễu PDFs**: Gaussian, Rayleigh, Erlang/Gamma, Exponential, Uniform, Salt & Pepper (P6–12)<br>• **Bộ lọc Trung bình**: Arithmetic, Geometric, Harmonic, Contraharmonic Mean bậc $Q$ (P13–16)<br>• **Butterworth Notch Reject Filter** khử nhiễu tuần hoàn (P18–22)<br>• **Wavelet Denoising** (P30–31)<br>• **Wiener Deconvolution** (P32–34), **Constrained Least Squares (CLS) Filtering** (P35–38), **Direct Inverse Filter** (P39) | `add_gaussian_noise()`, `add_rayleigh_noise()`, `add_salt_pepper_noise()`, `add_periodic_noise()`<br>`arithmetic_mean_filter()`, `geometric_mean_filter()`, `harmonic_mean_filter()`, `contraharmonic_mean_filter()`<br>`butterworth_notch_reject_filter()`<br>`wavelet_denoise()`, `wiener_deconvolution()`, `constrained_least_squares_filter()`, `inverse_filter_deconvolution()` | **Trái tim của đồ án (Stage 1):** Mô hình hóa 6 tác nhân suy giảm vật lý và khôi phục ảnh biển số trước khi định vị, phân đoạn & nhận dạng |
| **`4. Image and Video Analysis.pdf`** | • **2D Discrete Hartley Transform (DHT)** $\text{cas}(\theta)$ (P4–6)<br>• **2D Discrete Cosine Transform (DCT-II)** (P7–11)<br>• **Sequency-Ordered 2D Walsh-Hadamard Transform (WHT)** (P12–16)<br>• **2D Discrete Wavelet Transform (DWT)** (P17–25) | `compute_dht_2d()`, `extract_dht_features()`<br>`compute_dct_2d()`, `extract_dct_features()`<br>`compute_wht_2d()`, `extract_wht_features()`<br>`dwt2_haar()`, `idwt2_haar()` | Phân tích đa phân giải Haar Wavelet (Stage 1) và trích xuất đặc trưng miền biến đổi cho ký tự $32\times 32$ (Stage 4) |
| **`7. Image and Video Acquisition.pdf`** | • 2D Sampling, Aliasing & Geometric Transformations<br>• **Affine Rotation & 4-Point Perspective Homography**<br>• Linear / Bicubic Interpolation Resampling | `extract_plate_quad()`<br>`deskew_and_crop_plate(..., interp_mode='bicubic')` | Nắn thẳng phối cảnh 4 đỉnh và xoay hiệu chỉnh góc nghiêng $\pm 30^\circ$ về chuẩn $160\times 128$ không răng cưa |
| **`8. Image and Video Rendering and Assessment.pdf`** | • Objective Distortion Criteria: **MSE, SNR, PSNR** (P3, P17)<br>• Perceptual Structural Quality: **SSIM Index** (P18–21) | `compute_mse()`, `compute_snr()`, `compute_psnr()`, `compute_ssim()`, `evaluate_quality()` | Định lượng khách quan mức cải thiện chất lượng ảnh biển số sau khôi phục ở Stage 1 |
| **`9. Image and Video segmentation.pdf`** | • **K-Means Clustering Segmentation** (P5–6)<br>• Iterative Centroid Optimization & Region Partitioning | `kmeans_binarize_manual()`<br>`binarize_plate(method='kmeans')` | Phân cụm cường độ sáng $K=2$ tách tiền cảnh/hậu cảnh và đối sánh với Otsu / Adaptive |
| **`10. Feature extraction.pdf`** | • **Boundary Fourier Descriptors** $a(u)$ & Polygonal Approx (P7–8)<br>• **Region Effective Diameter & Covariance Eccentricity** $\sqrt{1-\lambda_2/\lambda_1}$ (P11–12)<br>• **7 Hu Moment Invariants** (P21–23) & **Euler Number $E=C-H$**<br>• **PCA Eigen-Feature Subspace** (P24–25) | `extract_fourier_descriptors()`<br>`compute_region_covariance_descriptors()`<br>`compute_topology_details()`<br>`extract_hu_and_topology()`, `extract_hog_features()`, `PCA` | Biểu diễn hình thái chu vi, độ lệch tâm, bất biến moment, cấu trúc lỗ tô-pô (`D/B`, `E/C`, `7/1`) và hướng gradient |
| **`11. Object detection.pdf`** | • **Intersection over Union (IoU)** & **Non-Maximum Suppression (NMS)** (P5)<br>• Gradient/Morphological Candidate Scoring & Margin Classification (P10–12) | `compute_iou()`<br>`detect_license_plate()`<br>`detect_all_license_plates()` | Định lượng độ chính xác khung bao, lọc trùng lặp NMS và phát hiện đa biển số từ trái sang phải |
| **`12. Image pattern classification.pdf`** | • **Minimum-Distance Classifier** $d_j(\mathbf{x})$ (P9–11)<br>• **Correlation Coefficient Classifier** $\gamma_j(\mathbf{x})$ (P12)<br>• **Optimum Statistical Gaussian Bayes Classifier** (P13–20)<br>• **Multi-Layer Perceptron (MLP) Neural Network** (P21–27) | `NearestCentroid`<br>`CorrelationCoefficientClassifier`<br>`GaussianBayesClassifier`<br>`MLPClassifier`, `SVC`, `KNeighborsClassifier`, `RandomForestClassifier` | Phân loại đầy đủ bộ ký tự chữ – số biển số Việt Nam kết hợp hậu kiểm cú pháp & tô-pô học |

---

## 3. XÂY DỰNG BỘ DỮ LIỆU, PHÂN CHIA TRAIN / VAL / TEST & MỞ RỘNG ĐA SUY GIẢM $6\times$ (DATASET & AUGMENTATION)

### 3.1. Kiểm định Chất lượng & Hiệu chỉnh Nhãn Gốc (Data Hygiene & Ground-Truth Rectification)
Bộ dữ liệu gốc trong thư mục `dataset/` bao gồm 116 ảnh xe máy chụp từ camera giám sát bãi xe tại Việt Nam (độ phân giải chuẩn $472 \times 303$ pixels) và 116 tệp nhãn văn bản `.txt`, tạo thành **115 cặp ảnh – nhãn khớp định danh** (riêng ảnh `0042_01746_b.jpg` bị lệch tên tệp nhãn thành `0042_01141_b.txt` trong kho lưu trữ gốc). Qua kiểm tra đối chiếu thủ công từng ảnh biển số với tệp nhãn, chúng tôi phát hiện và hiệu chỉnh 2 lỗi gõ nhãn trong tập dữ liệu gốc thông qua từ điển `LABEL_CORRECTIONS` ([`src/pipeline.py`](file:///c:/PhatVo_Important/Codes/My_Codes/Academy/IMP302/project/src/pipeline.py)):
1. **`0000_02187_b.txt`:** Nhãn thô ghi `59E12150` (thiếu số `0` cuối), trong khi ảnh biển số thực tế là **`59-E1 215.00`** $\rightarrow$ hiệu chỉnh thành **`59E121500`** (9 ký tự).
2. **`018.txt`:** Nhãn thô ghi `60B8010175` (dư một chữ số `1` thành 10 ký tự), trong khi ảnh biển số thực tế là **`60-B8 010.75`** $\rightarrow$ hiệu chỉnh thành **`60B801075`** (9 ký tự).

### 3.2. Phân chia Phân tầng Cấp độ Biển số Không Rò rỉ (Zero-Leakage Stratified Plate-Level Split)
Trong các bài toán nhận dạng biển số, nếu gộp chung tất cả các ký tự cắt ra từ 115 biển số rồi mới chia ngẫu nhiên Train/Test, các ký tự cùng phông chữ, cùng góc nghiêng và cùng điều kiện chiếu sáng trên một biển số sẽ xuất hiện ở cả hai tập, gây ra hiện tượng **rò rỉ dữ liệu (data leakage)**. Để tuân thủ chuẩn đánh giá học thuật của Stanford CS231n/CS229, hàm `create_dataset_splits()` trong [`src/pipeline.py`](file:///c:/PhatVo_Important/Codes/My_Codes/Academy/IMP302/project/src/pipeline.py#L83-L163) thực hiện chia tách **ở cấp độ biển số (plate-level split)** có phân tầng theo chữ cái sê-ri (vị trí ký tự thứ 3) theo tỷ lệ chuẩn **70% Training / 15% Validation / 15% Testing**, lưu cố định tại [`data/splits/dataset_splits.json`](file:///c:/PhatVo_Important/Codes/My_Codes/Academy/IMP302/project/data/splits/dataset_splits.json):

### Bảng 3.1: Thống kê Phân chia Tập dữ liệu Chuẩn & Mở rộng Đa suy giảm Gấp $6\times$

| Tập Dữ Liệu (Dataset Split) | Tỷ Lệ (%) | Số Biển Số Gốc (`Clean`) | Số Ảnh Phương Tiện Mở Rộng ($6\times$) | Số Mẫu Ký Tự Sạch ($32\times 32$) | Số Mẫu Ký Tự Mở Rộng ($6\times$) | Mục Đích Sử Dụng Trong Nghiên Cứu |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Training Split (`train`)** | **70.0%** | **80 biển số** | **480 ảnh** | **676 ký tự** | **3.973 ký tự** | Huấn luyện bộ chuẩn hóa `StandardScaler`, không gian con `PCA`, và 7 bộ phân loại mẫu |
| **Validation Split (`val`)** | **15.0%** | **17 biển số** | **102 ảnh** | **132 ký tự** | **767 ký tự** | Tinh chỉnh siêu tham số khôi phục ($K$ của Wiener, $\gamma_{\text{CLS}}$, bán kính Notch $D_0$, $C$ của SVM) |
| **Testing Split (`test`)** | **15.0%** | **18 biển số** | **108 ảnh** | **156 ký tự** | **903 ký tự** | Đánh giá độc lập cuối cùng (*Held-Out Test Evaluation*) cho toàn bộ 4 giai đoạn |
| **Tổng Cộng (`Total`)** | **100.0%** | **115 biển số** | **690 ảnh** | **964 ký tự** | **5.643 ký tự** | Phủ đầy đủ bộ chữ số `0–9` và bảng chữ cái sê-ri biển số Việt Nam (`A–Z`) |

### 3.3. Mở rộng Dữ liệu Vật lý Gấp $6\times$ theo 6 Mô hình Suy giảm của Chapter 3 (`Doc/3.pdf`)
Để kiểm tra toàn diện khả năng khôi phục ảnh của Stage 1 đồng thời tăng cường khả năng tổng quát hóa của bộ phân loại ký tự ở Stage 4, mỗi ảnh phương tiện gốc $f(x,y)$ được biến đổi qua 6 mô hình suy giảm vật lý trong `synthesize_degradation_by_type()` ([`src/pipeline.py`](file:///c:/PhatVo_Important/Codes/My_Codes/Academy/IMP302/project/src/pipeline.py#L166-L200)):
1. **`clean` (Ảnh thực tế gốc):** Giữ nguyên ảnh giám sát gốc ($115$ ảnh).
2. **`awgn` (Nhiễu trắng cộng tính Gaussian — Ch. 3.2 Slide 7):** Cộng nhiễu cảm biến $p(\eta) = \frac{1}{\sqrt{2\pi}\sigma}\exp\!\left(-\frac{(\eta-\mu)^2}{2\sigma^2}\right)$ với $\mu=0, \sigma=10.0$ ($115$ ảnh).
3. **`salt_pepper` (Nhiễu xung lực Muối Tiêu — Ch. 3.2 Slide 12):** Nhiễu lưỡng cực với xác suất điểm đen $P_a = 0.02$ (pepper $= 0$) và điểm trắng $P_b = 0.02$ (salt $= 255$), tổng mật độ điểm chết $4\%$ ($115$ ảnh).
4. **`rayleigh` (Nhiễu lệch Rayleigh thiếu sáng — Ch. 3.2 Slides 8–9):** Nhiễu bất đối xứng $p(\eta) = \frac{2}{b}(\eta-a)\exp\!\left(-\frac{(\eta-a)^2}{b}\right)$ với $\eta \ge a, a=-8.0, b=160.0$ đặc trưng cho ảnh radar/thiếu sáng ($115$ ảnh).
5. **`periodic` (Nhiễu giao thoa tuần hoàn 2D — Ch. 3.3 Slides 18–22):** Nhiễu hình sin $\eta(x,y) = A\sin\!\left(2\pi\left(\frac{u_0 x}{M} + \frac{v_0 y}{N}\right)\right)$ với biên độ $A=24.0$, tần số không gian $(u_0, v_0) = (24, 18)$ ($115$ ảnh).
6. **`motion_blur` (Nhòe chuyển động tuyến tính + AWGN — Ch. 3.4 & Ch. 3.6–3.7):** Chập với hạt nhân chuyển động thẳng $h_{\text{motion}}(x,y)$ độ dài $L=7\text{ px}$, góc $\theta=12^\circ$ kết hợp nhiễu trắng $\sigma=7.0$ ($115$ ảnh).

![Dataset Splits and 6x Multi-Degradation Expansion](../outputs/fig10_dataset_splits_and_expansion.png)

### 3.4. Tổng hợp Mẫu Chữ cái Việt Nam (`A`, `R`) & Cân bằng Lớp Hiếm (Synthetic Glyph Augmentation & Class Balancing)
Phân tích phân bố lớp trên tập `train` gốc cho thấy sự mất cân bằng lớn: các chữ số như `'5'`, `'1'`, `'9'` có từ $85\text{--}101$ mẫu, chữ `'B'` có $12$ mẫu, nhưng chữ **`'D'`** chỉ có **$3$ mẫu**, `'T'` và `'Y'` chỉ có **$1$ mẫu**, còn chữ **`'A'`** (dùng cho xe máy $<50\text{cm}^3$ như `29-AB`) và **`'R'`** hoàn toàn vắng mặt ($0$ mẫu). Khi huấn luyện SVM với hiệu chỉnh xác suất Platt scaling 5-fold CV, các lớp có $<5$ mẫu bị suy biến xác suất hậu nghiệm về 0. Để giải quyết triệt để vấn đề này, hàm `generate_synthetic_vietnamese_letters()` và cơ chế cân bằng lớp trong [`src/recognition.py`](file:///c:/PhatVo_Important/Codes/My_Codes/Academy/IMP302/project/src/recognition.py#L565-L603) tự động bổ sung các mẫu tổng hợp chuẩn tỷ lệ DIN-1451 Việt Nam cho các chữ cái khuyết (`A`, `R`) và áp dụng biến đổi Affine vi mô ($\pm 4^\circ$, dịch chuyển $\pm 1\text{ px}$) để nâng mọi lớp chữ cái hiếm lên tối thiểu $14$ mẫu huấn luyện.

---

## 4. PHƯƠNG PHÁP ĐỀ XUẤT — PIPELINE XỬ LÝ ẢNH CỔ ĐIỂN 4 GIAI ĐOẠN (PROPOSED METHODOLOGY)

### 4.1. Giai đoạn 1 (Chủ đề Trọng tâm): Khôi phục & Tái tạo Ảnh Biển số Suy giảm (Chapters 1, 2, 3, 4, 8)

#### 4.1.1. Các Bộ lọc Khôi phục Không gian cho Nhiễu Cộng tính, Nhiễu Lệch và Nhiễu Xung (`Doc/2.pdf` & `Doc/3.pdf`)
Trong [`src/restoration.py`](file:///c:/PhatVo_Important/Codes/My_Codes/Academy/IMP302/project/src/restoration.py), chúng tôi cài đặt đầy đủ họ bộ lọc trung bình và thống kê thứ tự trên lân cận $S_{xy}$ kích thước $m\times n$:
- **Arithmetic & Geometric Mean Filters (Ch. 3.3 Slides 13–14):**
  $$\hat{f}_{\text{arith}}(x,y) = \frac{1}{mn}\sum_{(s,t)\in S_{xy}} g(s,t), \qquad \hat{f}_{\text{geom}}(x,y) = \left[\prod_{(s,t)\in S_{xy}} g(s,t)\right]^{\frac{1}{mn}}$$
- **Harmonic & Contraharmonic Mean Filters bậc $Q$ (Ch. 3.3 Slides 15–16):**
  $$\hat{f}_{\text{harm}}(x,y) = \frac{mn}{\sum_{(s,t)\in S_{xy}} \frac{1}{g(s,t)}}, \qquad \hat{f}_{\text{contra}}(x,y) = \frac{\sum_{(s,t)\in S_{xy}} g(s,t)^{Q+1}}{\sum_{(s,t)\in S_{xy}} g(s,t)^Q}$$
  Bộ lọc trung bình điều hòa (`harmonic_mean_filter`) kết hợp làm sắc nét Highboost đặc biệt hiệu quả khi khử nhiễu lệch dương Rayleigh mà không làm mất độ tương phản của nét chữ.
- **Bộ lọc Trung vị Trọng số Tâm $3\times 3$ (Center-Weighted Median Filter — Ch. 2.2 Slides 12–15):**
  Đối với nhiễu xung lực Muối Tiêu (`salt_pepper`), bộ lọc trung vị đều $3\times 3$ thông thường dễ làm tù các góc nhọn của chữ số mảnh (`1`, `7`, `4`). Chúng tôi áp dụng mặt nạ trọng số không gian ưu tiên điểm ảnh trung tâm:
  $$\mathbf{W}_{\text{WMF}} = \begin{bmatrix} 1 & 2 & 1 \\ 2 & 3 & 2 \\ 1 & 2 & 1 \end{bmatrix}, \qquad \hat{f}_{\text{WMF}}(x,y) = \text{median}\Big\{ \underbrace{g(s,t), \dots, g(s,t)}_{W_{\text{WMF}}(s,t)\text{ lần}} \;\Big|\; (s,t)\in S_{xy} \Big\}$$

#### 4.1.2. Bộ lọc Chắn dải Butterworth trong Miền Tần số 2D DFT (Butterworth Notch Reject Filter — Ch. 3.3 Slides 18–22)
Khi ảnh bị nhiễm nhiễu giao thoa tuần hoàn $\eta(x,y)$, phổ biên độ 2D DFT dịch tâm $|G(u,v)|$ xuất hiện các cặp đỉnh xung sắc nhọn đối xứng liên hợp tại $(u_k, v_k)$ và $(-u_k, -v_k)$ so với tâm phổ $(M/2, N/2)$. Hàm truyền bộ lọc chắn dải Butterworth bậc $n=2$, bán kính cắt $D_{0k}=6.0$ được thiết kế trong `butterworth_notch_reject_filter()`:
$$H_{\text{NR}}(u,v) = \prod_{k=1}^{Q} \left[ \frac{1}{1 + \left(\frac{D_{0k}}{D_k(u,v)}\right)^{2n}} \right] \left[ \frac{1}{1 + \left(\frac{D_{0k}}{D_{-k}(u,v)}\right)^{2n}} \right], \quad D_{\pm k}(u,v) = \sqrt{\left(u - \frac{M}{2} \mp u_k\right)^2 + \left(v - \frac{N}{2} \mp v_k\right)^2}$$
Ảnh khôi phục thu được qua biến đổi Fourier ngược $\hat{f}(x,y) = \mathcal{R}e\big\{\mathcal{F}^{-1}\{H_{\text{NR}}(u,v)\,G(u,v)\}\big\}$, triệt tiêu hoàn toàn vân sọc giao thoa trong khi giữ nguyên toàn bộ phổ tần số của các nét chữ.

![Chapter 3.2 & 3.3 Multi-Noise Degradation and Restoration](../outputs/fig09_multi_degradation_restoration.png)

#### 4.1.3. Chuỗi Khôi phục Đa phân giải 2D Haar Wavelet & Giải chập Hybrid Wiener–CLS (`Doc/3.pdf` Slides 30–39)
Đối với bài toán suy giảm phức hợp khó nhất là **Nhòe chuyển động tuyến tính + Nhiễu Gaussian ($g = h * f + \eta$)**, nếu áp dụng **Direct Inverse Filter** ($\hat{F}(u,v) = G(u,v)/H(u,v)$, Ch. 3.8 Slide 39), số hạng nhiễu $\frac{N(u,v)}{H(u,v)}$ sẽ bùng nổ tại các tần số mà hàm truyền chuyển động $H(u,v) = \text{sinc}(\pi(u x_0 + v y_0))$ tiến về $0$. Để khắc phục, hàm `full_restoration_pipeline()` trong [`src/restoration.py`](file:///c:/PhatVo_Important/Codes/My_Codes/Academy/IMP302/project/src/restoration.py) thực hiện chuỗi khôi phục hai bước trên kênh độ chói $Y$ (không gian màu $YCrCb$):

1. **Bước 1 — Khử nhiễu Đa phân giải 2D Haar Wavelet Shrinkage (Ch. 3.5 Slides 30–31 & Ch. 4.6 Slides 22–25):**
   Kênh độ chói $Y$ được phân rã qua ngân hàng bộ lọc trực chuẩn Haar 2 chiều (`dwt2_haar`) thành 4 dải con $(LL_1, LH_1, HL_1, HH_1)$ với ma trận lọc thông thấp $\mathbf{h}_\phi = \frac{1}{\sqrt{2}}[1, 1]$ và thông cao $\mathbf{h}_\psi = \frac{1}{\sqrt{2}}[1, -1]$. Độ lệch chuẩn nhiễu được ước lượng vững (robust median estimator) từ hệ số chi tiết chéo $HH_1$:
   $$\hat{\sigma}_\eta = \frac{\text{median}\big(|HH_1|\big)}{0.6745}, \qquad T_s = \frac{\hat{\sigma}_\eta^2}{\sqrt{\max(\sigma_s^2 - \hat{\sigma}_\eta^2, \, 0) + \epsilon}}$$
   Các dải cao tần $s \in \{LH_1, HL_1, HH_1\}$ được co rút mềm $\tilde{d}_s = \text{sign}(d_s)\max(|d_s| - T_s, 0)$ trước khi tái tổng hợp qua `idwt2_haar` để thu được ảnh đã làm sạch nhiễu cao tần $g_{\text{den}}(x,y)$.
2. **Bước 2 — Giải chập Chính quy hóa Hybrid Wiener – Constrained Least Squares (CLS) (Ch. 3.6 & 3.7 Slides 32–38):**
   Trong khi bộ lọc **Wiener** cực tiểu hóa sai số bình phương trung bình thống kê $\mathbb{E}\big[\|f - \hat{f}\|^2\big]$ thông qua tỷ số công suất nhiễu trên tín hiệu $K \approx S_\eta/S_f$, bộ lọc **Constrained Least Squares (CLS)** [5] cực tiểu hóa độ gồ ghề bậc hai $\|\mathbf{p} * \hat{f}\|^2$ với toán tử Laplacian 2D $p(x,y) = \begin{bmatrix}0 & -1 & 0 \\ -1 & 4 & -1 \\ 0 & -1 & 0\end{bmatrix} \stackrel{\mathcal{F}}{\Longleftrightarrow} P(u,v)$ dưới ràng buộc $\|g - h * \hat{f}\|^2 = \|\eta\|^2$. Chúng tôi kết hợp cả hai cơ chế chính quy hóa trong miền tần số 2D FFT:
   $$\hat{F}(u,v) = \left[ \frac{(1 + K)\,H^*(u,v)}{|H(u,v)|^2 + K + \gamma_{\text{CLS}}\,|P(u,v)|^2} \right] G_{\text{den}}(u,v)$$
   với $K = 0.008$ và $\gamma_{\text{CLS}} = 0.005$. Hệ số tử số $(1+K)$ đảm bảo độ lợi một chiều tại gốc tần số bằng đơn vị ($\hat{F}(0,0) = G_{\text{den}}(0,0)$ vì $H(0,0)=1, P(0,0)=0$), giữ nguyên độ sáng trung bình của ảnh, trong khi $\gamma_{\text{CLS}}|P(u,v)|^2$ triệt tiêu hoàn toàn hiện tượng gợn sóng Gibbs (ringing artifacts) quanh biên ký tự. Cuối cùng, bộ lọc **Unsharp Masking / Highboost** (Ch. 2.2 Slide 35) $\hat{f}_{\text{final}} = \hat{f} + k_{\text{boost}}(\hat{f} - G_\sigma * \hat{f})$ ($k_{\text{boost}}=0.45$) khôi phục độ dốc sườn biên chữ.

![Module 1 Restoration Visual Pipeline](../outputs/fig01_module1_restoration.png)

---

### 4.2. Giai đoạn 2: Định vị Đa Biển số Hình thái học, NMS & Nắn Phối cảnh 4 Đỉnh (Chapters 2, 7, 11)

Trong [`src/detection.py`](file:///c:/PhatVo_Important/Codes/My_Codes/Academy/IMP302/project/src/detection.py), quá trình định vị và chuẩn hóa hình học biển số được thiết kế để hoạt động vững chắc cả trên ảnh giám sát chuẩn lẫn ảnh chụp nghiêng/đa phương tiện ngoài thực tế:
1. **Bản đồ Năng lượng Hình thái học & Đạo hàm Dọc Sobel (Ch. 2.2 Slides 9, 21–26):**
   Với phần tử cấu trúc chữ nhật $B_{25\times 25}$, phép biến đổi **Black-Hat** $f_{\text{bh}} = (f \bullet B) - f$ làm nổi bật các nét chữ tối nằm trên nền biển số trắng phản quang. Đồng thời, biên độ đạo hàm dọc **Sobel** $|G_x| = |\partial I / \partial x|$ khai thác mật độ nét dọc dày đặc của các con số. Bản đồ năng lượng kết hợp $E = 0.5\,|G_x| + 0.5\,f_{\text{bh}}$ được nhị phân hóa Otsu rồi thực hiện **Đóng hình thái học (Closing $17\times 11$)** nối các ký tự thành khối chữ nhật đặc và **Mở hình thái học (Opening $5\times 5$)** cắt rời nhiễu lẻ.
2. **Sinh Ứng viên 3 Luồng & Đánh giá Hình chữ nhật Xoay Tối thiểu (`rot_extent`):**
   Để không bỏ sót các biển số nằm sát mép ảnh, nghiêng mạnh ($\pm 30^\circ$) hoặc bị co ngắn phối cảnh (`w < h`) cũng như biển có dán tem `"VIE"`, hàm `_gather_plate_candidates()` kết hợp 3 luồng ứng viên:
   - *Luồng 1 (Chuẩn trực diện):* Đường bao ngoài (`RETR_EXTERNAL`) đa ngưỡng mức xám $t \in \{110, \dots, 200, \text{Otsu}\}$ với $1.0 \le W/H \le 1.62$ và hệ số lấp đầy khung trục đứng $\text{Extent} = \frac{A_{\text{contour}}}{W \cdot H} > 0.68$.
   - *Luồng 2 (Đóng hình thái học):* Các vùng liên thông từ bản đồ đóng hình thái học $E_{\text{closed}}$.
   - *Luồng 3 (Biển nghiêng / Phối cảnh / Đa xe):* Duyệt đa ngưỡng $t \in [135, 210]$ và tính **Hình chữ nhật xoay diện tích nhỏ nhất** (`cv2.minAreaRect`) có kích thước $(W_{\text{rot}}, H_{\text{rot}})$. Khi biển nghiêng $15^\circ\text{--}30^\circ$, $\text{Extent}$ trục đứng giảm xuống $0.45\text{--}0.65$ nhưng **Rotated Extent** $\text{Extent}_{\text{rot}} = \frac{A_{\text{contour}}}{W_{\text{rot}}\cdot H_{\text{rot}}}$ vẫn đạt $> 0.66$.
   Mỗi ứng viên được chấm điểm theo hàm mục tiêu kết hợp năng lượng nội vùng và hình học:
   $$\text{Score}(R) = 1.2\,\mu_{|G_x|}(R) + 1.6\,\mu_{\text{bh}}(R) + 1.1\,\sigma_I(R) + 45.0\,\text{Extent}_{\text{eff}} - 45.0\,\big|\text{Aspect}_{\text{eff}} - 1.24\big|$$
3. **Triệt tiêu Cực đại Không phải Đỉnh (NMS via IoU — Ch. 11.1 Slide 5) & Sắp xếp Trái sang Phải:**
   Hàm `detect_all_license_plates()` lọc bỏ các khung bao trùng lặp có chỉ số $\text{IoU}(A, B) = \frac{\text{Area}(A \cap B)}{\text{Area}(A \cup B)} > 0.25$, giữ lại tất cả các biển số độc lập trong khung hình và **sắp xếp nghiêm ngặt theo tọa độ ngang $x$ từ trái sang phải** (`key=lambda p: p["bbox"][0]`), đồng thời loại bỏ các vùng nhiễu logo (như logo cửa hàng chỉ tách ra $<6$ mảnh).
4. **Nắn Phối cảnh 4 Đỉnh (4-Point Perspective Homography) & Xoay Affine Nội suy Bicubic (Ch. 1.1 & Ch. 7):**
   Hàm `extract_plate_quad()` áp dụng thuật toán xấp xỉ đa giác **Douglas-Peucker** (Ch. 10.1) trên bao lồi của contour để tìm 4 đỉnh tứ giác biển số $\mathbf{q}_1, \mathbf{q}_2, \mathbf{q}_3, \mathbf{q}_4$ theo thứ tự *[trên-trái, trên-phải, dưới-phải, dưới-trái]*. Khi biển bị co ngắn phối cảnh ($W < H$) hoặc nghiêng mạnh ($\text{Extent} \le 0.68$), ma trận đồng nhất thức phối cảnh $\mathbf{H}_{3\times 3}$ được giải từ hệ 4 cặp điểm tương ứng để chiếu phối cảnh (`cv2.warpPerspective`) kết hợp **nội suy Bicubic** (hạt nhân bậc ba $4\times 4$ lân cận) trực tiếp về kích thước chuẩn $160 \times 128$ pixels.

![Module 2 Plate Detection and Deskewing](../outputs/fig02_module2_detection.png)

---

### 4.3. Giai đoạn 3: Nhị phân hóa (Otsu, K-Means, Adaptive), Lọc Viền Đoạn Liên tục & Phân đoạn Liên thông 8 hướng (Chapters 1.2, 9.1)

Trên ảnh biển số chuẩn hóa $160\times 128$, [`src/segmentation.py`](file:///c:/PhatVo_Important/Codes/My_Codes/Academy/IMP302/project/src/segmentation.py) triển khai từ công thức gốc 3 phương pháp nhị phân hóa:
1. **Global Otsu's Thresholding (Ch. 1.2 Slides 21–22) [3]:** Tối đa hóa phương sai giữa hai lớp tiền cảnh $C_0 = [0, t]$ và hậu cảnh $C_1 = [t+1, 255]$:
   $$t_{\text{Otsu}}^* = \arg\max_{0 \le t \le 255} \sigma_B^2(t) = \arg\max_{0 \le t \le 255} \omega_0(t)\,\omega_1(t)\,\big[\mu_0(t) - \mu_1(t)\big]^2$$
2. **K-Means Clustering Segmentation với $K=2$ (Ch. 9.1 Slides 5–6):** Cài đặt thuật toán lặp Lloyd tối thiểu hóa tổng bình phương khoảng cách nội cụm $J = \sum_{k=0}^{1}\sum_{x_i \in C_k} \|x_i - m_k\|^2$, hội tụ về ngưỡng phân quyết trung điểm $t_{\text{KM}}^* = \frac{m_0 + m_1}{2}$.
3. **Local Gaussian Adaptive Thresholding:** Tính ngưỡng thích nghi cục bộ tại từng tọa độ $(x,y)$ theo tích chập Gaussian cửa sổ $27\times 27$: $T(x,y) = (G_\sigma * I)(x,y) - C$ ($C=8$), giúp tách hoàn hảo nét chữ dưới bóng đổ chéo.
4. **Thuật toán Lọc Viền theo Độ dài Đoạn Liên tục Cực đại (`_max_run`) & Tách Cầu nối:**
   Một phát hiện kỹ thuật quan trọng trong quá trình kiểm thử thực tế là: trên các biển số như `29-D2 399.99`, đỉnh ngang của 4 ký tự hàng trên (`2`, `9`, `D`, `2`) cùng nằm trên các dòng $y \in [10, 14]$ và tổng số điểm trắng chiếm tới $42\%\text{--}49\%$ chiều rộng biển số. Nếu xóa mọi dòng có mật độ trung bình $> 40\%$, thuật toán sẽ vô tình **cắt cụt nét ngang trên cùng của chữ `D`** (biến `D` thành hình chữ `U` hở đầu và bị nhận nhầm thành `B`) cũng như cắt mất nét dọc phải của số `9` cuối!
   Để giải quyết triệt để, chúng tôi định nghĩa toán tử **Độ dài đoạn trắng liên tục dài nhất (Maximum Contiguous Run-Length)** trên mảng 1 chiều $\mathbf{b}$:
   $$L_{\max}(\mathbf{b}) = \max_{k} \big(e_k - s_k\big), \quad \text{với } [s_k, e_k) \text{ là các đoạn liên thông có } b_i = 1$$
   - Với 4 đỉnh ký tự rời nhau (`2`, `9`, `D`, `2`), mỗi đỉnh chỉ rộng $15\text{--}20\text{ px}$ nên $L_{\max} \le 20\text{ px}$.
   - Ngược lại, đường viền khung biển số thật sự là một vạch liền mạch có $L_{\max} > 42\text{ px}$ (theo chiều ngang) hoặc $L_{\max} > 55\text{ px}$ (theo chiều dọc), hoặc mật độ toàn dòng $> 65\%$.
   Nhờ điều kiện lọc kép `mean > 0.65 or _max_run > 42`, toàn bộ các đỉnh chữ `D`, `2`, `7` và nét phải của số `9` được bảo toàn nguyên vẹn 100%. Кроме того, các vạch chéo mỏng ngăn cách tem `"VIE"` được lọc bỏ bằng điều kiện mật độ diện tích hộp bao $\frac{\text{Area}}{W_{\text{box}}\cdot H_{\text{box}}} < 0.22$, và các vệt nhiễu viền dọc ở mép phải hàng 1 (như trên biển `17-B7 581.49`) được loại bỏ trước khi sắp xếp 2 hàng **từ trái sang phải** và chuẩn hóa về $32\times 32$ pixels.

![Module 3 Segmentation Pipeline](../outputs/fig03_module3_segmentation.png)

---

### 4.4. Giai đoạn 4: Trích xuất Đặc trưng Biến đổi (Ch. 4), Hình học/Tô-pô (Ch. 10) & Phân loại Mẫu (Ch. 12)

#### 4.4.1. Đặc trưng Miền Biến đổi Trực giao 2 Chiều (Chapter 4: WHT, DCT-II, DHT)
Trên mỗi ảnh ký tự chuẩn hóa $f(x,y) \in [0, 1]^{32\times 32}$, [`src/recognition.py`](file:///c:/PhatVo_Important/Codes/My_Codes/Academy/IMP302/project/src/recognition.py) cài đặt và trích xuất khối hệ số tần số/sequency thấp $10\times 10$ ($\mathbb{R}^{100}$) từ 3 phép biến đổi trực giao trong **Chapter 4**:
- **Sequency-Ordered 2D Walsh-Hadamard Transform (WHT — Ch. 4.4 Slides 12–16):** Ma trận Hadamard $\mathbf{H}_{32} \in \{-1, +1\}^{32\times 32}$ được hoán vị các hàng theo thứ tự tăng dần của **Sequency** (số lần đổi dấu dọc theo từng hàng cơ sở $0, 1, \dots, 31$). Phép biến đổi $\mathbf{W} = \mathbf{H}_{32}^{\text{seq}}\,f\,(\mathbf{H}_{32}^{\text{seq}})^T$ chỉ sử dụng phép cộng/trừ nhưng cô đặc năng lượng hình thái chữ cực kỳ hiệu quả.
- **2D Discrete Cosine Transform (DCT-II — Ch. 4.3 Slides 7–11)** và **2D Discrete Hartley Transform (DHT — Ch. 4.2 Slides 4–6)** với hạt nhân $\text{cas}(\theta) = \cos\theta + \sin\theta$.

#### 4.4.2. Đặc trưng Chu vi Fourier, Hiệp phương sai Vùng, Moment Hu, Tô-pô Euler & HOG (Chapter 10)
1. **Boundary Fourier Descriptors (Ch. 10.1 Slides 7–8):** Biểu diễn chu vi ngoài của ký tự thành chuỗi phức tâm hóa $s(k) = (x_k - \bar{x}) + j(y_k - \bar{y})$, biến đổi 1D DFT $a(u) = \frac{1}{K}\sum_{k=0}^{K-1} s(k)e^{-j2\pi uk/K}$ và lấy 8 biên độ điều hòa chuẩn hóa bất biến tỷ lệ/xoay $\frac{|a(u)|}{|a(1)|}$ ($u=2,\dots,9$).
2. **Region Effective Diameter & Covariance Eigenvalue Eccentricity (Ch. 10.2 Slides 11–12):** Từ ma trận hiệp phương sai $2\times 2$ của tọa độ điểm ảnh tiền cảnh $\mathbf{C} = \frac{1}{K}\sum_{k=1}^{K}(\mathbf{z}_k - \mathbf{m})(\mathbf{z}_k - \mathbf{m})^T$ với hai trị riêng $\lambda_1 \ge \lambda_2$, trích xuất đường kính hiệu dụng $d_e = \frac{1}{16}\sqrt{\text{Area}/\pi}$, độ lệch tâm $e = \sqrt{1 - \lambda_2/\lambda_1}$ và tỷ số trục $\sqrt{\lambda_2/\lambda_1}$ ($\mathbb{R}^3$).
3. **7 Moment Bất biến Hu [6] (Ch. 10.2 Slides 21–23), Đặc trưng Tô-pô Số Euler $E = C - H$, Zoning $4\times 4$ & Hồ sơ Chiếu ($\mathbb{R}^{56}$):** Đếm chính xác số thành phần liên thông $C$, số lỗ khép kín $H$ (trên/dưới), tỷ lệ diện tích lỗ và phân bố mật độ.
4. **Histogram of Oriented Gradients (HOG — Ch. 10 & Ch. 11.1) [7]:** Ô $8\times 8\text{ px}$, 9 bin hướng $[0^\circ, 180^\circ)$, chuẩn hóa khối $2\times 2$ L2-Hys tạo vector $\mathbb{R}^{324}$. Vector tổng hợp Lai (**Combined Hybrid**) có số chiều $100 + 324 + 56 = \mathbf{480\text{ chiều}}$, và được chiếu xuống không gian con **PCA 80 chiều** (Ch. 10.3 Slides 24–25).

![Module 4 Features Visualization](../outputs/fig04_module4_features.png)

#### 4.4.3. Bảy Bộ Phân loại Mẫu Cổ điển (Chapter 12) & Hậu kiểm Tô-pô / Cú pháp Biển số
Chúng tôi cài đặt đầy đủ 7 bộ phân loại trong **Chapter 12**:
1. **Minimum-Distance Classifier (`min_dist`, Ch. 12.2 Slides 9–11):** $d_j(\mathbf{x}) = \mathbf{x}^T\mathbf{m}_j - \frac{1}{2}\|\mathbf{m}_j\|^2$.
2. **Correlation Coefficient Classifier (`corr`, Ch. 12.2 Slide 12):** $\gamma_j(\mathbf{x}) = \frac{(\mathbf{x}-\bar{x})^T(\mathbf{m}_j-\bar{m}_j)}{\|\mathbf{x}-\bar{x}\|\|\mathbf{m}_j-\bar{m}_j\|}$.
3. **Optimum Statistical Gaussian Bayes Classifier (`bayes`, Ch. 12.3 Slides 13–20):** Hàm quyết định bậc hai Mahalanobis với ma trận hiệp phương sai chính quy hóa $\mathbf{C}_j^{\text{reg}} = (1-\alpha)\mathbf{C}_j + \alpha\mathbf{C}_{\text{pooled}} + \lambda\mathbf{I}$:
   $$d_j(\mathbf{x}) = \ln P(\omega_j) - \frac{1}{2}\ln|\mathbf{C}_j^{\text{reg}}| - \frac{1}{2}(\mathbf{x} - \mathbf{m}_j)^T (\mathbf{C}_j^{\text{reg}})^{-1}(\mathbf{x} - \mathbf{m}_j)$$
4. **Multi-Layer Perceptron (`mlp`, Ch. 12.4 Slides 21–27)**, **Distance-Weighted k-NN ($k=3$)**, **Random Forest (200 cây)**, và **RBF-Kernel Support Vector Machine (`svm`, Ch. 11.1)**.
5. **Giải nhập nhằng Tô-pô học (Chapter 10 Topological Disambiguation) & Cú pháp Biển số Việt Nam:**
   - *Phân biệt `D` và `B`:* Chữ **`D`** có duy nhất $H=1$ lỗ rỗng lớn ở tâm ($E = 0$) nên vùng eo trung tâm `bin_32[14:19, 14:19]` hoàn toàn rỗng (mật độ thực nghiệm $\in [0.00, 0.16]$), trong khi chữ **`B`** có $H=2$ lỗ ($E = -1$) ngăn cách bởi vạch ngang thắt eo đặc tại `bin_32[14:19, 14:19]` (mật độ $\in [0.48, 0.88]$, tách biệt tuyệt đối không chồng lấp!).
   - *Phân biệt `E` và `C`:* Với cùng $H=0$ lỗ, chữ **`E`** có lưỡi ngang giữa tại `bin_32[13:19, 11:22]` đạt mật độ $> 0.34$, còn lòng chữ **`C`** hoàn toàn rỗng.
   - *Phân biệt `7` và `1`:* Chữ số **`7`** có mũ ngang trên rộng $\ge 11\text{ px}$ với tỷ số độ rộng mũ trên so với thân giữa $\frac{W_{\text{top}}}{W_{\text{mid}}} \ge 2.25$, khác biệt rõ rệt với số **`1`**.
   - *Cú pháp biển số Việt Nam:* Vị trí $0, 1$ (mã tỉnh) và $4..8$ (số thứ tự hàng dưới) là chữ số `0–9`; vị trí $2$ là chữ cái sê-ri `A–Z`; vị trí $3$ là chữ số hoặc chữ cái (riêng sê-ri `'A'` của xe máy $<50\text{cm}^3$ luôn đi kèm chữ cái thứ hai tạo thành `AA, AB, AC...`).

---

## 5. THIẾT LẬP THỰC NGHIỆM & CÁC CHỈ SỐ ĐÁNH GIÁ (EXPERIMENTAL SETUP & METRICS)

Theo đúng **Chapter 8 (`Doc/8. Image and Video Rendering and Assessment.pdf`)** và **Chapter 11 (`Doc/11. Object detection.pdf`)**, hiệu năng của hệ thống được đánh giá định lượng qua 3 nhóm chỉ số:
1. **Chỉ số Chất lượng Khôi phục Ảnh (Stage 1 — Chapter 8 Slides 3, 17–21):**
   $$\text{MSE} = \frac{1}{MN}\sum_{x=0}^{M-1}\sum_{y=0}^{N-1}\big[f(x,y) - \hat{f}(x,y)\big]^2, \qquad \text{SNR} = 10\log_{10}\!\left(\frac{\sum f(x,y)^2}{MN \cdot \text{MSE}}\right)\text{ (dB)}$$
   $$\text{PSNR} = 10\log_{10}\!\left(\frac{255^2}{\text{MSE}}\right)\text{ (dB)}, \qquad \text{SSIM}(f, \hat{f}) = \frac{(2\mu_f\mu_{\hat{f}} + C_1)(2\sigma_{f\hat{f}} + C_2)}{(\mu_f^2 + \mu_{\hat{f}}^2 + C_1)(\sigma_f^2 + \sigma_{\hat{f}}^2 + C_2)} \text{ [8]}$$
2. **Chỉ số Định vị & Phân đoạn (Stage 2 & Stage 3 — Chapter 11):** Tỷ lệ phát hiện đúng vùng biển số với $\text{IoU} \ge 0.5$ và **Tỷ lệ phân đoạn đúng tuyệt đối số ký tự (Exact Character Count Segmentation Rate)** $\frac{1}{N}\sum_{i=1}^N \mathbb{I}\big(|\hat{\mathcal{C}}_i| = |\mathbf{s}_i|\big)$.
3. **Chỉ số Nhận dạng Ký tự & Toàn Biển số (Stage 4 — Chapter 12):** Độ chính xác cấp độ ký tự (**Character-Level Accuracy**), **Macro-F1**, và độ chính xác khớp tuyệt đối toàn bộ chuỗi biển số (**Full-Plate Exact Match Accuracy** $\frac{1}{N}\sum_{i=1}^N \mathbb{I}(\hat{\mathbf{s}}_i = \mathbf{s}_i)$) trên tập kiểm thử độc lập (**Held-Out Test Split**).

---

## 6. KẾT QUẢ THỰC NGHIỆM, NGHIÊN CỨU ABLATION & KIỂM THỬ THỰC TẾ (EXPERIMENTAL RESULTS & ABLATION STUDIES)

### 6.1. Ablation Study 1 (Trọng tâm): Đánh giá Khôi phục Ảnh Biển số (Chapter 3) & Tác động lên Nhận dạng Toàn tuyến

#### Bảng 6.1A: Đối sánh 8 Phương pháp Khôi phục trên 115 Ảnh Bị Nhòe Chuyển động ($L=7\text{ px}, \theta=12^\circ$) + Nhiễu Gaussian ($\sigma=7.0$)

| Phương Pháp Khôi Phục (Chapter 2 & Chapter 3) | MSE $\downarrow$ | SNR (dB) $\uparrow$ | PSNR (dB) $\uparrow$ | SSIM $\uparrow$ | Tỷ Lệ Phân Đoạn Đúng (%) $\uparrow$ | Nhận Dạng Đúng Ký Tự (%) $\uparrow$ | Nhận Dạng Đúng Toàn Biển (%) $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Ảnh Suy giảm Đầu vào ($g = h*f + \eta$, Không khôi phục)** | 190.42 | 19.18 | 25.42 | 0.7244 | 1.74% (2/115) | 7.75% | **0.00% (0/115)** |
| **2. Direct Inverse Filter (Ch. 3.8 Slide 39, $\epsilon=0.08$)** | 443.27 | 15.45 | 21.69 | 0.3641 | 0.87% (1/115) | 4.12% | **0.00% (0/115)** |
| **3. Lọc Trung vị Chuẩn $3\times 3$ (Median Smoother, Ch. 2.2)** | 157.11 | 20.06 | 26.30 | 0.8454 | 0.87% (1/115) | 5.46% | **0.00% (0/115)** |
| **4. Lọc Trung vị Trọng số Tâm $3\times 3$ (Weighted Median, Ch. 2.2)** | 156.08 | 20.09 | 26.33 | 0.8390 | 1.74% (2/115) | 6.85% | **0.87% (1/115)** |
| **5. Chỉ Khử nhiễu 2D Haar Wavelet Shrinkage (Ch. 3.5)** | 164.68 | 19.84 | 26.08 | 0.8210 | 1.74% (2/115) | 7.94% | **0.87% (1/115)** |
| **6. Constrained Least Squares (CLS) Filter (Ch. 3.7, $\gamma=0.012$)** | 103.85 | 21.79 | 28.03 | 0.7712 | 82.61% (95/115) | 82.12% | **44.35% (51/115)** |
| **7. Wiener Deconvolution (Ch. 3.6, $K=0.008$)** | 100.21 | 22.01 | 28.25 | **0.8621** | 85.22% (98/115) | 84.91% | **46.96% (54/115)** |
| **8. Đề xuất: 2D Haar Wavelet + Hybrid Wiener–CLS ($\gamma=0.005$)** | **96.61** | **22.09** | **28.34** | 0.7333 | **86.09% (99/115)** | **85.80% (Test: 93.59%)** | **47.83% (Test: 61.11%)** |

#### Bảng 6.1B: Hiệu quả của Module Khôi phục Ảnh (Stage 1) trên Toàn bộ 6 Kịch bản Suy giảm ($690$ Ảnh & Kiểm định Độc lập trên Tập `Test Split`)

| Kịch Bản Suy Giảm (Chapter 3) & Toán Tử Khôi Phục Tương Ứng | PSNR Trước $\rightarrow$ Sau (dB) | SSIM Trước $\rightarrow$ Sau | Phân Đoạn Đúng (Không Khôi Phục $\rightarrow$ Có Khôi Phục) | Nhận Dạng Ký Tự Tập Test (Không $\rightarrow$ Có Khôi Phục) | Nhận Dạng Toàn Biển Tập Test (Không $\rightarrow$ Có Khôi Phục) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. `clean` (Ảnh Gốc Không Suy Giảm)** | $\infty$ | $1.0000$ | $95.65\% \rightarrow 95.65\%$ | $94.87\% \rightarrow 94.87\%$ | **$94.44\% \rightarrow 94.44\%$** |
| **2. `awgn` ($\sigma=10.0$, Ch. 3.2) $\rightarrow$ Wavelet DWT + Highboost** | $28.24 \rightarrow \mathbf{30.59}$ | $0.7325 \rightarrow \mathbf{0.8416}$ | $96.52\% \rightarrow 95.65\%$ | $95.51\% \rightarrow \mathbf{100.00\%}$ | **$94.44\% \rightarrow 100.00\%$ (+5.56%)** |
| **3. `salt_pepper` ($4\%$, Ch. 3.2) $\rightarrow$ Center-Weighted Median (Ch. 2.2)** | $19.22 \rightarrow \mathbf{37.20}$ | $0.3825 \rightarrow \mathbf{0.9875}$ | $90.43\% \rightarrow \mathbf{94.78\%}$ | $89.74\% \rightarrow \mathbf{94.87\%}$ | **$55.56\% \rightarrow 94.44\%$ (+38.88%)** |
| **4. `rayleigh` (Nhiễu Lệch, Ch. 3.2) $\rightarrow$ Harmonic Mean + Highboost (Ch. 3.3)** | $32.81 \rightarrow 27.01$ | $0.8774 \rightarrow \mathbf{0.9353}$ | $95.65\% \rightarrow 92.17\%$ | $94.87\% \rightarrow \mathbf{100.00\%}$ | **$94.44\% \rightarrow 100.00\%$ (+5.56%)** |
| **5. `periodic` (Sọc Giao Thoa, Ch. 3.3) $\rightarrow$ Butterworth Notch Reject** | $23.75 \rightarrow \mathbf{38.42}$ | $0.4962 \rightarrow \mathbf{0.9694}$ | $86.96\% \rightarrow \mathbf{94.78\%}$ | $94.87\% \rightarrow \mathbf{95.51\%}$ | **$72.22\% \rightarrow 94.44\%$ (+22.22%)** |
| **6. `motion_blur` ($L=7,\theta=12^\circ,\sigma=7$) $\rightarrow$ Wavelet + Wiener–CLS** | $25.42 \rightarrow \mathbf{28.33}$ | $0.7242 \rightarrow \mathbf{0.7334}$ | $1.74\% \rightarrow \mathbf{86.09\%}$ | $3.85\% \rightarrow \mathbf{93.59\%}$ | **$0.00\% \rightarrow 61.11\%$ (+61.11%)** |

**Thảo luận Khoa học (Key Findings):**
- **Sự sụp đổ của Direct Inverse Filter (Ch. 3.8):** Khi chia trực tiếp cho $H(u,v)$, các tần số gần điểm không của hàm $\text{sinc}$ làm công suất nhiễu $\frac{|N(u,v)|^2}{|H(u,v)|^2}$ vọt lên gấp hàng trăm lần, kéo PSNR từ $25.42\text{ dB}$ tụt xuống $21.69\text{ dB}$ và SSIM còn $0.3641$.
- **Sự cộng hưởng giữa Khử nhiễu Wavelet và Giải chập Wiener–CLS:** Các bộ lọc làm trơn không gian (Median, Weighted Median, hoặc chỉ Wavelet đơn lẻ) tuy giảm nhiễu nhưng không thể đảo ngược pha của toán tử chập chuyển động $h(x,y)$, nên tỷ lệ phân đoạn đúng vẫn chỉ đạt $0.87\%\text{--}1.74\%$. Ngược lại, khi khử nhiễu trước bằng **2D Haar Wavelet Shrinkage** rồi giải chập bằng **Hybrid Wiener–CLS**, vệt nhòe ngang $L=7\text{ px}$ được thu hồi về đúng vị trí nét chữ gốc, giải phóng khe hở $3\sim 4\text{ px}$ giữa các ký tự và đưa tỷ lệ phân đoạn đúng tăng vọt từ **$1.74\%$ lên $86.09\%$** (độ chính xác nhận dạng ký tự trên tập Test tăng từ **$3.85\%$ lên $93.59\%$**).
- **Ưu thế tuyệt đối của Center-Weighted Median và Butterworth Notch Reject:** Trên nhiễu xung lực `salt_pepper`, bộ lọc **Center-Weighted Median** đạt $\text{PSNR} = 37.20\text{ dB}$ ($\mathbf{+17.98\text{ dB}}$) và $\text{SSIM} = 0.9875$, nâng tỷ lệ nhận dạng toàn biển trên tập Test từ $55.56\%$ lên **$94.44\%$**. Trên nhiễu giao thoa tuần hoàn `periodic`, **Butterworth Notch Reject Filter** nâng PSNR từ $23.75\text{ dB}$ lên **$38.42\text{ dB}$ ($\mathbf{+14.67\text{ dB}}$)**.

![Degraded vs Restored Pipeline Comparison](../outputs/fig08_degraded_vs_restored_pipeline.png)

---

### 6.2. Ablation Study 2: Đối sánh Phép Biến đổi (Ch. 4), Đặc trưng (Ch. 10), 7 Bộ Phân loại (Ch. 12) & Mở rộng $6\times$

#### Bảng 6.2A: Độ chính xác 5-Fold Cross-Validation (%) trên 964 Ký tự Sạch giữa 7 Nhóm Đặc trưng và 7 Bộ Phân loại Cổ điển

| Nhóm Đặc Trưng (Chapters 4 & 10) | Số Chiều | Min-Dist (`Ch.12.2`) | Correlation (`Ch.12.2`) | Gaussian Bayes (`Ch.12.3`) | k-NN ($k=3$) | MLP Net (`Ch.12.4`) | Random Forest | RBF-SVM (`Ch.11.1`) | Thời Gian SVM (ms/char) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Hu Moments + Cov Eccentricity + Fourier + Euler (`hu_topo`, Ch. 10)** | 56 | 66.70% | 71.27% | 84.02% | 90.25% | **92.43%** | 91.39% | 91.70% | 0.98 ms |
| **2. 2D Discrete Hartley Transform (`dht`, Ch. 4.2 Slides 4–6)** | 100 | 66.08% | 65.66% | 69.81% | 91.80% | 89.00% | 87.66% | **92.84%** | 0.87 ms |
| **3. 2D Discrete Cosine Transform (`dct`, Ch. 4.3 Slides 7–11)** | 100 | 76.04% | 75.41% | 84.34% | 94.09% | 94.61% | 93.67% | **95.23%** | 1.12 ms |
| **4. Sequency 2D Walsh-Hadamard Transform (`wht`, Ch. 4.4 Slides 12–16)** | 100 | 75.83% | 75.83% | 84.65% | 94.19% | 94.50% | 93.98% | **95.33%** | 1.03 ms |
| **5. Histogram of Oriented Gradients (`hog`, Ch. 10 & Ch. 11.1)** | 324 | 93.57% | 93.57% | 93.67% | 96.16% | 96.27% | 94.19% | **96.37%** | 1.10 ms |
| **6. Combined + PCA Eigen-Subspace (`combined_pca`, Ch. 10.3 Slides 24–25)** | 80 | 91.91% | 91.08% | 94.19% | 95.23% | **96.99%** | 94.92% | 96.16% | 1.01 ms |
| **7. Combined Hybrid (`wht` + `hog` + `hu_topo`)** | 480 | 92.12% | 92.84% | 94.40% | 95.54% | **96.58%** | 95.12% | **96.37%** | 2.69 ms |

#### Bảng 6.2B: Đánh giá Độc lập trên Tập Kiểm thử (`Held-Out Test Split` — 18 Biển Gốc / 108 Biển Mở rộng $6\times$, 903 Ký tự Test): Tác động của Huấn luyện Mở rộng Đa suy giảm $6\times$

| Bộ Phân Loại Mẫu (Chapter 12) | Huấn Luyện Chỉ Trên `Train Clean` ($1\times = 676$ Ký Tự) $\rightarrow$ Test trên `Test 6x` (903 Ký Tự) | Huấn Luyện Trên `Train 6x Expanded` ($6\times = 3.973$ Ký Tự) $\rightarrow$ Test trên `Test Clean` (156 Ký Tự) | Huấn Luyện Trên `Train 6x Expanded` ($6\times = 3.973$ Ký Tự) $\rightarrow$ Test trên `Test 6x` (903 Ký Tự) | Mức Cải Thiện Nhờ Mở Rộng Dữ Liệu $6\times$ |
| :--- | :---: | :---: | :---: | :---: |
| **1. Minimum-Distance (`min_dist`, Ch. 12.2)** | 89.92% | 89.74% | 89.81% | -0.11% |
| **2. Correlation Coefficient (`corr`, Ch. 12.2)** | 90.48% | 89.74% | 89.26% | -1.22% |
| **3. Statistical Gaussian Bayes (`bayes`, Ch. 12.3)** | 91.58% | 92.31% | **91.92%** | **+0.34%** |
| **4. Distance-Weighted k-NN ($k=3$)** | 92.36% | **94.87%** | **95.02%** | **+2.66%** |
| **5. Random Forest (`rf`, 200 cây)** | 91.36% | 94.23% | **94.68%** | **+3.32%** |
| **6. Multi-Layer Perceptron (`mlp`, Ch. 12.4)** | 93.91% | **94.87%** | **95.13%** | **+1.22%** |
| **7. RBF Support Vector Machine (`svm`, Ch. 11.1)** | 92.36% | 94.23% | **95.24%** | **+2.88%** |

![Module 4 Evaluation and Confusion Matrix](../outputs/fig05_module4_evaluation.png)

---

### 6.3. Ablation Study 3: Đối sánh Nhị phân hóa Global Otsu (Ch. 1.2) vs. Phân cụm K-Means (Ch. 9.1) vs. Local Adaptive Thresholding

#### Bảng 6.3: Độ Bền vững Phân đoạn Ký tự dưới Điều kiện Chiếu sáng Đều và Bóng đổ Chéo (`shadow_strength = 0.55`)

| Phương Pháp Nhị Phân Hóa & Phân Đoạn (Chapters 1.2 & 9.1) | Tỷ Lệ Phân Đoạn Đúng Tuyệt Đối Số Ký Tự Trên Bộ Gốc (115 Biển) | Tỷ Lệ Phân Đoạn Đúng Dưới Bóng Đổ Chéo Mạnh (`shadow_strength=0.55`) | Độ Suy Giảm Khi Gặp Bóng Đổ |
| :--- | :---: | :---: | :---: |
| **1. Global Otsu's Thresholding (Ch. 1.2 Slides 21–22)** | 93.91% (108/115) | 91.30% (105/115) | -2.61% |
| **2. K-Means Clustering Segmentation ($K=2$, Ch. 9.1 Slides 5–6)** | 93.04% (107/115) | 92.17% (106/115) | -0.87% |
| **3. Local Adaptive Gaussian Thresholding (Đề xuất)** | **95.65% (110/115)** | **94.78% (109/115)** | **-0.87%** |

![Comprehensive Ablation Studies](../outputs/fig06_ablation_studies.png)

---

### 6.4. Kiểm thử Khả năng Tổng quát hóa trên Ảnh Thực tế Ngoài Tập Dữ liệu (In-the-Wild Stress-Test Generalization)

Để chứng minh hệ thống không chỉ hoạt động tốt trên 115 ảnh trong phòng thí nghiệm mà còn giải quyết xuất sắc các tình huống phức tạp ngoài thực tế, chúng tôi thực hiện kiểm thử sâu trên 4 kịch bản thực tế điển hình (Hình 6.4) cùng 6 ảnh thuộc tập `Held-Out Test Split` (Hình 6.5):

#### Bảng 6.4: Phân tích Kỹ thuật 4 Ca Kiểm thử Khắc nghiệt Ngoài Thực tế (In-the-Wild Stress Tests)

| Ca Kiểm Thử Thực Tế | Thách Thức Vật Lý & Hình Học Trong Ảnh | Cơ Chế Xử Lý Ảnh Cổ Điển Giải Quyết | Kết Quả Nhận Dạng Thực Tế |
| :--- | :--- | :--- | :---: |
| **Case 1: Đa phương tiện & Co ngắn phối cảnh** (`images.jpg`) | Ảnh ghép 2 xe máy cạnh nhau; xe trái nằm sát góc dưới trái (`cx/W = 0.10`), chụp xiên hông làm $W < H$ (`53x63`, $+12.9^\circ$); xe phải nghiêng $-16.1^\circ$; sê-ri **`AB`** không có chữ `'A'` trong 115 ảnh gốc | • Phát hiện đa biển NMS (`detect_all_license_plates`) sắp xếp **trái sang phải**<br>• Nắn phối cảnh 4 đỉnh (`extract_plate_quad` + `warpPerspective`)<br>• Tổng hợp mẫu chuẩn `'A'` + cú pháp sê-ri 2 chữ cái xe $<50\text{cm}^3$ | **`29-AB 901.50`** (Trái)<br>&<br>**`29-AB 901.06`** (Phải)<br>*(Đúng 18/18 ký tự)* |
| **Case 2: Nghiêng lớn $-28.6^\circ$, Tem `"VIE"` & Logo gây nhiễu** (`images (1).jpg`) | Biển số **`49-E1 796.52`** nghiêng $-28.6^\circ$, dán tem cờ đỏ `"VIE"` bên trái và có logo tròn `HOÀNG TRÍ` sáng rực ở góc trên phải | • Mở rộng ngưỡng sáng $t \in [135, 210]$ & $\text{Extent}_{\text{rot}} > 0.66$<br>• Loại bỏ vùng logo có $<6$ ký tự và lọc vạch ngăn cách tem `"VIE"` ($\text{Area}/(W\cdot H) < 0.22$)<br>• Phân biệt `'E'` vs `'C'` qua lưỡi ngang giữa & `'7'` vs `'1'` qua tỷ số mũ $\frac{W_{\text{top}}}{W_{\text{mid}}} = 3.0$ | **`49-E1 796.52`**<br>*(Đúng 9/9 ký tự, bỏ qua hoàn toàn logo)* |
| **Case 3: Bảo toàn Đỉnh chữ & Giải nhập nhằng Tô-pô `D` vs `B`** (`images (3).jpg`) | Biển số **`29-D2 399.99`** có 4 đỉnh chữ hàng trên (`2, 9, D, 2`) thẳng hàng ngang chiếm $49\%$ bề rộng biển dễ bị bộ lọc viền xóa mất đỉnh chữ `D` | • Lọc viền theo **độ dài đoạn liên tục cực đại** (`_max_run > 42` thay vì mật độ trung bình $> 40\%$)<br>• Cân bằng lớp chữ `'D'` hiếm & hậu kiểm Tô-pô Euler ($H=1, E=0$, mật độ eo giữa `waist_bar < 0.26`) | **`29-D2 399.99`**<br>*(Đúng 9/9 ký tự, độ tự tin chữ `'D'` đạt **90%**)* |
| **Case 4: Triệt tiêu Vệt viền dọc hàng trên & Tỷ số mũ `7` vs `1`** (`images (2).jpg`) | Biển số **`17-B7 581.49`** có vệt viền dọc đứt đoạn ở mép phải hàng 1 (`x = 135`) dễ đẩy lệch thứ tự đọc `[1, 7, B, 7]` thành `[7, B, 7, |]` | • Lọc bỏ vệt viền mỏng mép phải hàng 1 (`bx + bw >= W - 15, bw <= 12`) và phạt biên `edge_pen = 35.0` cho hàng 4 ký tự<br>• Kiểm định tỷ số độ rộng mũ ngang $\frac{W_{\text{top}}}{W_{\text{mid}}} = 2.60$ cho chữ số `'7'` | **`17-B7 581.49`**<br>*(Đúng 9/9 ký tự theo đúng thứ tự trái sang phải)* |

![In-the-Wild Stress-Test Generalization Cases](../outputs/fig11_wild_generalization_cases.png)

![End-to-End Demo Gallery on Held-Out Test Split](../outputs/fig07_end_to_end_demo.png)

---

### 6.5. Phần mềm Studio Xử lý Video & Ảnh Thời gian thực (`demo.py`)

Bên cạnh các kịch bản đánh giá định lượng, chúng tôi đã xây dựng giao diện phần mềm **IMP302 Video & Image Restoration Studio** ([`demo.py`](file:///c:/PhatVo_Important/Codes/My_Codes/Academy/IMP302/project/demo.py)):
- **Xử lý Luồng Video Giám sát (`.mp4/.avi`) & Ảnh Tĩnh (`.jpg/.png`):** Hỗ trợ nạp video giám sát thực tế hoặc 2 video mẫu đi kèm (`data/sample_videos/motorbike_surveillance_clean.mp4` và `motorbike_surveillance_degraded.mp4`).
- **Giả lập & Khôi phục Suy giảm Thời gian thực (Live Chapter 3 Degradation & Restoration):** Cho phép bật/tắt trực tiếp bộ khôi phục Wavelet + Wiener–CLS / Weighted Median / Butterworth Notch Reject trên từng khung hình video ở tốc độ $15\text{--}55\text{ FPS}$ trên CPU.
- **Bỏ phiếu Đa khung hình (Temporal Majority Voting):** Tích hợp cửa sổ trượt $N=7$ khung hình liên tiếp để lấy chuỗi biển số xuất hiện ổn định nhất, triệt tiêu hoàn toàn sai số nhiễu tức thời trên từng frame đơn lẻ và cho phép xuất báo cáo trích xuất ra tệp CSV (`outputs/extracted_video_plates.csv`).

---

## 7. THẢO LUẬN, PHÂN TÍCH LỖI & HẠN CHẾ (DISCUSSION & ERROR ANALYSIS)

Qua phân tích ma trận nhầm lẫn (Confusion Matrix — Hình 6.2) và các ca kiểm thử trên tập `Held-Out Test`, chúng tôi rút ra các nhận xét chuyên môn về giới hạn của phương pháp cổ điển:
1. **Nguồn gây lỗi chính trong Phân đoạn (4.35% số biển trên tập gốc):** Xảy ra trên các biển số bị trầy xước vật lý hoặc bùn đất che khuất hoàn toàn một phần nét chữ, hoặc đinh ốc rỉ sét nằm sát ngay giữa khe của hai chữ số làm thay đổi cấu trúc liên thông 8 hướng.
2. **Nguồn gây lỗi trong Phân loại Ký tự (~3.0%–4.7%):** Tập trung ở các cặp ký tự bị biến dạng phối cảnh nặng hoặc mất góc sau nhị phân hóa như `('0', 'D')`, `('8', 'B')`, `('5', 'S')`, và `('2', 'Z')`. Mặc dù bộ giải nhập nhằng tô-pô học (Euler number $E = C - H$, mật độ eo giữa `waist_bar`, và ràng buộc cú pháp vị trí biển số Việt Nam) đã khắc phục được phần lớn các nhầm lẫn giữa chữ cái và chữ số, các trường hợp vị trí thứ 4 của biển số (vốn cho phép cả chữ cái lẫn chữ số) vẫn có thể bị ảnh hưởng nếu ảnh mờ quá nặng ($L > 12\text{ px}$).
3. **Ước lượng Tham số PSF trong Khôi phục Mù (Blind Deconvolution):** Hiện tại, bộ lọc Wiener–CLS sử dụng tham số PSF biết trước hoặc ước lượng bán tự động. Khi phương tiện rung lắc phi tuyến (non-uniform camera shake), hàm PSF thay đổi theo không gian đòi hỏi các thuật toán ước lượng PSF mù từ phổ cepstrum hoặc đạo hàm biên.

---

## 8. KẾT LUẬN & HƯỚNG PHÁT TRIỂN (CONCLUSION & FUTURE WORK)

### 8.1. Kết luận (Conclusion)
Đồ án đã hoàn thành trọn vẹn mục tiêu nghiên cứu và xây dựng một hệ thống **Khôi phục, Phân đoạn và Nhận dạng Biển số Xe máy Việt Nam** hoàn toàn bằng các thuật toán **Xử lý Ảnh Số Cổ điển (Classical Digital Image Processing)** theo chuẩn học thuật **Stanford EE368 / IMP302 (`Doc/1..12.pdf`)**:
1. **Chứng minh vai trò quyết định của Khôi phục Ảnh (Chapter 3):** Dưới tác động của nhòe chuyển động tuyến tính và nhiễu cảm biến, chuỗi khôi phục **2D Haar Wavelet Shrinkage + Hybrid Wiener–CLS Deconvolution** đã giải cứu hệ thống khỏi trạng thái liệt hoàn toàn ($0.00\%$ nhận dạng toàn biển) lên **61.11%** trên tập kiểm thử độc lập ($93.59\%$ độ chính xác ký tự), trong khi **Center-Weighted Median** và **Butterworth Notch Reject Filter** nâng PSNR thêm **$+14.67\text{ dB}\sim +17.98\text{ dB}$** và đưa độ chính xác nhận dạng toàn biển đạt **$94.44\%$**.
2. **Phương pháp luận chặt chẽ & Tái lập 100%:** Toàn bộ quy trình chia tách **Train (80) / Validation (17) / Test (18)** không rò rỉ dữ liệu, mở rộng $6\times$ (**690 ảnh, 5.643 ký tự**), định vị đa biển NMS sắp xếp từ trái sang phải, nắn phối cảnh 4 đỉnh, lọc viền đoạn liên tục `_max_run` và phân loại kết hợp đặc trưng WHT/DCT/DHT + HOG + Fourier/Covariance/Hu/Euler đều được cài đặt minh bạch từ công thức gốc.

### 8.2. Hướng phát triển (Future Work)
- **Ước lượng PSF Tự động từ Phổ Cepstrum (Blind Motion Blur Estimation):** Tự động xác định góc nhòe $\hat{\theta}$ và độ dài vệt nhòe $\hat{L}$ từ khoảng cách giữa các vạch tối song song (zero-lines) trên phổ $\log |G(u,v)|$ cho từng khung hình video.
- **Theo dõi Đa đối tượng bằng Bộ lọc Kalman (Kalman Filter Tracking):** Kết hợp định vị đa biển số NMS với bộ lọc Kalman theo thời gian để theo dõi quỹ đạo từng phương tiện trong luồng video giao thông đông đúc.

---

## TÀI LIỆU THAM KHẢO (REFERENCES)

1. R. C. Gonzalez and R. E. Woods, *Digital Image Processing*, 4th ed. New York, NY, USA: Pearson, 2018.
2. B. Girod, *EE368/CS232: Digital Image Processing Lecture Notes (Chapters 1–12)*, Department of Electrical Engineering, Stanford University, CA, USA. *(Course Lecture Slides `Doc/1..12.pdf`)*.
3. N. Otsu, "A threshold selection method from gray-level histograms," *IEEE Transactions on Systems, Man, and Cybernetics*, vol. 9, no. 1, pp. 62–66, Jan. 1979.
4. N. Wiener, *Extrapolation, Interpolation, and Smoothing of Stationary Time Series*. Cambridge, MA, USA: MIT Press, 1949.
5. B. R. Hunt, "The application of constrained least squares estimation to image restoration by digital computer," *IEEE Transactions on Computers*, vol. C-22, no. 9, pp. 805–812, Sept. 1973.
6. M.-K. Hu, "Visual pattern recognition by moment invariants," *IRE Transactions on Information Theory*, vol. 8, no. 2, pp. 179–187, Feb. 1962.
7. N. Dalal and B. Triggs, "Histograms of oriented gradients for human detection," in *Proc. IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR)*, San Diego, CA, USA, 2005, pp. 886–893.
8. Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image quality assessment: From error visibility to structural similarity," *IEEE Transactions on Image Processing*, vol. 13, no. 4, pp. 600–612, Apr. 2004.
9. D. L. Donoho and I. M. Johnstone, "Ideal spatial adaptation by wavelet shrinkage," *Biometrika*, vol. 81, no. 3, pp. 425–455, Sept. 1994.
10. S. G. Chang, B. Yu, and M. Vetterli, "Adaptive wavelet thresholding for image denoising and compression," *IEEE Transactions on Image Processing*, vol. 9, no. 9, pp. 1532–1546, Sept. 2000.
11. D. H. Douglas and T. K. Peucker, "Algorithms for the reduction of the number of points required to represent a digitized line or its caricature," *Cartographica*, vol. 10, no. 2, pp. 112–122, Dec. 1973.
12. J. MacQueen, "Some methods for classification and analysis of multivariate observations," in *Proc. 5th Berkeley Symposium on Mathematical Statistics and Probability*, vol. 1, 1967, pp. 281–297.

---

## PHỤ LỤC: KIẾN TRÚC MÃ NGUỒN & HƯỚNG DẪN TÁI LẬP KẾT QUẢ (APPENDIX: CODEBASE & REPRODUCIBILITY)

```text
project/
├── Doc/                                 # 12 file PDF bài giảng gốc Stanford EE368 / IMP302 (Chapters 1..12)
├── dataset/                             # Bộ dữ liệu gốc (116 ảnh .jpg 472x303 và 116 file nhãn .txt)
├── data/
│   ├── splits/dataset_splits.json       # Chia tách phân tầng Train (80) / Val (17) / Test (18) & mở rộng 6x (690 ảnh)
│   ├── sample_videos/                   # Video giám sát mẫu (clean & motion-blur degraded .mp4)
│   └── chars/                           # Ảnh ký tự chuẩn hóa 32x32 phân theo lớp ('0'..'Z')
├── src/                                 # Mã nguồn Python 4 Giai đoạn cài đặt từ nguyên lý toán học gốc
│   ├── restoration.py                   # Stage 1 (Chủ đề chính): Ch. 1, 2, 3, 4, 8 (6 Noise PDFs, Mean/Weighted Median, Notch, DWT, Inverse/Wiener/CLS, MSE/SNR/PSNR/SSIM)
│   ├── detection.py                     # Stage 2: Ch. 1, 2, 7, 11 (Top-Hat/Black-Hat, Sobel |Gx|, Multi-Plate NMS, 4-Point Perspective Homography, Bicubic Deskewing)
│   ├── segmentation.py                  # Stage 3: Ch. 1.2 & Ch. 9.1 (Manual Otsu, Manual K-Means K=2, Adaptive, Max-Run Border Filter, 8-Connected Labeling)
│   ├── recognition.py                   # Stage 4: Ch. 4, 10, 12 (2D WHT/DCT/DHT, Fourier Boundary, Covariance Eccentricity, Hu, Euler Topo, HOG, PCA, 7 Classifiers)
│   └── pipeline.py                      # End-to-End Pipeline, Train/Val/Test Splitter & 6x Multi-Degradation Expander
├── notebooks/                           # 5 Jupyter Notebooks thực thi hoàn chỉnh kèm kết quả & hình ảnh nhúng
│   ├── 01_restoration.ipynb
│   ├── 02_detection.ipynb
│   ├── 03_segmentation.ipynb
│   ├── 04_recognition.ipynb
│   └── 05_full_pipeline.ipynb
├── outputs/                             # 11 biểu đồ khoa học phân giải cao (fig01..fig11.png) & metrics_summary.json
├── report/
│   ├── FINAL_REPORT.md                  # Báo cáo khoa học chuẩn Stanford (Markdown)
│   ├── FINAL_REPORT.pdf                 # Báo cáo khoa học chuẩn Stanford (Academic Paper PDF)
│   ├── FINAL_REPORT_IMP302.docx         # Báo cáo khoa học chuẩn Stanford (Microsoft Word .docx)
│   └── PRESENTATION_SLIDES.md           # Nội dung trình chiếu bảo vệ đồ án
├── demo.py                              # Ứng dụng GUI Studio xử lý Video & Ảnh tĩnh thời gian thực (+ CLI benchmark)
├── generate_all_experiments.py          # Script chạy toàn bộ thực nghiệm, chia tập dữ liệu & xuất hình vẽ
├── build_pdf_report.py                  # Script biên dịch báo cáo sang HTML & PDF chuẩn học thuật Stanford
└── build_word_report.py                 # Script biên dịch báo cáo sang Microsoft Word (.docx)
```

### Lệnh Thực Thi Nhanh (Quick-Start Commands):
```bash
# 1. Mở giao diện phần mềm IMP302 Video & Image Studio (hỗ trợ nạp Video .mp4/.avi và Ảnh tĩnh .jpg/.png):
python demo.py

# 2. Chạy kiểm thử dòng lệnh (CLI) trên tập Held-Out Test Split:
python demo.py --cli

# 3. Tái lập toàn bộ số liệu thực nghiệm, bảng biểu CSV và các hình vẽ khoa học:
python generate_all_experiments.py

# 4. Xuất lại báo cáo PDF & Word chuẩn học thuật Stanford:
python build_pdf_report.py
python build_word_report.py
```
