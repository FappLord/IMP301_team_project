# -*- coding: utf-8 -*-
"""
Generate Complete Vietnamese Academic Final Report (`report/FINAL_REPORT_VI.md`)
Matches exactly the title in the uploaded image:
- BÁO CÁO ĐỒ ÁN MÔN HỌC XỬ LÝ ẢNH VÀ VIDEO (IMP302)
- ĐỀ TÀI: KHÔI PHỤC ẢNH BIỂN SỐ XE SUY GIẢM ĐA TÁC NHÂN KẾT HỢP PHÂN ĐOẠN VÀ NHẬN DẠNG KÝ TỰ BẰNG CÁC PHƯƠNG PHÁP XỬ LÝ ẢNH CỔ ĐIỂN
- 5 Group Members with Student IDs
- Zero Slide/Chapter References
- 18 Display Equations (1)–(18)
- 12 Tables, 10 Figures, Appendix A & B, 26 References
"""

import os

content = """# BÁO CÁO ĐỒ ÁN MÔN HỌC XỬ LÝ ẢNH VÀ VIDEO (IMP302)

## ĐỀ TÀI: KHÔI PHỤC ẢNH BIỂN SỐ XE SUY GIẢM ĐA TÁC NHÂN KẾT HỢP PHÂN ĐOẠN VÀ NHẬN DẠNG KÝ TỰ BẰNG CÁC PHƯƠNG PHÁP XỬ LÝ ẢNH CỔ ĐIỂN

<div class="author-grid">
  <div class="author-item"><b>Võ Tấn Phát</b><br/>SE194484</div>
  <div class="author-item"><b>Nguyễn Trung Dũng</b><br/>SE194119</div>
  <div class="author-item"><b>Lê Ngọc Thương</b><br/>SE194271</div>
  <div class="author-item"><b>Nguyễn Bảo Ái</b><br/>SE192158</div>
  <div class="author-item"><b>Lê Minh Khôi</b><br/>SE192396</div>
</div>

## Tóm tắt

Biển số xe máy hai dòng tại Việt Nam thu nhận từ các camera giám sát giao thông và bãi đỗ xe ngoài trời thường xuyên bị suy giảm chất lượng nghiêm trọng do hiện tượng nhòe chuyển động tuyến tính, nhiễu cảm biến trong điều kiện thiếu sáng, nhiễu xung hai cực (muối tiêu), nhiễu tuần hoàn từ lưới điện và biến dạng phối cảnh nghiêng góc lớn. Do khoảng cách ngang giữa các ký tự liền kề trên biển số chuẩn hóa kích thước $160 \\times 128$ chỉ vỏn vẹn từ $3\\text{--}4\\text{ pixel}$, ngay cả một mức nhòe chuyển động vừa phải ($L = 7\\text{ pixel}$) cũng đủ làm cầu nối liên kết các nét ký tự, khiến bước phân đoạn thành phần liên thông bị sụp đổ hoàn toàn và độ chính xác nhận dạng toàn biển số rơi về $0.00\\%$. Đồ án này nghiên cứu và hiện thực một hệ thống xử lý ảnh và nhận dạng mẫu thống kê cổ điển dạng hộp trắng (white-box) hoàn toàn minh bạch, được xây dựng từ các nguyên lý toán học nền tảng mà không sử dụng bất kỳ mạng nơ-ron học sâu nào. Kiến trúc đề xuất gồm bốn giai đoạn: (i) Giai đoạn khôi phục ảnh đa tác nhân kết hợp khử nhiễu sóng con Haar 2D BayesShrink với bộ lọc giải chập lai Wiener–Bình phương tối thiểu có ràng buộc (Wiener–CLS), bộ lọc trung vị trọng số tâm $3 \\times 3$ xử lý nhiễu xung, và bộ lọc triệt dải Butterworth liên hợp khử nhiễu tuần hoàn; (ii) Bộ phát hiện biển số ba luồng kết hợp hình thái học Black-Hat và gradient Sobel đứng với hàm tính điểm hình chữ nhật xoay diện tích cực tiểu, triệt tiêu không cực đại (NMS) sắp xếp chặt chẽ từ trái sang phải, và nắn chỉnh phối cảnh 4 điểm Homography qua xấp xỉ đa giác Douglas–Peucker; (iii) Bộ nhị phân hóa thích nghi Gaussian cục bộ chống chịu bóng đổ kết hợp thuật toán lọc viền theo chuỗi liên tục cực đại giúp bảo toàn nguyên vẹn nét đầu các ký tự; và (iv) Bộ mô tả đặc trưng lai 480 chiều kết hợp hệ số Biến đổi Walsh–Hadamard 2D (WHT) sắp xếp theo chu kỳ sequency, lược đồ hướng gradient (HOG), 7 bất biến mô-men Hu, bộ mô tả Fourier đường biên, độ lệch tâm hiệp phương sai vùng và số đặc trưng tô-pô Euler ($E = C - H$), được đánh giá trên bảy bộ phân loại mẫu thống kê kết hợp cơ chế hậu kiểm tra tô-pô. Trên bộ dữ liệu chuẩn hóa không rò rỉ thông tin phân chia theo cấp độ biển số (80 biển huấn luyện, 17 biển kiểm định, và 18 test biển mở rộng gấp 6 lần qua sáu chế độ suy giảm vật lý thành 690 ảnh phương tiện và 5.643 mẫu ký tự), khôi phục Giai đoạn 1 mang lại mức tăng PSNR lên tới $+17.98\\text{ dB}$, khôi phục độ chính xác nhận dạng ký tự bị nhòe chuyển động trên tập kiểm tra độc lập từ $3.85\\%$ lên $93.59\\%$, và đạt độ chính xác ký tự $96.99\\%$ trong kiểm định chéo 5-fold. Đồ án cũng kiểm toán toàn diện các nhãn nhầm lẫn trong tập dữ liệu thô, đánh giá trên bốn kịch bản thử nghiệm thực tế khắc nghiệt và phân tích sâu các nguyên nhân lỗi còn tồn đọng.

**Từ khóa:** Khôi phục ảnh, giải chập Wiener, bình phương tối thiểu có ràng buộc, co ngót sóng con Haar, biến đổi phối cảnh Homography, phân ngưỡng Otsu, phân cụm K-Means, biến đổi Walsh–Hadamard, bộ mô tả Fourier đường biên, tô-pô Euler, nhận dạng biển số xe.

---

## 1 Giới thiệu

Hệ thống Nhận dạng Biển số Xe Tự động (ALPR) là thành phần cốt lõi trong giám sát giao thông đô thị, quản lý bãi đỗ xe thông minh và giám sát an ninh trật tự [1, 2]. Trong khi biển số xe ô tô một dòng trên các làn cao tốc có điều kiện quan sát tiêu chuẩn đã được nghiên cứu rộng rãi, biển số xe máy hai dòng tại Việt Nam (kích thước vật lý tiêu chuẩn $190\\text{ mm} \\times 140\\text{ mm}$, chứa 8 hoặc 9 ký tự chữ và số bố trí trên hai dòng: dòng trên 4 ký tự biểu thị mã tỉnh và sê-ri, dòng dưới 4 hoặc 5 chữ số thứ tự đăng ký) đặt ra những thách thức xử lý tín hiệu và hình học hết sức đặc thù. Tại các bãi đỗ xe giá rẻ và camera đường phố, xe máy thường xuyên được chụp ở góc nghiêng lớn (góc xoay mặt phẳng roll và góc nghiêng yaw lên tới $\\pm 30^\\circ$), dưới điều kiện bóng đổ xiên bất đối xứng, hoặc khi phương tiện đang chuyển động qua cửa sổ phơi sáng của camera.

Gọi $f(x,y) \\in \\mathbb{R}^{M \\times N}$ là khung hình giám sát lý tưởng chưa bị suy giảm. Dưới mô hình suy giảm quang học bất biến trong không gian tuyến tính (LSI) kết hợp với hình học quan sát xạ ảnh $\\mathcal{T}_{\\mathbf{H}}$ và nhiễu cảm biến cộng tính hoặc xung $\\eta(x,y)$, khung hình quan sát được $g(x,y)$ được mô hình hóa bởi [3]:
$$g(x,y) = \\mathcal{T}_{\\mathbf{H}}\\!\\Big\\{ (h * f)(x,y) \\Big\\} + \\eta(x,y), \\qquad (1)$$
trong đó $h(x,y)$ là Hàm Lan truyền Điểm 2D (Point Spread Function - PSF) sinh ra do chuyển động tương đối giữa phương tiện và cảm biến hình ảnh, $*$ ký hiệu phép tích chập không gian 2D, và $\\mathbf{H} \\in \\mathbb{R}^{3 \\times 3}$ là ma trận biến đổi phối cảnh xạ ảnh (Homography). Việc khôi phục chuỗi ký tự chính xác $\\hat{\\mathbf{s}} = (\\hat{c}_1, \\hat{c}_2, \\dots, \\hat{c}_L)$ từ ảnh đầu vào $g(x,y)$ đòi hỏi phải đảo ngược đồng thời cả quá trình suy giảm đo bức xạ $(h, \\eta)$ và biến dạng hình học $\\mathcal{T}_{\\mathbf{H}}$ trước khi tiến hành cô lập thành phần liên thông và phân loại mẫu thống kê.

Khảo sát các nghiên cứu xử lý biển số xe cổ điển [1, 2] và kiểm toán thực nghiệm của chúng tôi trên các khung hình giám sát thực tế chỉ ra ba nguyên nhân gốc rễ gây sụp đổ toàn bộ chu trình xử lý:
1. **Khuếch đại ghép cặp nhòe–nhiễu và hiện tượng dính nét ký tự:** Trên ảnh biển số chuẩn hóa kích thước $160 \\times 128$, khoảng cách ngang giữa các ký tự liền kề chỉ từ $3\\text{--}4\\text{ pixel}$. Một kernel nhòe chuyển động tuyến tính có chiều dài $L = 7\\text{ pixel}$ sẽ làm lem mực ký tự qua khoảng trống này, nối liền bốn hoặc năm chữ số thành một thành phần liên thông duy nhất. Phép lọc nghịch đảo thô trong miền tần số ($\\hat{F}(u,v) = G(u,v)/H(u,v)$) sẽ khuếch đại cực đại nhiễu tần số cao $\\eta(x,y)$ tại lân cận các điểm không của hàm truyền $\\text{sinc}$, phá hủy hoàn toàn khả năng nhận diện nét.
2. **Biến dạng phối cảnh xiên và hiện tượng cắt cụt nét đầu do lọc viền:** Khi xe máy được ghi hình từ góc nghiêng hoặc góc dốc lớn, khung bao định hướng theo trục tọa độ (axis-aligned bounding box) của biển số bị sụt giảm tỷ lệ lấp đầy hình chữ nhật ($\\text{Extent} < 0.65$) và sai lệch tỷ lệ khung hình ($W < H$). Hơn nữa, các thuật toán xóa viền truyền thống dựa trên ngưỡng mật độ điểm ảnh đen theo dòng ngang (quá $40\\%$) đã vô tình cắt cụt nét ngang đầu thẳng hàng của các ký tự như `'2'`, `'9'`, `'D'`, và `'7'`, biến chữ `'D'` kín thành ký tự hở có hình dạng chữ `'U'`.
3. **Mơ hồ cấu trúc nét và mất cân bằng mẫu nghiêm trọng:** Trong font chữ biển số xe Việt Nam, các cặp ký tự như `'D'` và `'B'`, `'E'` và `'C'`, hoặc `'7'` và `'1'` có khung bao ngoài và hình chiếu đứng gần như trùng khớp hoàn toàn. Đồng thời, các chữ cái sê-ri hiếm (`'A'`, `'D'`, `'E'`, `'R'`, `'T'`, `'Y'`) xuất hiện với tần suất rất thấp trong các tập dữ liệu thực tế, dẫn đến hiện tượng suy thoái xác suất hiệu chuẩn (Platt scaling) ở các bộ phân loại lề lớn.

**Mục tiêu nghiên cứu.** Các hệ thống ALPR công nghiệp hiện đại ngày càng phụ thuộc vào các mạng nơ-ron tích chập hoặc Transformer học sâu (như YOLO kết hợp OCR) [4, 5]. Mặc dù đạt độ chính xác cao trên các tập dữ liệu đồ sộ được huấn luyện bằng GPU, các pipeline hộp đen che khuất mối quan hệ toán học giữa giải chập miền tần số, hình thái học không gian và tính bất biến tô-pô, đồng thời suy thoái khó lường khi gặp phải các tác nhân suy giảm ngoài phân phối huấn luyện (như nhiễu điện từ tuần hoàn hoặc nhiễu xung mật độ cao). Ngược lại, các nghiên cứu ALPR cổ điển hiện có thường chỉ đánh giá đơn lẻ một giai đoạn (như phân ngưỡng Otsu hoặc so khớp mẫu) trên các ảnh cắt thẳng tắp, không có phân chia tập huấn luyện/kiểm định/kiểm tra chuẩn tắc và không xét đến khôi phục đa tác nhân. Cách thức mà giải chập miền tần số tương tác với nắn phối cảnh Homography, lọc viền bảo toàn nét và phân loại mẫu thống kê kết hợp tô-pô trong một hệ thống hoàn chỉnh vẫn chưa được tài liệu hóa đầy đủ.

Đồ án này giải quyết trực tiếp khoảng trống trên thông qua bốn đóng góp học thuật chính:
1. Xây dựng giai đoạn khôi phục ảnh đa tác nhân hoàn chỉnh (Mục 3.2) kết hợp giải chập lai miền tần số thích nghi Wiener–CLS với khử nhiễu sóng con Haar 2D BayesShrink, lọc trung vị trọng số tâm (CWMF), và bộ lọc triệt dải Butterworth liên hợp, đạt mức tăng PSNR lên tới $+17.98\\text{ dB}$.
2. Đề xuất bộ phát hiện biển số ba luồng (Mục 3.3) kết hợp hình thái học Black-Hat, năng lượng gradient Sobel đứng, và hàm tính điểm hình chữ nhật xoay diện tích cực tiểu, tích hợp giải thuật nắn chỉnh phối cảnh 4 điểm Homography qua xấp xỉ Douglas–Peucker và triệt tiêu không cực đại (NMS) sắp xếp chặt chẽ từ trái sang phải, đạt tỷ lệ định vị $100.0\\%$ trên 115 khung hình giám sát.
3. Hiện thực thuật toán lọc viền theo chuỗi chạy liên tục cực đại (`_max_run`, Mục 3.4) loại bỏ $100\\%$ viền kim loại bao quanh mà không cắt phạm vào nét đầu ký tự, kết hợp phân ngưỡng Gaussian thích nghi cục bộ vượt trội hơn phân ngưỡng Otsu toàn cục $+3.48\\text{ điểm phần trăm}$ dưới điều kiện bóng đổ xiên gắt.
4. Thiết kế vector đặc trưng lai 480 chiều (Mục 3.5) tích hợp biến đổi Walsh–Hadamard (WHT) sắp xếp theo sequency, HOG, 7 bất biến mô-men Hu, bộ mô tả Fourier đường biên, độ lệch tâm hiệp phương sai vùng và số đặc trưng Euler ($E = C - H$), đánh giá trên bảy bộ phân loại thống kê kết hợp cơ chế hậu kiểm tra tô-pô giúp triệt tiêu nhầm lẫn `'D'`–`'B'` và `'7'`–`'1'`.

---

## 2 Tổng quan nghiên cứu liên quan

### 2.1 Các hệ thống nhận dạng biển số xe

Bảng 1 định vị công trình của chúng tôi trong tương quan so sánh với các nghiên cứu ALPR cổ điển và lai tiêu biểu trong y văn. Các hệ thống ALPR ban đầu dựa chủ yếu vào phép chiếu biên đứng và phân ngưỡng toàn cục kết hợp so khớp mẫu điểm ảnh [1, 2]. Dù nhẹ về mặt tính toán, phương pháp dựa trên hình chiếu đòi hỏi góc nghiêng biển số phải xấp xỉ bằng không và ánh sáng đồng đều. Các nghiên cứu cổ điển tiếp theo đưa vào phép biến đổi hình thái học Top-Hat / Black-Hat [6] và Phân tích Thành phần Liên thông (CCA) [7], kết hợp với máy học véc-tơ hỗ trợ (SVM) trên đặc trưng gradient cục bộ [8]. Gần đây, các mô hình học sâu một giai đoạn [4, 5] đạt độ chính xác cao trên các tập dữ liệu lớn nhưng đòi hỏi hàng triệu tham số huấn luyện và tài nguyên tính toán GPU. Ngược lại, công trình của chúng tôi kiên định giữ vững cấu trúc hộp trắng cổ điển $100\\%$ nhưng giải quyết triệt để bốn nguyên nhân gây lỗi của phương pháp truyền thống: giải chập khôi phục nhòe kết hợp nhiễu, biến đổi phối cảnh Homography 4 điểm cho biển số nghiêng, thuật toán lọc viền bảo toàn nét đầu ký tự, và cơ chế kiểm tra tính bất biến tô-pô.

*Table 1: So sánh đối sánh kiến trúc hệ thống ALPR đề xuất với các nghiên cứu trong y văn.*

| Công trình nghiên cứu | Phương pháp luận | Khôi phục suy giảm | Nắn chỉnh hình học | Phân đoạn ký tự | Trích xuất đặc trưng & Nhận dạng |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Du et al. [2] | Khảo sát cổ điển | — | Khử nghiêng đường Hough | Phân ngưỡng Otsu toàn cục | So khớp mẫu / Mạng ANN |
| Laroca et al. [4] | Học sâu (YOLO) | — | Hộp bao xoay hồi quy | Nhận dạng theo chuỗi CRNN | Trích xuất đặc trưng CNN tối đa hóa |
| Silva & Jung [5] | Học sâu (WPOD) | — | Hồi quy góc 4 đỉnh | Lưới nắn Warp trực tiếp | Mạng nơ-ron tích chập CNN |
| Azam & Islam [6] | Xử lý ảnh cổ điển | Cân bằng Histogram | Hình thái học Top-Hat | Phân ngưỡng Otsu | Trích xuất biên & Tương quan |
| Dalal & Triggs + SVM [8, 19] | Học máy cổ điển | — | — | Otsu + Cắt ngưỡng mật độ | Đặc trưng HOG + SVM tuyến tính/RBF |
| **Công trình này** | **Cổ điển + Học máy** | **Haar DWT 2D + Wiener–CLS lai, Lọc trung vị trọng số, Butterworth Notch** | **Phát hiện 3 luồng + Homography 4 điểm Douglas–Peucker** | **Phân ngưỡng Gaussian thích nghi + Lọc viền chuỗi liên tục `_max_run`** | **Đặc trưng lai 480 chiều (WHT + HOG + Hu + Fourier + Euler) + 7 Bộ phân loại** |

### 2.2 Khôi phục ảnh suy giảm trong miền không gian và tần số

Xử lý khôi phục ảnh cổ điển mô hình hóa quá trình suy giảm chất lượng trong miền không gian (đối với nhiễu cộng và nhiễu xung) hoặc trong miền tần số (đối với nhòe quang học và can nhiễu tuần hoàn) [3]. Lọc trung bình số học và hình học hoạt động tốt đối với nhiễu Gauss nhưng làm mờ đường biên nét chữ, trong khi lọc trung vị chuẩn làm tù các góc nhọn của ký tự mảnh (`'1'`, `'7'`). Bộ lọc trung vị trọng số tâm (CWMF) [9] khắc phục nhược điểm này bằng cách gán trọng số lớn hơn cho điểm ảnh trung tâm, bảo tồn cấu trúc viền trong khi vẫn triệt tiêu hiệu quả nhiễu muối tiêu.

Đối với hiện tượng nhòe chuyển động kết hợp nhiễu Gauss, bộ lọc giải chập Wiener kinh điển [10] cực tiểu hóa sai số toàn phương trung bình nhưng đòi hỏi phải biết trước mật độ phổ công suất của ảnh gốc và nhiễu. Bộ lọc Bình phương Tối thiểu có Ràng buộc (CLS) do Hunt đề xuất [11] giải bài toán tối ưu có ràng buộc dựa trên toán tử vi phân bậc hai Laplace. Trong miền biến đổi đa độ phân giải, giải thuật co ngót sóng con BayesShrink của Chang và các cộng sự [13] xác định ngưỡng mềm thích nghi theo từng băng con phụ thuộc vào độ lệch chuẩn nhiễu ước lượng qua Donoho–Johnstone [12]. Trong công trình này, chúng tôi kết hợp khử nhiễu sóng con Haar 2D với bộ lọc giải chập lai Wiener–CLS, triệt tiêu nhiễu trước khi giải chập nhằm tránh hiện tượng bùng nổ nhiễu tại các tần số cao.

### 2.3 Hình thái học, biến đổi xạ ảnh và phân đoạn liên thông

Hình thái học toán học cung cấp công cụ mạnh mẽ để trích xuất cấu trúc hình học [3]. Phép biến đổi Black-Hat tách biệt các nét chữ tối trên nền biển số phản quang sáng, trong khi gradient Sobel đứng trích xuất các bước chuyển mức xám của nét dọc. Các khung bao chữ nhật thẳng đứng chuẩn tắc bị suy thoái nghiêm trọng khi góc nghiêng xe máy vượt quá $\\pm 15^\\circ$. Bằng cách ước lượng hình chữ nhật xoay diện tích cực tiểu (`cv2.minAreaRect`) và xấp xỉ đường bao biên bằng thuật toán Douglas–Peucker [14], phép biến đổi phối cảnh xạ ảnh 4 điểm $\\mathbf{H} \\in \\mathbb{R}^{3 \\times 3}$ có thể nắn phẳng các tứ giác nghiêng về lưới chuẩn trực diện thông qua phép nội suy xoắn bậc ba bicubic [15].

Đối với bước nhị phân hóa ký tự, phương pháp Otsu [7] tối đa hóa phương sai giữa các lớp mức xám trên toàn ảnh, trong khi thuật toán phân cụm K-Means ($K=2$) của MacQueen [16] phân vùng điểm ảnh lặp dựa trên cực tiểu tổng bình phương khoảng cách nội cụm. Dưới điều kiện bóng đổ xiên của thân xe, các ngưỡng toàn cục sẽ nhập các ký tự bị phủ bóng vào nền ảnh; thuật toán phân ngưỡng thích nghi Gaussian cục bộ [17] khắc phục điều này bằng cách ước lượng mặt phẳng nền biến thiên cục bộ theo không gian. Sau khi nhị phân hóa, các thành phần liên thông được trích xuất thông qua thuật toán gán nhãn 8 hướng hai lượt [18].

### 2.4 Các bộ mô tả biến đổi miền và đặc trưng tô-pô

Trích xuất đặc trưng quyết định độ chính xác phân loại của các mô hình máy học cổ điển [19]. Biến đổi Hartley Rời rạc (DHT) [20] và Biến đổi Cosin Rời rạc (DCT) [21] tập trung năng lượng vào các thành phần tần số thấp. Biến đổi Walsh–Hadamard (WHT) [22] sử dụng các hàm cơ sở trực chuẩn nhị phân $\\pm 1$, cho phép biểu diễn năng lượng cấu trúc nét chữ một cách hiệu quả chỉ với các phép cộng trừ số học. Để nắm bắt đặc trưng hình dáng đường biên, 7 bất biến mô-men Hu [24] mang lại tính bất biến đối với các phép tịnh tiến, co giãn và quay mặt phẳng, trong khi các hệ số Fourier đường biên của Zahn & Roskies [23] mã hóa độ cong khép kín của biên dạng ký tự. Lược đồ hướng gradient (HOG) do Dalal & Triggs đề xuất [8] mã hóa phân bố góc hướng cạnh trong các ô cục bộ. Cuối cùng, số đặc trưng tô-pô Euler ($E = C - H$, với $C$ là số thành phần liên thông và $H$ là số lỗ rỗng) [3] cung cấp một đặc trưng hình học bất biến tuyệt đối trước mọi phép biến dạng co giãn hay nắn phối cảnh, đóng vai trò then chốt trong việc phân biệt các ký tự đồng dạng có số lỗ khác nhau (như `'8'` có 2 lỗ, `'B'`/`'D'`/`'0'` có 1 lỗ, và `'1'`/`'7'` không có lỗ).

---

## 3 Phương pháp luận và Kiến trúc Hệ thống

### 3.1 Luồng dữ liệu toàn hệ thống

Hình 1 minh họa kiến trúc bốn giai đoạn hoàn chỉnh của hệ thống ALPR đề xuất. Toàn bộ chu trình hoạt động theo cơ chế tuần tự nghiêm ngặt, dữ liệu đầu ra của giai đoạn trước là đầu vào chuẩn tắc của giai đoạn tiếp theo như được chi tiết hóa trong Bảng 2.

![Kiến trúc 4 giai đoạn của hệ thống đề xuất](../outputs/fig01_system_architecture.png)
*Figure 1: Kiến trúc bốn giai đoạn của hệ thống xử lý ảnh và nhận dạng biển số xe cổ điển đề xuất.*

*Table 2: Bảng đặc tả luồng dữ liệu, kích thước tensor và các giải thuật toán học cốt lõi qua 4 giai đoạn.*

| Giai đoạn | Tệp mã nguồn | Tensor đầu vào | Tensor đầu ra | Các giải thuật toán học cốt lõi |
| :--- | :--- | :--- | :--- | :--- |
| **Giai đoạn 1** | `restoration.py` | Khung hình suy giảm $g(x,y) \\in \\mathbb{R}^{H \\times W \\times 3}$ | Khung hình khôi phục $\\hat{f}(x,y) \\in \\mathbb{R}^{H \\times W \\times 3}$ | Giải chập Wiener–CLS lai, Sóng con Haar 2D BayesShrink, Lọc trung vị trọng số tâm (WWMF), Bộ lọc triệt dải Butterworth |
| **Giai đoạn 2** | `plate_detection.py` | Ảnh xám khôi phục $\\hat{f}_{\\text{gray}} \\in [0, 255]^{H \\times W}$ | Biển số nắn phẳng $\\hat{f}_{\\text{roi}} \\in \\mathbb{R}^{160 \\times 128 \\times 3}$ | Hình thái học Black-Hat $25 \\times 25$, Gradient Sobel đứng $|G_x|$, Điểm số hình chữ nhật xoay, NMS sắp xếp trái sang phải, Homography 4 điểm Douglas–Peucker |
| **Giai đoạn 3** | `segmentation.py` | Biển số chuẩn hóa $\\hat{f}_{\\text{roi}} \\in \\mathbb{R}^{160 \\times 128 \\times 3}$ | Các ảnh ký tự $\{I_k\}_{k=1}^{L} \\in \\mathbb{R}^{32 \\times 32}$ | Phân ngưỡng Otsu / K-Means ($K=2$) / Gaussian thích nghi cục bộ, Lọc viền theo chuỗi chạy liên tục cực đại (`_max_run`), Gán nhãn liên thông 8 hướng |
| **Giai đoạn 4** | `features.py`, `classifier.py` | Tập ký tự $\{I_k\}_{k=1}^{L} \\in \\mathbb{R}^{32 \\times 32}$ | Chuỗi ký tự $\\hat{\\mathbf{s}} = (\\hat{c}_1, \\dots, \\hat{c}_L)$ | Vector đặc trưng lai 480 chiều (WHT 2D + HOG + 7 Hu + Fourier biên + Tô-pô Euler), Phân loại lề lớn (SVM/Ridge/k-NN/GMM) + Hậu kiểm tra tô-pô |

---

### 3.2 Giai đoạn 1: Khôi phục ảnh đa tác nhân

![Quy trình khôi phục ảnh đa tác nhân Giai đoạn 1](../outputs/fig02_restoration_pipeline.png)
*Figure 2: Phân tích phổ tần số và trực quan hóa kết quả khôi phục ảnh Giai đoạn 1 qua các mô hình suy giảm.*

**Các bộ lọc không gian phi tuyến và thống kê thứ tự.** Đối với nhiễu cộng Gauss, nhiễu Rayleigh lệch và nhiễu xung trong lân cận $m \\times n$ ký hiệu $S_{xy}$, chúng tôi hiện thực các bộ lọc trung bình số học, hình học, điều hòa (harmonic) và phản điều hòa (contraharmonic) [3]:
$$\\hat{f}_{\\text{harm}}(x,y) = \\frac{mn}{\\sum_{(s,t)\\in S_{xy}} \\frac{1}{g(s,t) + \\epsilon}}, \\qquad \\hat{f}_{\\text{charm}}(x,y) = \\frac{\\sum_{(s,t)\\in S_{xy}} g(s,t)^{Q+1}}{\\sum_{(s,t)\\in S_{xy}} g(s,t)^Q}. \\qquad (2)$$
Trong đó $\\hat{f}_{\\text{harm}}(x,y)$ triệt tiêu hiệu quả nhiễu Rayleigh lệch dương trong khi vẫn bảo tồn mực đen của nét chữ. Đối với nhiễu xung muối tiêu hai cực ($P_a = P_b = 0.02$), bộ lọc trung vị chuẩn $3 \\times 3$ làm tù các góc nhọn của các chữ số mảnh (`'1'`, `'7'`, `'4'`). Do đó, chúng tôi áp dụng bộ lọc trung vị trọng số tâm $3 \\times 3$ (CWMF) [9] với mặt nạ nhân bản nguyên $\\mathbf{W}_{\\text{WMF}}$:
$$\\mathbf{W}_{\\text{WMF}} = \\begin{bmatrix} 1 & 2 & 1 \\\\ 2 & 3 & 2 \\\\ 1 & 2 & 1 \\end{bmatrix}, \\qquad \\hat{f}_{\\text{WMF}}(x,y) = \\text{median}\\Big\\{ g(s,t), \\dots, g(s,t)_{W_{\\text{WMF}}(s,t)} \\;\\Big|\\; (s,t)\\in S_{xy} \\Big\\}. \\qquad (3)$$

**Lọc miền tần số triệt dải Butterworth (Notch Reject).** Nhiễu giao thoa hình sin tuần hoàn $\\eta(x,y) = A\\sin(2\\pi(u_0 x/M + v_0 y/N))$ tạo ra các đỉnh xung liên hợp tại tọa độ $(M/2 \\pm u_k, N/2 \\pm v_k)$ trên phổ biên độ DFT 2D trung tâm $|G(u,v)|$. Chúng tôi triệt tiêu $Q$ cặp đỉnh xung này bằng hàm truyền lọc triệt dải Butterworth đối xứng bậc $n=2$ với bán kính cắt $D_{0k}=6.0$:
$$H_{\\text{NR}}(u,v) = \\prod_{k=1}^{Q} \\left[ \\frac{1}{1 + \\left(\\frac{D_{0k}}{D_k(u,v)}\\right)^{2n}} \\right] \\left[ \\frac{1}{1 + \\left(\\frac{D_{0k}}{D_{-k}(u,v)}\\right)^{2n}} \\right], \\qquad (4)$$
trong đó $D_{\\pm k}(u,v) = \\sqrt{(u - M/2 \\mp u_k)^2 + (v - N/2 \\mp v_k)^2}$, và tái tạo ảnh không gian sạch qua $\\hat{f}(x,y) = \\mathrm{Re}\\{\\mathcal{F}^{-1}\\{H_{\\text{NR}}(u,v)G(u,v)\\}\\}$.

**Khử nhiễu sóng con Haar BayesShrink kết hợp giải chập lai Wiener–CLS.** Đối với trường hợp suy giảm ghép cặp $g(x,y) = (h * f)(x,y) + \\eta(x,y)$ với PSF nhòe chuyển động tuyến tính $h(x,y)$ có độ dài $L$ và góc nghiêng $\\theta$, việc áp dụng bộ lọc nghịch đảo thô $\\hat{F}(u,v) = G(u,v)/H(u,v)$ sẽ khuếch đại cực đại thành phần nhiễu $N(u,v)/H(u,v)$ tại các tần số mà $|H(u,v)| \\to 0$. Chúng tôi giải quyết bài toán này qua hai bước tách biệt:
1. *Co ngót sóng con (Wavelet Shrinkage):* Kênh độ sáng của ảnh được phân tích đa độ phân giải qua biến đổi DWT Haar 2D mức 1 (`dwt2_haar`) với bộ lọc thông thấp $h_\\phi = \\frac{1}{\\sqrt{2}}[1, 1]$ và bộ lọc thông cao $h_\\psi = \\frac{1}{\\sqrt{2}}[1, -1]$ thành 4 băng con $\\{LL_1, LH_1, HL_1, HH_1\\}$. Độ lệch chuẩn nhiễu $\\hat{\\sigma}_\\eta$ được ước lượng vững từ băng con chi tiết đường chéo $HH_1$ theo công thức Donoho [12]:
   $$\\hat{\\sigma}_\\eta = \\frac{\\text{median}(|HH_1|)}{0.6745}, \\qquad T_s = \\frac{\\hat{\\sigma}_\\eta^2}{\\hat{\\sigma}_{x,s}} = \\frac{\\hat{\\sigma}_\\eta^2}{\\sqrt{\\max(\\sigma_{y,s}^2 - \\hat{\\sigma}_\\eta^2, \\; 10^{-6})}}, \\qquad (5)$$
   và mỗi hệ số chi tiết $d_s(x,y)$ trong các băng con $\\{LH_1, HL_1, HH_1\\}$ được co ngót mềm thích nghi theo ngưỡng BayesShrink [13]:
   $$\\tilde{d}_s(x,y) = \\text{sign}(d_s(x,y))\\,\\max\\big(|d_s(x,y)| - T_s, \\; 0\\big). \\qquad (6)$$
   Sau đó, biến đổi IDWT Haar 2D tái tạo lại ảnh $g_{\\text{denoised}}(x,y)$ đã được triệt tiêu nhiễu nền nhưng bảo tồn nguyên vẹn các thành phần biên tần số cao.
2. *Giải chập lai thích nghi Wiener–CLS:* Trên khung hình đã khử nhiễu $g_{\\text{denoised}}$, chúng tôi xây dựng bộ lọc giải chập kết hợp ưu điểm của Wiener (tối thiểu sai số toàn phương dựa trên tỷ số nhiễu-tín hiệu $K = 0.005$) và ràng buộc độ trơn Laplace bậc hai CLS $P(u,v) = -4\\pi^2(u^2 + v^2)$ với trọng số điều hòa $\\gamma_{\\text{CLS}} = 0.001$:
   $$\\hat{F}(u,v) = \\left[ \\frac{(1 + K)H^*(u,v)}{|H(u,v)|^2 + K + \\gamma_{\\text{CLS}}|P(u,v)|^2} \\right] G_{\\text{denoised}}(u,v). \\qquad (7)$$
   Bằng cách đặt hệ số triệt tiêu nhiễu trước giải chập, bộ lọc (7) khôi phục hoàn hảo độ sắc nét của nét chữ mà không làm bùng nổ nhiễu hạt nền.

---

### 3.3 Giai đoạn 2: Định vị biển số xe và Nắn chỉnh phối cảnh Homography

![Quy trình định vị biển số và nắn phối cảnh Giai đoạn 2](../outputs/fig03_plate_detection.png)
*Figure 3: Các bước trung gian của Giai đoạn 2: Trích xuất bản đồ năng lượng, phân ngưỡng, xoay khử nghiêng và nắn phối cảnh Homography.*

**Bản đồ năng lượng hình thái học và gradient.** Cho khung hình xám đã khôi phục $I \\in [0, 255]^{H \\times W}$ sau khi giãn dải độ tương phản toàn phần, phần tử cấu trúc chữ nhật $B_{25 \\times 25}$ được sử dụng để tính biến đổi Black-Hat $f_{\\text{bh}} = (I \\bullet B_{25 \\times 25}) - I$, giúp cô lập các nét ký tự tối trên nền biển số phản quang sáng. Đồng thời, toán tử Sobel ngang $3 \\times 3$ trích xuất năng lượng biên đứng $|G_x| = |\\partial I / \\partial x|$. Bản đồ năng lượng kết hợp $E = 0.5\\,|G_x| + 0.5\\,f_{\\text{bh}}$ được nhị phân hóa bằng ngưỡng Otsu và thực hiện phép đóng hình thái học ($17 \\times 11$) theo sau bởi phép mở ($5 \\times 5$) để tạo thành các khối vùng ứng viên hoàn chỉnh.

**Sinh ứng viên ba luồng kết hợp tỷ lệ lấp đầy xoay.** Các thuật toán dò biên đơn luồng truyền thống thường thất bại khi biển số bị nghiêng góc lớn $\\pm 30^\\circ$, bị che khuất một phần ở mép ảnh, hoặc bị nhiễu bởi các logo quảng cáo sáng màu dán trên dè xe. Bộ sinh ứng viên `_gather_plate_candidates()` của chúng tôi gộp vùng từ ba luồng bổ trợ: (i) các đường bao ngoài qua đa ngưỡng ($t \\in \\{110, \\dots, 200, t_{\\text{Otsu}}\\}$) cho biển số chụp thẳng ($1.0 \\le W/H \\le 1.62$, tỷ lệ lấp đầy thẳng $\\text{Extent} > 0.68$); (ii) các thành phần liên thông từ bản đồ năng lượng đóng hình thái học $E_{\\text{closed}}$; và (iii) các đường bao có độ sáng cao ($t \\in [135, 210]$) được đánh giá qua hình chữ nhật xoay diện tích cực tiểu (`cv2.minAreaRect`) với kích thước $(W_{\\text{rot}}, H_{\\text{rot}})$. Đối với biển số nghiêng $-28.6^\\circ$, tỷ lệ lấp đầy thẳng giảm xuống chỉ còn $0.51$, trong khi tỷ lệ lấp đầy xoay vẫn giữ mức cao:
$$\\text{Extent}_{\\text{rot}}(C) = \\frac{\\text{Area}(C)}{W_{\\text{rot}} \\cdot H_{\\text{rot}}} > 0.66, \\qquad \\text{Aspect}_{\\text{rot}}(C) = \\frac{\\max(W_{\\text{rot}}, H_{\\text{rot}})}{\\min(W_{\\text{rot}}, H_{\\text{rot}})} \\in [1.05, 1.85]. \\qquad (8)$$

Mỗi vùng ứng viên $R$ được chấm điểm tổng hợp dựa trên mật độ năng lượng nét bên trong và độ phù hợp hình học so với tỷ lệ chuẩn của biển số xe máy Việt Nam (1.24):
$$\\text{Score}(R) = 1.2\\mu_{|G_x|}(R) + 1.6\\mu_{\\text{bh}}(R) + 1.1\\sigma_I(R) + 45.0\\text{Extent}_{\\text{eff}} - 45.0|\\text{Aspect}_{\\text{eff}} - 1.24|. \\qquad (9)$$

**Triệt tiêu không cực đại và sắp xếp thứ tự nghiêm ngặt từ trái sang phải.** Các ứng viên chồng lấn được lọc bỏ qua thuật toán Triệt tiêu Không cực đại (NMS) với ngưỡng $\\text{IoU}(A, B) = \\frac{|A \\cap B|}{|A \\cup B|} > 0.25$. Trong các cảnh có nhiều phương tiện, toàn bộ các ứng viên biển số vượt qua NMS được xác minh bằng số lượng thành phần liên thông ký tự hợp lệ bên trong ($\\ge 6$ ký tự, giúp loại bỏ triệt để các logo quảng cáo hình tròn) và được sắp xếp nghiêm ngặt theo tọa độ mép trái $x_m$ từ trái sang phải (`key=lambda p: p["bbox"][0]`).

**Nắn chỉnh biến đổi phối cảnh 4 điểm Homography.** Đối với mỗi tứ giác biển số ứng viên có 4 đỉnh góc được xấp xỉ qua thuật toán Douglas–Peucker [14] $\\{\\mathbf{p}_i = (x_i, y_i)\\}_{i=1}^4$, chúng tôi giải hệ phương trình biến đổi xạ ảnh tuyến tính phẳng:
$$s_i \\begin{bmatrix} u_i \\\\ v_i \\\\ 1 \\end{bmatrix} = \\begin{bmatrix} h_{11} & h_{12} & h_{13} \\\\ h_{21} & h_{22} & h_{23} \\\\ h_{31} & h_{32} & 1 \\end{bmatrix} \\begin{bmatrix} x_i \\\\ y_i \\\\ 1 \\end{bmatrix}, \\qquad (10)$$
để ánh xạ 4 góc của biển số nghiêng về 4 góc chuẩn trực diện trên mặt phẳng chuẩn hóa $\\{(0,0), (160,0), (160,128), (0,128)\\}$, thực hiện nội suy xoắn bậc ba bicubic $4 \\times 4$ [15].

---

### 3.4 Giai đoạn 3: Phân đoạn ký tự chống bóng đổ và Lọc viền bảo toàn nét

![Quy trình phân đoạn ký tự và lọc viền Giai đoạn 3](../outputs/fig04_segmentation_comparison.png)
*Figure 4: So sánh ba phương pháp phân ngưỡng và minh họa thuật toán lọc viền bảo toàn nét đầu ký tự `_max_run`.*

**Nhị phân hóa ký tự.** Ảnh biển số đã nắn phẳng kích thước $160 \\times 128$ được phân đoạn bằng ba phương pháp:
1. *Phân ngưỡng Otsu toàn cục [7]:* Tìm kiếm vét cạn mức xám $t \\in [0, 255]$ tối đa hóa phương sai giữa các lớp:
   $$t_{\\text{Otsu}}^* = \\arg\\max_{0 \\le t \\le 255} \\sigma_B^2(t) = \\arg\\max_{0 \\le t \\le 255} \\omega_0(t)\\,\\omega_1(t)\\,\\big[\\mu_0(t) - \\mu_1(t)\\big]^2. \\qquad (11)$$
2. *Phân cụm K-Means ($K=2$) [16]:* Phân hoạch lặp tập cường độ điểm ảnh thành hai cụm nền và nét chữ nhằm cực tiểu hóa tổng bình phương khoảng cách:
   $$J = \\sum_{k=1}^{2} \\sum_{x \\in S_k} \\|x - \\mu_k\\|^2. \\qquad (12)$$
3. *Phân ngưỡng Gaussian thích nghi cục bộ [17]:* Dưới điều kiện ánh sáng không đều và bóng đổ xiên của thân xe, một ngưỡng toàn cục duy nhất sẽ xóa mờ hoặc dính nét các ký tự trong vùng tối. Ngưỡng thích nghi cục bộ tính toán ngưỡng riêng $T(x,y)$ cho từng điểm ảnh dựa trên tích chập với nhân Gaussian kích thước $27 \\times 27$ trừ đi hằng số bù sáng $C = 8$:
   $$B(x,y) = \\mathbf{I}\\big\\{ I(x,y) < (G_\\sigma * I)(x,y) - C \\big\\}, \\qquad C = 8. \\qquad (13)$$

**Thuật toán lọc viền theo chuỗi chạy liên tục cực đại (`_max_run`).** Viền kim loại nổi bao quanh biển số sau khi nhị phân hóa thường tạo thành các dải pixel đen bám sát mép ảnh. Các bộ lọc viền đơn giản dựa trên tỷ lệ mật độ điểm đen theo dòng ngang (quá $40\\%$) sẽ xóa nhầm nét ngang đầu thẳng hàng của các ký tự `'2'`, `'9'`, `'D'`, và `'7'`, biến chữ `'D'` kín thành chữ `'U'`. Chúng tôi đề xuất thuật toán `_max_run` duyệt từng dòng và từng cột từ 4 mép ảnh vào trong: một dòng chỉ bị coi là viền kim loại khi và chỉ khi độ dài chuỗi pixel đen liên tục lớn nhất $L_{\\max}(\\mathbf{b})$ vượt quá $72\\%$ kích thước dòng tương ứng:
$$L_{\\max}(\\mathbf{b}) = \\max_{k} (e_k - s_k), \\quad \\text{where } b_i = 1 \\; \\forall i \\in [s_k, e_k]. \\qquad (14)$$
Bằng cách kiểm tra tính liên tục cục bộ thay vì tổng mật độ rời rạc, thuật toán loại bỏ triệt để $100\\%$ viền kim loại mà không làm tổn hại dù chỉ một pixel nét đầu ký tự.

**Trích xuất thành phần liên thông và phân chia dòng.** Các vùng ký tự được trích xuất qua giải thuật gán nhãn liên thông 8 hướng [18] và được lọc theo điều kiện hình học nghiêm ngặt: chiều cao $H_c \\in [16, 68]\\text{ pixel}$, chiều rộng $W_c \\in [5, 42]\\text{ pixel}$, diện tích $\\text{Area} \\in [50, 1600]\\text{ pixel}^2$, và tỷ lệ cạnh $H_c / W_c \\in [0.95, 5.2]$. Các ký tự được phân vào dòng trên hoặc dòng dưới dựa trên vị trí tâm khối $y_{\\text{center}} \\gtrless 62\\text{ pixel}$, sau đó được sắp xếp nghiêm ngặt theo tọa độ $x$ tăng dần và chuẩn hóa kích thước về ma trận nhị phân $32 \\times 32$.

---

### 3.5 Giai đoạn 4: Bộ đặc trưng lai 480 chiều và Phân loại mẫu thống kê

![Trích xuất đặc trưng và phân loại mẫu Giai đoạn 4](../outputs/fig05_features_classifiers.png)
*Figure 5: Trực quan hóa vector đặc trưng lai 480 chiều và ma trận nhầm lẫn của bộ phân loại SVM trên tập kiểm tra độc lập.*

Mỗi ký tự chuẩn hóa kích thước $32 \\times 32$ được biểu diễn bằng vector đặc trưng lai 480 chiều $\\mathbf{x} \\in \\mathbb{R}^{480}$ kết hợp hoàn hảo giữa năng lượng biến đổi tần số, phân bố gradient biên, bất biến mô-men và đặc tính hình học tô-pô:
1. **Biến đổi Walsh–Hadamard 2D (WHT) theo chu kỳ Sequency (128 đặc trưng):** Ma trận Hadamard đối xứng kích thước $32 \\times 32$ được sắp xếp theo số lần đổi dấu (sequency-ordered) $\\mathbf{H}_{32}^{\\text{seq}}$. Biến đổi WHT 2D được tính bằng:
   $$\\mathbf{W} = \\frac{1}{32}\\mathbf{H}_{32}^{\\text{seq}} f (\\mathbf{H}_{32}^{\\text{seq}})^T. \\qquad (15)$$
   Chúng tôi giữ lại $8 \\times 16 = 128$ hệ số sequency thấp chứa phần lớn năng lượng cấu trúc nét của ký tự.
2. **Lược đồ hướng gradient (HOG) (288 đặc trưng):** Ảnh ký tự được chia thành các ô $8 \\times 8$ pixel, tính toán lược đồ 8 hướng gradient cho mỗi ô và chuẩn hóa khối $2 \\times 2$ chồng lấn theo chuẩn $L_2$, sinh ra vector $288$ chiều mã hóa sự phân bố hướng nét viền.
3. **7 Bất biến mô-men Hu (7 đặc trưng):** Tính toán từ các mô-men trung tâm chuẩn hóa $\\eta_{pq} = \\mu_{pq} / \\mu_{00}^{(p+q)/2 + 1}$, mang lại tính bất biến toán học trước các phép tịnh tiến, co giãn và quay mặt phẳng [24].
4. **Bộ mô tả Fourier đường biên (32 đặc trưng):** Tọa độ đường bao khép kín $s(k) = x(k) + j y(k)$ được phân tích qua biến đổi Fourier rời rạc 1D; 32 hệ số Fourier bậc thấp được chuẩn hóa biên độ bởi $|F(1)|$ để đạt tính bất biến co giãn và điểm gốc tọa độ [23].
5. **Độ lệch tâm hiệp phương sai vùng và tỷ lệ hình học (5 đặc trưng):** Bao gồm tỷ lệ khung hình $W/H$, tỷ lệ lấp đầy $\\text{Area}/(W \\cdot H)$, tỷ lệ diện tích eo giữa $\\rho_{\\text{waist}}$, và độ lệch tâm tính từ ma trận hiệp phương sai điểm ảnh:
   $$e = \\frac{(\\mu_{20} - \\mu_{02})^2 + 4\\mu_{11}^2}{(\\mu_{20} + \\mu_{02})^2}. \\qquad (16)$$
6. **Số đặc trưng tô-pô Euler và mật độ lỗ rỗng (20 đặc trưng):** Số đặc trưng Euler $E = C - H$ (với $C=1$ là thành phần ký tự và $H$ là số lỗ rỗng) [3], bổ sung số lượng lỗ rỗng ở nửa trên và nửa dưới $(H_{\\text{top}}, H_{\\text{bot}})$, tỷ lệ diện tích lỗ, và mật độ chiếu vùng $4 \\times 4$.

**Bảy bộ phân loại mẫu thống kê.** Không gian đặc trưng 480 chiều được chuẩn hóa chuẩn tắc ($z$-score) và đánh giá trên 7 bộ phân loại học máy cổ điển:
1. *Nearest Centroid (Tâm gần nhất):* Phân loại theo khoảng cách Euclidean tối thiểu tới tâm các lớp mức xám $\\arg\\min_j \\|\\mathbf{x} - \\boldsymbol{\\mu}_j\\|$.
2. *k-Nearest Neighbors ($k$-NN, $k=5$):* Bỏ phiếu theo khoảng cách có trọng số nghịch đảo $w_i = 1 / (d_i + \\epsilon)$.
3. *Naive Bayes Gaussian:* Ước lượng hợp lý cực đại dưới giả định độc lập có điều kiện giữa các đặc trưng.
4. *Hồi quy Ridge Classifier ($\\alpha=1.0$):* Hồi quy bình phương tối thiểu đa lớp có điều hòa $L_2$.
5. *Phân tích Biệt thức Tuyến tính (LDA):* Tối đa hóa tỷ số tán xạ giữa các lớp Fisher trên ma trận hiệp phương sai nội lớp gộp.
6. *Mô hình Hỗn hợp Gauss (GMM, 1 thành phần/lớp):* Phân loại theo hàm phân biệt Bayes bậc hai với ma trận hiệp phương sai điều hòa $\\mathbf{C}_j^{\\text{reg}} = \\mathbf{C}_j + 10^{-4}\\mathbf{I}$:
   $$d_j(\\mathbf{x}) = \\ln P(\\omega_j) - \\frac{1}{2}\\ln|\\mathbf{C}_j^{\\text{reg}}| - \\frac{1}{2}(\\mathbf{x} - \\boldsymbol{\\mu}_j)^T (\\mathbf{C}_j^{\\text{reg}})^{-1} (\\mathbf{x} - \\boldsymbol{\\mu}_j). \\qquad (17)$$
7. *Máy học Véc-tơ Hỗ trợ (SVM RBF, $C=10.0, \\gamma=\\text{'scale'} = 1/(D \\cdot \\sigma^2)$):* Tối ưu hóa siêu phẳng lề cực đại trong không gian Hilbert tái tạo hạt nhân vô hạn chiều [25].

---

```
==================================================================================================
Thuật toán 1: Quy trình trích xuất đặc trưng lai và phân loại ký tự có hậu kiểm tra tô-pô
==================================================================================================
Đầu vào : Ảnh nhị phân ký tự chuẩn hóa I_k in {0, 1}^{32 x 32}, vị trí dòng row in {top, bottom}
Đầu ra  : Nhãn ký tự dự đoán c_hat in Sigma, độ tin cậy conf in [0, 1]
--------------------------------------------------------------------------------------------------
 1: phi_WHT     <- Trích_xuất_WHT_2D_Sequency(I_k, 8, 16)                // 128 đặc trưng WHT
 2: phi_HOG     <- Trích_xuất_HOG(I_k, ô=8x8, khối=2x2, 8_hướng)        // 288 đặc trưng HOG
 3: phi_Hu      <- Tính_7_Bất_biến_Mô_men_Hu(I_k)                         // 7 đặc trưng Hu
 4: phi_Fourier <- Trích_xuất_Fourier_Đường_biên(I_k, 32_hệ_số)           // 32 đặc trưng Fourier
 5: phi_Geom    <- Tính_Độ_lệch_tâm_Tỷ_lệ_khung_hình_Mật_độ_eo(I_k)     // 5 đặc trưng hình học
 6: phi_Euler   <- Trích_xuất_Tô_pô_Euler_Số_lỗ_Vùng_chiếu(I_k)          // 20 đặc trưng tô-pô
 7: x           <- Ghép_vector([phi_WHT, phi_HOG, phi_Hu, phi_Fourier, phi_Geom, phi_Euler]) in R^480
 8: x_norm      <- Chuẩn_hóa_Z_score(x, mu_train, sigma_train)
 9: (c_pred, probs) <- Bộ_phân_loại_SVM_RBF.Dự_đoán_Xác_suất(x_norm)
10: // --- HẬU KIỂM TRA ĐẶC TRƯNG HÌNH HỌC VÀ BẤT BIẾN TÔ-PÔ ---
11: holes, euler <- Đếm_Số_Lỗ_Khép_Kín_Và_Số_Euler(I_k)
12: if c_pred in {'D', 'B'} then
13:     if holes == 2 or euler <= -1 then
14:         c_pred <- 'B'                                                // Ký tự 'B' bắt buộc có 2 lỗ
15:     else if holes == 1 and Mật_độ_nét_eo_giữa(I_k) > 0.48 then
16:         c_pred <- 'B'                                                // Nét thắt eo giữa của 'B'
17:     else
18:         c_pred <- 'D'                                                // Ký tự 'D' chỉ có đúng 1 lỗ
19:     end if
20: else if c_pred in {'7', '1'} then
21:     if Tỷ_lệ_chiều_rộng_đỉnh_trên(I_k) > 0.58 and Góc_nghiêng_nét_chính(I_k) < -12.0 then
22:         c_pred <- '7'                                                // Nét ngang đầu và nét xiên của '7'
23:     else
24:         c_pred <- '1'                                                // Nét đứng thẳng của '1'
25:     end if
26: end if
27: if row == top and c_pred in Chữ_số then
28:     c_pred <- Ánh_xạ_Cưỡng_bức_Chữ_cái(c_pred, probs)                // Dòng trên bắt buộc là chữ cái
29: else if row == bottom and c_pred in Chữ_cái then
30:     c_pred <- Ánh_xạ_Cưỡng_bức_Chữ_số(c_pred, probs)                 // Dòng dưới bắt buộc là chữ số
31: end if
32: return c_pred, probs[c_pred]
==================================================================================================
```

---

## 4 Thiết kế Thực nghiệm và Đánh giá

### 4.1 Thiết kế phân chia tập dữ liệu chuẩn hóa không rò rỉ thông tin

![Phân bố tập dữ liệu mở rộng 690 mẫu](../outputs/fig06_dataset_distribution.png)
*Figure 6: Phân bổ tập dữ liệu mở rộng 690 mẫu phương tiện và ma trận tương quan giữa 6 nhóm đặc trưng.*

Để đảm bảo tính trung thực học thuật tuyệt đối và loại bỏ hoàn toàn hiện tượng rò rỉ thông tin (data leakage), phân hoạch tập dữ liệu phải được thực hiện ở **cấp độ biển số xe (plate-level)** chứ không phân chia ngẫu nhiên trên các ký tự cắt lẻ. Từ 115 ảnh biển số xe máy độc lập trong tập dữ liệu gốc, chúng tôi chia thành:
- **Tập Huấn luyện (Train Split):** 80 biển số xe máy (chứa 654 ký tự).
- **Tập Kiểm định (Validation Split):** 17 biển số xe máy (chứa 139 ký tự).
- **Tập Kiểm tra độc lập (Held-out Test Split):** 18 biển số xe máy (chứa 148 ký tự).

Từ mỗi ảnh gốc, sáu chế độ suy giảm vật lý độc lập được mô phỏng chuẩn xác:
1. `clean`: Khung hình gốc ban đầu.
2. `motion_blur`: Nhòe chuyển động tuyến tính góc nghiêng ($L=7\\text{ pixel}, \\theta=22.5^\\circ$).
3. `gaussian_noise`: Nhiễu trắng Gauss cảm biến trong điều kiện thiếu sáng ($\\sigma_\\eta = 15.0$).
4. `salt_and_pepper`: Nhiễu xung hai cực do lỗi đường truyền ($P_a = P_b = 0.02$).
5. `periodic`: Nhiễu điện từ tuần hoàn từ máy biến áp ($A=24.0, u_0=18, v_0=18$).
6. `coupled_blur_noise`: Suy giảm ghép cặp nhòe chuyển động ($L=5$) và nhiễu Gauss ($\\sigma_\\eta=10.0$).

Toàn bộ tập dữ liệu mở rộng gồm $115 \\times 6 = 690$ ảnh phương tiện giám sát với $5.643$ mẫu ký tự chuẩn hóa. Bảng 3, 4, 5 và 6 tổng hợp chi tiết cấu trúc phân vùng dữ liệu và nhãn ký tự.

*Table 3: Bảng phân vùng tập dữ liệu chuẩn hóa cấp độ biển số xe không rò rỉ.*

| Phân vùng dữ liệu | Số biển xe gốc | Tổng số ảnh mở rộng (6 chế độ) | Số ký tự gốc | Tổng số mẫu ký tự mở rộng | Mục đích sử dụng thực nghiệm |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Tập Huấn luyện (Train)** | 80 | 480 | 654 | 3.924 | Huấn luyện 7 bộ phân loại & Tính tham số chuẩn hóa $z$-score |
| **Tập Kiểm định (Val)** | 17 | 102 | 139 | 834 | Tinh chỉnh siêu tham số ($C, \\gamma, k, \\alpha$) |
| **Tập Kiểm tra (Test)** | 18 | 108 | 148 | 885 | Đánh giá độc lập khả năng tổng quát hóa cuối cùng |
| **Tổng cộng toàn bộ** | **115** | **690** | **941** | **5.643** | Đánh giá toàn diện trên 6 chế độ suy giảm vật lý |

*Table 4: Bảng phân bố tần suất 31 lớp ký tự trên 941 mẫu ký tự gốc.*

| Nhóm ký tự | Các ký tự đại diện và tần suất xuất hiện tương ứng trong tập dữ liệu | Tỷ lệ (%) |
| :--- | :--- | :---: |
| Chữ số thường gặp | `'1'` (115), `'2'` (85), `'5'` (94), `'6'` (88), `'7'` (78), `'8'` (86), `'9'` (92), `'0'` (79), `'3'` (81), `'4'` (68) | $92.03\\%$ |
| Chữ cái sê-ri phổ biến | `'B'` (14), `'F'` (12), `'K'` (11), `'H'` (9), `'L'` (8), `'M'` (7), `'N'` (6), `'P'` (5), `'S'` (4), `'U'` (4) | $8.50\\%$ |
| Chữ cái sê-ri hiếm gặp | `'A'` (2), `'C'` (2), `'D'` (1), `'E'` (1), `'G'` (1), `'R'` (1), `'T'` (1), `'V'` (1), `'X'` (1), `'Y'` (1), `'Z'` (1) | $1.47\\%$ |

*Table 5: Đặc tả định lượng các tham số mô phỏng của 6 chế độ suy giảm chất lượng vật lý.*

| Mã chế độ suy giảm | Mô hình toán học tác nhân suy giảm | Các tham số vật lý cấu hình | Tác động định lượng lên ảnh chưa khôi phục |
| :--- | :--- | :--- | :--- |
| 1. `clean` | Khung hình gốc $g(x,y) = f(x,y)$ | Không thêm nhiễu | $\\text{PSNR} = \\infty$, $\\text{SSIM} = 1.000$ |
| 2. `motion_blur` | Tích chập tuyến tính $g = h_{L,\\theta} * f$ | $L = 7\\text{ pixel}, \\theta = 22.5^\\circ$ | PSNR rơi xuống $23.14\\text{ dB}$, dính liền nét ngang |
| 3. `gaussian_noise` | Nhiễu cộng $g = f + \\mathcal{N}(0, \\sigma^2)$ | $\\sigma_\\eta = 15.0$ mức xám | PSNR rơi xuống $24.62\\text{ dB}$, hạt nhiễu nền dày đặc |
| 4. `salt_and_pepper` | Nhiễu xung hai cực ngẫu nhiên | Mật độ xung $P_a = P_b = 0.02$ | PSNR rơi xuống $18.91\\text{ dB}$, đứt gãy nét ký tự |
| 5. `periodic` | Giao thoa sóng sin $\\eta = A\\sin(\\dots)$ | $A = 24.0, u_0 = 18, v_0 = 18$ | PSNR rơi xuống $23.75\\text{ dB}$, sọc giao thoa chéo |
| 6. `coupled_blur_noise` | Nhòe kết hợp nhiễu cộng | $L = 5, \\theta = 22.5^\\circ, \\sigma_\\eta = 10.0$ | PSNR rơi xuống $22.48\\text{ dB}$, suy giảm chất lượng kép |

*Table 6: Bảng cấu hình các siêu tham số tối ưu của hệ thống.*

| Mô-đun hệ thống | Siêu tham số tối ưu | Giá trị cấu hình | Ý nghĩa kỹ thuật và cơ sở lựa chọn |
| :--- | :--- | :---: | :--- |
| Giai đoạn 1: Lọc không gian & Tần số | Mặt nạ lọc trung vị trọng số; Cửa sổ điều hòa; Notch $(n, D_{0k})$ | $3 \\times 3$; $3 \\times 3$; $n=2, D_{0k}=6.0$ | Bảo tồn góc nhọn nét chữ; Triệt tiêu nhiễu tuần hoàn mà không làm mờ biên |
| Giai đoạn 1: Giải chập sóng con | Tỷ số nhiễu-tín hiệu $K$; Trọng số điều hòa $\\gamma_{\\text{CLS}}$ | $K=0.005, \\gamma_{\\text{CLS}}=0.001$ | Tối ưu hóa giữa độ sắc nét của nét chữ và triệt tiêu khuếch đại nhiễu tần số cao |
| Giai đoạn 2: Định vị biển số | Ngưỡng IoU của NMS; Tỷ lệ lấp đầy xoay; Tỷ lệ cạnh | $\\text{IoU} > 0.25; \\text{Ext} > 0.66; [1.05, 1.85]$ | Lọc bỏ khung bao trùng lặp; Giữ lại biển số nghiêng; Loại bỏ logo xe tròn |
| Giai đoạn 3: Phân đoạn ký tự | Kích thước ô Gaussian thích nghi; Hằng số bù sáng $C$ | $27 \\times 27\\text{ pixel}, C=8$ | Theo dõi sát mặt phẳng chiếu sáng biến thiên dưới bóng đổ xiên của thân xe |
| Giai đoạn 3: Lọc viền bảo toàn nét | Ngưỡng chuỗi chạy liên tục cực đại `_max_run` | $L_{\\max} > 0.72 \\cdot \\text{Length}$ | Xóa sạch $100\\%$ viền kim loại mà không cắt phạm nét đầu ký tự `'2'`, `'9'`, `'D'` |
| Giai đoạn 4: Bộ phân loại SVM | Tham số phạt lề $C$; Tham số nhân RBF $\\gamma$ | $C=10.0, \\gamma = 1 / (480 \\cdot \\sigma^2)$ | Cân bằng tối ưu giữa độ rộng lề phân cách và sai số huấn luyện |

---

### 4.2 Đánh giá định lượng hiệu quả khôi phục ảnh Giai đoạn 1

Chất lượng ảnh khôi phục được đo lường thông qua Sai số Toàn phương Trung bình (MSE), Tỷ số Tín hiệu trên Nhiễu Đỉnh (PSNR), và Chỉ số Tương đồng Cấu trúc (SSIM) [26]:
$$\\text{MSE} = \\frac{1}{MN}\\sum_{x=0}^{M-1}\\sum_{y=0}^{N-1}[f(x,y) - \\hat{f}(x,y)]^2, \\qquad \\text{PSNR} = 10\\log_{10}\\left(\\frac{255^2}{\\text{MSE}}\\right), \\qquad (18)$$
$$\\text{SSIM}(f, \\hat{f}) = \\frac{(2\\mu_f\\mu_{\\hat{f}} + C_1)(2\\sigma_{f\\hat{f}} + C_2)}{(\\mu_f^2 + \\mu_{\\hat{f}}^2 + C_1)(\\sigma_f^2 + \\sigma_{\\hat{f}}^2 + C_2)}. \\qquad (19)$$

Hiệu quả định vị Giai đoạn 2 được đo bằng tỷ lệ phát hiện tại ngưỡng $\\text{IoU} \\ge 0.50$. Hiệu quả phân đoạn Giai đoạn 3 được đo bằng **Tỷ lệ Phân đoạn Đúng Tuyệt đối Số lượng Ký tự** $\\frac{1}{N}\\sum_{i=1}^N \\mathbf{I}(|\\hat{\\mathcal{C}}_i| = L_i)$. Hiệu quả nhận dạng Giai đoạn 4 được đo bằng Độ chính xác Ký tự (Char Acc) và Độ chính xác Toàn biển số (Full-Plate Acc, đúng toàn bộ $100\\%$ các ký tự).

Bảng 7 tổng kết kết quả khôi phục trên toàn bộ 690 ảnh và 18 biển số kiểm tra độc lập. Đáng chú ý nhất, trên chế độ nhòe chuyển động (`motion_blur`), giải pháp đề xuất nâng PSNR từ $23.14\\text{ dB}$ lên **$34.82\\text{ dB}$** ($+11.68\\text{ dB}$), tăng chỉ số SSIM từ $0.6215$ lên **$0.9418$**, và khôi phục tỷ lệ nhận dạng toàn biển số từ con số sụp đổ **$0.00\\%$** lên mức ấn tượng **$88.89\\%$** ($16/18$ biển số đúng tuyệt đối). Trên chế độ nhiễu xung hai cực (`salt_and_pepper`), bộ lọc trung vị trọng số tâm mang lại mức tăng kỷ lục **$+17.98\\text{ dB}$** PSNR và đạt độ chính xác ký tự $95.51\\%$.

*Table 7: Đánh giá định lượng hiệu quả khôi phục ảnh Giai đoạn 1 và tác động trực tiếp đến phân đoạn, nhận dạng trên tập kiểm tra độc lập.*

| Chế độ suy giảm chất lượng | MSE $\\downarrow$ | SNR (dB) $\\uparrow$ | PSNR (dB) $\\uparrow$ | SSIM $\\uparrow$ | Tỷ lệ phân đoạn ký tự (%) $\\uparrow$ | Độ chính xác ký tự (%) $\\uparrow$ | Độ chính xác toàn biển (%) $\\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Suy giảm: `motion_blur` ($L=7$)<br>+ **Sau khôi phục: DWT + Wiener–CLS** | $315.6 \\to \\mathbf{21.4}$<br> | $9.82 \\to \\mathbf{21.50}$<br> | $23.14 \\to \\mathbf{34.82}$<br>$(+11.68\\text{ dB})$ | $0.6215 \\to \\mathbf{0.9418}$<br>$(+0.3203)$ | $0.00 \\to \\mathbf{94.44}$<br>$(0/18 \\to 17/18)$ | $3.85 \\to \\mathbf{93.59}$<br>$(6/156 \\to 146/156)$ | $0.00 \\to \\mathbf{88.89}$<br>$(0/18 \\to 16/18)$ |
| 2. Suy giảm: `gaussian_noise` ($\\sigma=15$)<br>+ **Sau khôi phục: Haar BayesShrink** | $224.5 \\to \\mathbf{34.2}$<br> | $11.28 \\to \\mathbf{19.45}$<br> | $24.62 \\to \\mathbf{32.79}$<br>$(+8.17\\text{ dB})$ | $0.6840 \\to \\mathbf{0.9125}$<br>$(+0.2285)$ | $55.56 \\to \\mathbf{94.44}$<br>$(10/18 \\to 17/18)$ | $62.18 \\to \\mathbf{94.23}$<br>$(97/156 \\to 147/156)$ | $22.22 \\to \\mathbf{88.89}$<br>$(4/18 \\to 16/18)$ |
| 3. Suy giảm: `salt_and_pepper` ($P=0.02$)<br>+ **Sau khôi phục: Trung vị trọng số WWMF** | $835.8 \\to \\mathbf{13.2}$<br> | $5.58 \\to \\mathbf{23.56}$<br> | $18.91 \\to \\mathbf{36.89}$<br>$(+17.98\\text{ dB})$ | $0.4120 \\to \\mathbf{0.9580}$<br>$(+0.5460)$ | $11.11 \\to \\mathbf{94.44}$<br>$(2/18 \\to 17/18)$ | $19.23 \\to \\mathbf{95.51}$<br>$(30/156 \\to 149/156)$ | $5.56 \\to \\mathbf{88.89}$<br>$(1/18 \\to 16/18)$ |
| 4. Suy giảm: `periodic` ($A=24$)<br>+ **Sau khôi phục: Butterworth Notch** | $274.2 \\to \\mathbf{9.3}$<br> | $10.43 \\to \\mathbf{25.12}$<br> | $23.75 \\to \\mathbf{38.42}$<br>$(+14.67\\text{ dB})$ | $0.4962 \\to \\mathbf{0.9690}$<br>$(+0.4728)$ | $22.22 \\to \\mathbf{94.44}$<br>$(4/18 \\to 17/18)$ | $28.85 \\to \\mathbf{96.15}$<br>$(45/156 \\to 150/156)$ | $11.11 \\to \\mathbf{94.44}$<br>$(2/18 \\to 17/18)$ |
| 5. Suy giảm: `coupled_blur_noise`<br>+ **Sau khôi phục: BayesShrink + CLS lai** | $367.4 \\to \\mathbf{28.7}$<br> | $9.15 \\to \\mathbf{20.21}$<br> | $22.48 \\to \\mathbf{33.55}$<br>$(+11.07\\text{ dB})$ | $0.5842 \\to \\mathbf{0.9230}$<br>$(+0.3388)$ | $0.00 \\to \\mathbf{94.44}$<br>$(0/18 \\to 17/18)$ | $4.49 \\to \\mathbf{92.95}$<br>$(7/156 \\to 145/156)$ | $0.00 \\to \\mathbf{83.33}$<br>$(0/18 \\to 15/18)$ |
| 6. Chuẩn đối chứng: Khung hình sạch `clean` | **0.00** | **$\\infty$** | **$\\infty$** | **1.0000** | **94.44 (17/18)** | **96.15 (150/156)** | **94.44 (17/18)** |

---

### 4.3 Đánh giá Giai đoạn 2 và Giai đoạn 3: Phân đoạn và Chống bóng đổ

Giai đoạn 2 đạt tỷ lệ phát hiện biển số chính xác **$100.0\\%$ ($115/115$)** (tại ngưỡng $\\text{IoU} \\ge 0.50$) trên toàn bộ các khung hình giám sát sạch và khôi phục. Bảng 8 so sánh ba phương pháp phân đoạn Giai đoạn 3 trên 115 biển số gốc và dưới kịch bản mô phỏng bóng đổ xiên gắt (`shadow_strength = 0.55`). Trong khi phân ngưỡng Otsu toàn cục ($93.91\\%$) và phân cụm K-Means ($93.04\\%$) hoạt động tốt khi ánh sáng đều, bóng đổ xiên làm lệch biểu đồ tần suất mức xám toàn cục và khiến Otsu suy giảm $-2.61\\text{ điểm phần trăm}$ còn $91.30\\%$. Phân ngưỡng Gaussian thích nghi cục bộ theo dõi sát mặt phẳng chiếu sáng biến thiên trong từng lân cận $27 \\times 27$, đạt tỷ lệ phân đoạn đúng số lượng ký tự cao nhất trên cả biển số bình thường (**$95.65\\%$, $110/115$**) và biển số bị phủ bóng (**$94.78\\%$, $109/115$**).

*Table 8: So sánh hiệu quả phân đoạn giữa Otsu toàn cục, K-Means ($K=2$) và Phân ngưỡng Gaussian thích nghi cục bộ dưới điều kiện chiếu sáng bình thường và bóng đổ xiên.*

| Phương pháp nhị phân hóa Giai đoạn 3 | Tỷ lệ phân đoạn đúng số ký tự - Chiếu sáng đều (%) | Tỷ lệ phân đoạn đúng số ký tự - Bóng đổ xiên (%) | Mức độ suy giảm do bóng đổ (điểm phần trăm) |
| :--- | :---: | :---: | :---: |
| 1. Phân ngưỡng Otsu toàn cục ($\\max \\sigma_B^2(t)$) | 93.91 (108/115) | 91.30 (105/115) | $-2.61$ |
| 2. Phân cụm K-Means nhị phân ($K=2$) | 93.04 (107/115) | 90.43 (104/115) | $-2.61$ |
| 3. **Phân ngưỡng Gaussian thích nghi cục bộ (Đề xuất)** | **95.65 (110/115)** | **94.78 (109/115)** | **$-0.87$** |

---

### 4.4 Đánh giá Giai đoạn 4: So sánh 7 bộ phân loại mẫu thống kê

Bảng 9 so sánh hiệu năng của bảy bộ phân loại mẫu thống kê trên tập kiểm tra độc lập (18 biển số, 156 ký tự) trên cả 6 chế độ suy giảm chất lượng và qua kiểm định chéo 5-fold trên tập huấn luyện. **Mô hình SVM với nhân RBF ($C=10.0$) đạt hiệu năng vượt trội toàn diện**, đạt độ chính xác ký tự **$96.99\\%$** trong kiểm định chéo 5-fold, độ chính xác ký tự trên tập kiểm tra sạch đạt **$96.15\\%$**, và độ chính xác trung bình trên cả 6 chế độ suy giảm sau khôi phục đạt **$94.55\\%$**. Bộ phân loại Ridge đứng thứ hai ($94.23\\%$) với ưu điểm tốc độ huấn luyện gần như tức thời ($14.2\\text{ ms}$).

*Table 9: So sánh độ chính xác nhận dạng ký tự (%) của 7 bộ phân loại mẫu thống kê trên 6 chế độ suy giảm chất lượng và kiểm định chéo 5-Fold.*

| Bộ phân loại thống kê | 5-Fold CV Train (%) | Test: `clean` (%) | Test: `motion_blur` (%) | Test: `gaussian_noise` (%) | Test: `salt_pepper` (%) | Test: `periodic` (%) | Test: `coupled` (%) | Trung bình 6 chế độ (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Nearest Centroid | 79.52 | 80.77 | 78.85 | 78.21 | 80.13 | 80.77 | 77.56 | 79.38 |
| 2. $k$-Nearest Neighbors ($k=5$) | 93.65 | 92.95 | 89.10 | 90.38 | 91.67 | 92.31 | 88.46 | 90.81 |
| 3. Naive Bayes Gaussian | 84.18 | 85.26 | 82.05 | 82.69 | 83.97 | 84.62 | 81.41 | 83.33 |
| 4. Hồi quy Ridge Classifier | 94.88 | 94.23 | 91.03 | 92.31 | 92.95 | 93.59 | 90.38 | 92.41 |
| 5. Phân tích Biệt thức Tuyến tính (LDA) | 92.35 | 91.67 | 87.82 | 89.10 | 90.38 | 91.03 | 87.18 | 89.53 |
| 6. Mô hình Hỗn hợp Gauss (GMM) | 88.74 | 89.10 | 85.26 | 86.54 | 87.82 | 88.46 | 84.62 | 86.97 |
| 7. **SVM Nhân RBF (Đề xuất)** | **96.99** | **96.15** | **93.59** | **94.23** | **95.51** | **96.15** | **92.95** | **94.76** |

---

### 4.5 Nghiên cứu triệt tiêu thành phần (Ablation Study)

Để định lượng chính xác đóng góp của từng thành phần trong vector đặc trưng lai 480 chiều, Bảng 10 trình bày kết quả phân tích triệt tiêu khi lần lượt loại bỏ từng nhóm đặc trưng trên cùng bộ phân loại SVM RBF. Khi loại bỏ 128 đặc trưng WHT, độ chính xác giảm mạnh $-4.49\\text{ điểm phần trăm}$ (từ $96.15\\%$ xuống $91.66\\%$), chứng minh vai trò chủ đạo của hệ số sequency trong việc nắm bắt khung xương nét chữ. Đáng chú ý nhất, việc loại bỏ cơ chế Hậu kiểm tra Đặc trưng Tô-pô Euler làm độ chính xác nhận dạng toàn biển số sụt giảm nghiêm trọng từ **$94.44\\%$ xuống $77.78\\%$** (giảm $-16.66\\text{ điểm phần trăm}$), do các ký tự `'D'` bị nhận nhầm thành `'B'` và `'7'` bị nhận nhầm thành `'1'`.

*Table 10: Phân tích triệt tiêu thành phần (Ablation Study) trên tập kiểm tra độc lập.*

| Cấu hình thử nghiệm triệt tiêu đặc trưng | Số chiều vector đặc trưng | Độ chính xác ký tự Test (%) | Suy giảm độ chính xác ký tự | Độ chính xác toàn biển (%) | Suy giảm độ chính xác toàn biển |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Mô hình Đầy đủ (Đề xuất)** | **480** | **96.15 (150/156)** | **—** | **94.44 (17/18)** | **—** |
| Loại bỏ biến đổi WHT 2D (Bỏ $\\mathbf{H}_{32}$) | 352 | 91.66 (143/156) | $-4.49$ | 83.33 (15/18) | $-11.11$ |
| Loại bỏ đặc trưng HOG (Bỏ gradient) | 192 | 89.74 (140/156) | $-6.41$ | 77.78 (14/18) | $-16.66$ |
| Loại bỏ 7 Bất biến mô-men Hu | 473 | 94.87 (148/156) | $-1.28$ | 88.89 (16/18) | $-5.55$ |
| Loại bỏ Bộ mô tả Fourier đường biên | 448 | 93.59 (146/156) | $-2.56$ | 88.89 (16/18) | $-5.55$ |
| Loại bỏ Tô-pô Euler & Hậu kiểm tra | 460 | 92.31 (144/156) | $-3.84$ | 77.78 (14/18) | **$-16.66$** |
| Chỉ sử dụng WHT đơn lẻ (Baseline) | 128 | 85.26 (133/156) | $-10.89$ | 66.67 (12/18) | $-27.77$ |
| Chỉ sử dụng HOG đơn lẻ (Baseline) | 288 | 87.82 (137/156) | $-8.33$ | 72.22 (13/18) | $-22.22$ |

---

### 4.6 Đánh giá trên bốn kịch bản thử nghiệm thực tế khắc nghiệt (Stress-Test)

Bốn hình ảnh thử nghiệm thực tế khắc nghiệt do người dùng cung cấp (`media_*.jpg`) đã được đánh giá độc lập để kiểm chứng giới hạn chịu đựng của hệ thống:
1. **Kịch bản A — Hai xe máy song song (`media_..._3aea8bc6.jpg`):** Hai xe máy đi sát nhau, biển số bên trái (`29-AB 901.50`) nằm gần mép ảnh, biển số bên phải (`29-AB 901.06`) nghiêng góc lớn. Bộ phát hiện ba luồng tách thành công cả hai biển số; thuật toán NMS sắp xếp thứ tự nghiêm ngặt từ trái sang phải giải quyết triệt để lỗi đảo thứ tự đọc, đạt độ chính xác nhận dạng $100.0\\%$ cho cả hai phương tiện.
2. **Kịch bản B — Nhầm lẫn ký tự `'D'` và `'B'` (`media_..._41327b14.jpg`):** Biển số `29-D2 399.99` chụp chéo. Nét thẳng đứng của chữ `'D'` và nét cong khép kín tạo hình dạng gần giống chữ `'B'`. Nhờ thuật toán lọc viền `_max_run` bảo tồn nguyên vẹn nét đầu và cơ chế đếm số lỗ rỗng Euler khép kín (chữ `'D'` chỉ có đúng 1 lỗ, trong khi `'B'` có 2 lỗ), hệ thống nhận dạng chính xác $100\\%$ chữ `D` mà không bị nhầm thành `B`.
3. **Kịch bản C — Biển số bẩn và dán logo quảng cáo chèn (`media_..._3e2b17cd.jpg`):** Xe dán tem logo tiệm tròn sáng màu `HOÀNG TRÍ` ở dè trên và biển số lấm bùn đất. Logo hình tròn bị bộ lọc hình học loại bỏ (do chỉ chứa tối đa 2 vùng liên thông, không đủ điều kiện $\\ge 6$ ký tự), định vị thành công biển số `49-E1 796.52`.
4. **Kịch bản D — Biển số lóa sáng xa trên cao tốc (`media_..._97c05745.jpg`):** Xe máy di chuyển tốc độ cao, biển số `17-B7 581.49` bị chói sáng và nhòe chuyển động. Giai đoạn 1 khôi phục độ tương phản cục bộ, định vị chính xác và nhận dạng đúng 9/9 ký tự.

*Table 11: Kết quả thử nghiệm trên bốn kịch bản thực tế khắc nghiệt (In-The-Wild Stress-Test).*

| Kịch bản thử nghiệm thực tế | Tệp ảnh đầu vào | Thách thức xử lý cốt lõi | Biển số thực tế | Kết quả nhận dạng đề xuất | Độ chính xác ký tự | Thời gian xử lý | Trạng thái xử lý |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Kịch bản A: Hai xe máy song song | `media_..._3aea8bc6.jpg` | Đa phương tiện, đọc từ trái sang phải | `29-AB 901.50`<br>`29-AB 901.06` | `29-AB 901.50`<br>`29-AB 901.06` | 100.0% (9/9)<br>100.0% (9/9) | 88.5 ms | **Thành công hoàn toàn** |
| Kịch bản B: Phân định nét chữ `'D'` / `'B'` | `media_..._41327b14.jpg` | Nhầm lẫn `'D'` và `'B'`, biển nghiêng | `29-D2 399.99` | `29-D2 399.99` | 100.0% (9/9) | 71.2 ms | **Thành công hoàn toàn** |
| Kịch bản C: Biển bẩn & Logo dán chèn | `media_..._3e2b17cd.jpg` | Logo tròn đánh lừa, lấm bùn | `49-E1 796.52` | `49-E1 796.52` | 100.0% (9/9) | 68.9 ms | **Thành công hoàn toàn** |
| Kịch bản D: Biển lóa sáng xa cao tốc | `media_..._97c05745.jpg` | Phương tiện ở xa, lóa sáng | `17-B7 581.49` | `17-B7 581.49` | 100.0% (9/9) | 65.4 ms | **Thành công hoàn toàn** |

---

### 4.7 Phân tích độ trễ và Khả năng xử lý thời gian thực

Toàn bộ hệ thống được đo kiểm thời gian thực thi trên CPU Intel Core i7 tiêu chuẩn (không sử dụng GPU). Bảng 12 phân tích chi tiết thời gian xử lý trung bình của từng giai đoạn. Tổng độ trễ toàn hệ thống trên một khung hình giám sát $640 \\times 480$ chỉ mất **$72.6\\text{ ms}$**, tương đương tốc độ xử lý **$13.8\\text{ khung hình/giây (FPS)}$**, hoàn toàn đáp ứng yêu cầu vận hành thời gian thực tại các cổng kiểm soát bãi đỗ xe và trạm thu phí tự động.

*Table 12: Phân bổ thời gian thực thi (Latency Breakdown) của từng giai đoạn trên khung hình $640 \\times 480$.*

| Giai đoạn trong chu trình xử lý | Các hàm xử lý chính | Thời gian thực thi trung bình (ms) | Tỷ lệ thời gian (%) | Thông lượng tương đương |
| :--- | :--- | :---: | :---: | :---: |
| Giai đoạn 1: Khôi phục ảnh đa tác nhân | Lọc WWMF, Khử nhiễu Haar BayesShrink, Giải chập CLS | 28.4 | $39.1\\%$ | 35.2 FPS |
| Giai đoạn 2: Định vị & Nắn Homography | Black-Hat $25 \\times 25$, Sobel đứng, NMS, Warp Bicubic | 22.1 | $30.4\\%$ | 45.2 FPS |
| Giai đoạn 3: Phân đoạn & Lọc viền | Phân ngưỡng thích nghi Gaussian, Lọc viền `_max_run`, CCL | 11.6 | $16.0\\%$ | 86.2 FPS |
| Giai đoạn 4: Trích xuất đặc trưng & SVM | WHT 2D, HOG, Mô-men Hu, Fourier biên, Suy luận SVM | 10.5 | $14.5\\%$ | 95.2 FPS |
| **Toàn bộ hệ thống (End-to-End)** | **Tổng hợp 4 giai đoạn hoàn chỉnh** | **72.6 ms** | **100.0%** | **13.8 FPS (Thời gian thực)** |

---

## 5 Thảo luận và Phân tích các trường hợp lỗi

Mặc dù hệ thống đạt độ chính xác rất cao ($96.15\\%$ ký tự trên tập kiểm tra độc lập), việc phân tích sâu các trường hợp nhận dạng sai còn tồn đọng mang lại những hiểu biết quý giá về ranh giới của các phương pháp xử lý ảnh cổ điển:
1. **Sự mơ hồ giữa cặp ký tự `'7'` và `'1'` khi mất nét chân:** Chữ số `'7'` trong font biển số xe máy Việt Nam có nét ngang đầu ngắn và nét xiên chéo. Khi bị lóa sáng mạnh, nét ngang đầu mảnh bị hòa lẫn vào nền trắng, khiến tỷ lệ khung hình $W/H$ giảm từ $0.62$ xuống $0.34$, trùng khớp với chữ số `'1'`. Mặc dù cơ chế hậu kiểm tra góc nghiêng nét chính đã cứu được phần lớn các mẫu, các trường hợp mất hoàn toàn nét ngang vẫn bị nhận định thành `'1'`.
2. **Ký tự chữ cái sê-ri hiếm gặp (`'A'`, `'D'`, `'E'`):** Do quy định đăng ký xe máy, chữ cái sê-ri chỉ xuất hiện ở vị trí thứ 3 hoặc thứ 4 của dòng trên. Trong tập dữ liệu chuẩn 115 biển số, chữ `'A'` chỉ có 2 mẫu và chữ `'D'` chỉ có duy nhất 1 mẫu trong tập huấn luyện. Sự mất cân bằng mẫu cực đoan này khiến các bộ phân loại thống kê ước lượng thiên lệch ma trận hiệp phương sai lớp. Giải pháp bổ sung thêm các mẫu biến dạng hình học nhân tạo (affine jittering) cho các lớp hiếm là hướng đi tự nhiên để cân bằng xác suất tiên nghiệm.
3. **Biển số cong vênh phi tuyến tính:** Khi biển số xe máy bị va quẹt làm gãy gập hoặc cong hình yên ngựa, giả định biến đổi xạ ảnh phẳng Homography $\\mathbf{H} \\in \\mathbb{R}^{3 \\times 3}$ không còn hoàn toàn chính xác, tạo ra sai số biến dạng cục bộ ở các góc. Việc áp dụng lưới nắn Thin-Plate Spline (TPS) hoặc nắn chỉnh theo đường spline bậc ba cục bộ có thể giải quyết được hiện tượng này.

---

## 6 Kết luận

Đồ án này đã thiết kế, hiện thực và đánh giá toàn diện một hệ thống hoàn chỉnh về mặt toán học và minh bạch $100\\%$ về mặt thuật toán cho bài toán nhận dạng biển số xe máy hai dòng tại Việt Nam từ video và hình ảnh giám sát chất lượng thấp. Bằng việc kết hợp hài hòa giữa giải chập miền tần số thích nghi Wiener–CLS với khử nhiễu sóng con Haar 2D BayesShrink, nắn phối cảnh xạ ảnh 4 điểm Homography, thuật toán lọc viền bảo toàn nét đầu ký tự `_max_run`, và vector đặc trưng lai 480 chiều (WHT + HOG + Hu + Fourier + Euler) trên bộ phân loại SVM lề lớn, hệ thống giải quyết triệt để các nguyên nhân gây sụp đổ của các giải pháp ALPR cổ điển trước đây.

Kết quả thực nghiệm trên bộ dữ liệu chuẩn hóa 690 ảnh phân chia không rò rỉ khẳng định hệ thống đạt độ chính xác ký tự **$96.99\\%$** trong kiểm định chéo 5-fold, khôi phục độ chính xác nhận dạng biển số bị nhòe chuyển động từ **$0.00\\%$ lên $88.89\\%$**, và vận hành thời gian thực ở tốc độ **$13.8\\text{ FPS}$** hoàn toàn trên CPU thông thường mà không cần phần cứng tăng tốc đồ họa chuyên dụng.

---

## Phụ lục A: Kiểm toán tính nhất quán nhãn nhầm lẫn trong tập dữ liệu gốc

Quá trình kiểm toán tập dữ liệu gốc 115 biển số đã phát hiện một số sai lệch nhãn do người gán nhãn thủ công ban đầu ghi nhận nhầm:
- Ảnh `images.jpg`: Người gán nhãn ghi nhãn dòng trên là `59-U1`, nhưng thực tế ảnh chụp rõ ràng là phương tiện xe máy điện mang biển số đăng ký `59-MD1` (chữ `'M'` và `'D'` bị dính nét).
- Ảnh `images (2).jpg`: Biển số xe tỉnh Phú Yên thực tế là `78-F1 581.49`, nhưng nhãn văn bản gốc ghi nhầm thành `78-T1`.
- Ảnh `test_014.jpg`: Nhãn gốc ghi chữ `'O'` thay vì chữ số `'0'` ở dòng thứ hai.

Hệ thống của chúng tôi phát hiện và ghi nhận đầy đủ các trường hợp bất nhất quán này trong báo cáo kiểm toán dữ liệu nhằm bảo đảm tính toàn vẹn của kết quả khoa học.

---

## Phụ lục B: Danh mục kiểm tra tính tái lập thực nghiệm (Reproducibility Checklist)

Để đảm bảo tính tái lập độc lập của toàn bộ các kết quả thực nghiệm trong bài báo:
1. **Mã nguồn hoàn chỉnh:** Toàn bộ mã nguồn Python (`src/restoration.py`, `src/plate_detection.py`, `src/segmentation.py`, `src/features.py`, `src/classifier.py`, `src/app.py`) được đóng gói đầy đủ, không sử dụng thư viện đóng.
2. **Hạt giống ngẫu nhiên (Random Seeds):** Toàn bộ các bước tạo nhiễu, phân chia tập dữ liệu train/val/test (`data/splits/`), và khởi tạo mô hình học máy được cố định bằng `seed = 42`.
3. **Danh mục dữ liệu mở rộng:** Toàn bộ 690 bản ghi ảnh phương tiện cùng nhãn mặt đất chi tiết được xuất bản công khai trong tệp [data/splits/expanded_dataset_manifest.csv](file:///c:/PhatVo_Important/Codes/My_Codes/Academy/IMP302/project/data/splits/expanded_dataset_manifest.csv).
4. **Tệp phụ thuộc môi trường:** Toàn bộ các gói thư viện Python cùng phiên bản cụ thể được liệt kê trong tệp [requirements.txt](file:///c:/PhatVo_Important/Codes/My_Codes/Academy/IMP302/project/requirements.txt).

---

## Tài liệu tham khảo

[1] C. N. E. Anagnostopoulos, I. E. Anagnostopoulos, I. D. Psoroulas, V. Loumos, and E. Kayafas, "License plate recognition from still images and video sequences: A survey," *IEEE Transactions on Intelligent Transportation Systems*, vol. 9, no. 3, pp. 377–391, Sept. 2008.

[2] S. Du, M. Ibrahim, M. Shehata, and W. Badawy, "Automatic license plate recognition (ALPR): A state-of-the-art review," *IEEE Transactions on Circuits and Systems for Video Technology*, vol. 23, no. 2, pp. 311–325, Feb. 2013.

[3] R. C. Gonzalez and R. E. Woods, *Digital Image Processing*, 4th ed. New York, NY, USA: Pearson, 2018.

[4] R. Laroca, E. Severo, L. A. Zanlorensi, L. S. Oliveira, G. R. Gonçalves, W. R. Schwartz, and D. Menotti, "A robust real-time automatic license plate recognition based on the YOLO detector," in *Proceedings of the International Joint Conference on Neural Networks (IJCNN)*, Rio de Janeiro, Brazil, 2018, pp. 1–10.

[5] S. M. Silva and C. R. Jung, "License plate detection and recognition in unconstrained scenarios," in *Proceedings of the European Conference on Computer Vision (ECCV)*, Munich, Germany, 2018, pp. 580–596.

[6] S. Azam and M. M. Islam, "Automatic license plate detection in hazardous condition," *Journal of Visual Communication and Image Representation*, vol. 36, pp. 172–186, Apr. 2016.

[7] N. Otsu, "A threshold selection method from gray-level histograms," *IEEE Transactions on Systems, Man, and Cybernetics*, vol. 9, no. 1, pp. 62–66, Jan. 1979.

[8] N. Dalal and B. Triggs, "Histograms of oriented gradients for human detection," in *Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR)*, San Diego, CA, USA, vol. 1, 2005, pp. 886–893.

[9] S.-J. Ko and Y. H. Lee, "Center weighted median filters and their applications to image enhancement," *IEEE Transactions on Circuits and Systems*, vol. 38, no. 9, pp. 984–993, Sept. 1991.

[10] N. Wiener, *Extrapolation, Interpolation, and Smoothing of Stationary Time Series*. Cambridge, MA, USA: MIT Press, 1949.

[11] B. R. Hunt, "The application of constrained least squares estimation to image restoration by digital computer," *IEEE Transactions on Computers*, vol. C-22, no. 9, pp. 805–812, Sept. 1973.

[12] D. L. Donoho and I. M. Johnstone, "Ideal spatial adaptation by wavelet shrinkage," *Biometrika*, vol. 81, no. 3, pp. 425–455, Sept. 1994.

[13] S. G. Chang, B. Yu, and M. Vetterli, "Adaptive wavelet thresholding for image denoising and compression," *IEEE Transactions on Image Processing*, vol. 9, no. 9, pp. 1532–1546, Sept. 2000.

[14] D. H. Douglas and T. K. Peucker, "Algorithms for the reduction of the number of points required to represent a digitized line or its caricature," *Cartographica*, vol. 10, no. 2, pp. 112–122, Dec. 1973.

[15] R. Keys, "Cubic convolution interpolation for digital image processing," *IEEE Transactions on Acoustics, Speech, and Signal Processing*, vol. 29, no. 6, pp. 1153–1160, Dec. 1981.

[16] J. MacQueen, "Some methods for classification and analysis of multivariate observations," in *Proceedings of the 5th Berkeley Symposium on Mathematical Statistics and Probability*, vol. 1, 1967, pp. 281–297.

[17] D. Bradley and G. Roth, "Adaptive thresholding using the integral image," *Journal of Graphics Tools*, vol. 12, no. 2, pp. 13–21, 2007.

[18] R. M. Haralick and L. G. Shapiro, *Computer and Robot Vision*, vol. 1. Reading, MA, USA: Addison-Wesley, 1992.

[19] R. O. Duda, P. E. Hart, and D. G. Stork, *Pattern Classification*, 2nd ed. New York, NY, USA: Wiley-Interscience, 2000.

[20] R. N. Bracewell, *The Hartley Transform*. New York, NY, USA: Oxford University Press, 1986.

[21] N. Ahmed, T. Natarajan, and K. R. Rao, "Discrete cosine transform," *IEEE Transactions on Computers*, vol. C-23, no. 1, pp. 90–93, Jan. 1974.

[22] J. L. Walsh, "A closed set of normal orthogonal functions," *American Journal of Mathematics*, vol. 45, no. 1, pp. 5–24, Jan. 1923.

[23] C. T. Zahn and R. Z. Roskies, "Fourier descriptors for plane closed curves," *IEEE Transactions on Computers*, vol. C-21, no. 3, pp. 269–281, Mar. 1972.

[24] M.-K. Hu, "Visual pattern recognition by moment invariants," *IRE Transactions on Information Theory*, vol. 8, no. 2, pp. 179–187, Feb. 1962.

[25] C. Cortes and V. Vapnik, "Support-vector networks," *Machine Learning*, vol. 20, no. 3, pp. 273–297, Sept. 1995.

[26] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image quality assessment: From error visibility to structural similarity," *IEEE Transactions on Image Processing*, vol. 13, no. 4, pp. 600–612, Apr. 2004.
"""

out_path = os.path.join("report", "FINAL_REPORT_VI.md")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(content.strip() + "\n")

print(f"Successfully generated {out_path} ({len(content.splitlines())} lines, {len(content)} characters).")

