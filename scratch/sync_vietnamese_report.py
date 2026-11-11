with open('report/FINAL_REPORT_VI.md', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace any old/placeholder figure tags
import re
text = re.sub(r'!\[.*?\]\(\.\./outputs/fig01_system_architecture\.png\)\n\*Figure 1.*?\*\n', '', text)
text = re.sub(r'!\[.*?\]\(\.\./outputs/fig02_restoration_pipeline\.png\)\n\*Figure 2.*?\*\n', '', text)
text = re.sub(r'!\[.*?\]\(\.\./outputs/fig03_plate_detection\.png\)\n\*Figure 3.*?\*\n', '', text)
text = re.sub(r'!\[.*?\]\(\.\./outputs/fig04_segmentation_comparison\.png\)\n\*Figure 4.*?\*\n', '', text)
text = re.sub(r'!\[.*?\]\(\.\./outputs/fig05_features_classifiers\.png\)\n\*Figure 5.*?\*\n', '', text)
text = re.sub(r'!\[.*?\]\(\.\./outputs/fig06_dataset_distribution\.png\)\n\*Figure 6.*?\*\n', '', text)

# Now insert the 11 authentic figures at the correct locations:

# 1. In 3.2 after Section header
fig_g1 = """![Quy trình khôi phục và giải chập ảnh Giai đoạn 1](../outputs/fig01_module1_restoration.png)

![Các mô hình suy giảm đa tác nhân và bộ lọc khôi phục tương ứng](../outputs/fig09_multi_degradation_restoration.png)
"""
text = text.replace("### 3.2 Giai đoạn 1: Khôi phục ảnh đa tác nhân\n", f"### 3.2 Giai đoạn 1: Khôi phục ảnh đa tác nhân\n\n{fig_g1}\n")

# 2. In 3.3 after Section header
fig_g2 = """![Quy trình định vị hình thái học và nắn phẳng phối cảnh Homography Giai đoạn 2](../outputs/fig02_module2_detection.png)
"""
text = text.replace("### 3.3 Giai đoạn 2: Định vị biển số xe và Nắn chỉnh phối cảnh Homography\n", f"### 3.3 Giai đoạn 2: Định vị biển số xe và Nắn chỉnh phối cảnh Homography\n\n{fig_g2}\n")

# 3. In 3.4 after Section header
fig_g3 = """![Nhị phân hóa và phân đoạn ký tự liên thông 8 hướng Giai đoạn 3](../outputs/fig03_module3_segmentation.png)
"""
text = text.replace("### 3.4 Giai đoạn 3: Phân đoạn ký tự chống bóng đổ và Lọc viền bảo toàn nét\n", f"### 3.4 Giai đoạn 3: Phân đoạn ký tự chống bóng đổ và Lọc viền bảo toàn nét\n\n{fig_g3}\n")

# 4. In 3.5 after Section header
fig_g4 = """![Bộ mô tả biến đổi miền WHT, gradient HOG và bất biến tô-pô Giai đoạn 4](../outputs/fig04_module4_features.png)
"""
text = text.replace("### 3.5 Giai đoạn 4: Bộ đặc trưng lai 480 chiều và Phân loại mẫu thống kê\n", f"### 3.5 Giai đoạn 4: Bộ đặc trưng lai 480 chiều và Phân loại mẫu thống kê\n\n{fig_g4}\n")

# 5. In 4.1 after Section header
fig_g5 = """![Cấu trúc phân vùng tập dữ liệu và mức tăng PSNR qua các mô hình suy giảm](../outputs/fig10_dataset_splits_and_expansion.png)
"""
text = text.replace("### 4.1 Thiết kế phân chia tập dữ liệu chuẩn hóa không rò rỉ thông tin\n", f"### 4.1 Thiết kế phân chia tập dữ liệu chuẩn hóa không rò rỉ thông tin\n\n{fig_g5}\n")

# 6. In 4.2 after Section header
fig_g6 = """![So sánh đối sánh toàn bộ chu trình giữa ảnh suy giảm và ảnh đã khôi phục](../outputs/fig08_degraded_vs_restored_pipeline.png)
"""
text = text.replace("### 4.2 Đánh giá định lượng hiệu quả khôi phục ảnh Giai đoạn 1\n", f"### 4.2 Đánh giá định lượng hiệu quả khôi phục ảnh Giai đoạn 1\n\n{fig_g6}\n")

# 7. In 4.4 after Section header
fig_g7 = """![Ma trận nhầm lẫn của bộ phân loại SVM và chuẩn đối sánh các phương pháp](../outputs/fig05_module4_evaluation.png)
"""
text = text.replace("### 4.4 Đánh giá Giai đoạn 4: So sánh 7 bộ phân loại mẫu thống kê\n", f"### 4.4 Đánh giá Giai đoạn 4: So sánh 7 bộ phân loại mẫu thống kê\n\n{fig_g7}\n")

# 8. In 4.5 after Section header
fig_g8 = """![Biểu đồ phân tích triệt tiêu thành phần đặc trưng (Ablation Study)](../outputs/fig06_ablation_studies.png)
"""
text = text.replace("### 4.5 Nghiên cứu triệt tiêu thành phần (Ablation Study)\n", f"### 4.5 Nghiên cứu triệt tiêu thành phần (Ablation Study)\n\n{fig_g8}\n")

# 9. In 4.6 after Section header
fig_g9 = """![Đánh giá trên các kịch bản thử nghiệm thực tế khắc nghiệt (In-the-Wild)](../outputs/fig11_wild_generalization_cases.png)

![Thư viện kết quả thực nghiệm hoàn chỉnh trên tập kiểm tra độc lập](../outputs/fig07_end_to_end_demo.png)
"""
text = text.replace("### 4.6 Đánh giá trên bốn kịch bản thử nghiệm thực tế khắc nghiệt (Stress-Test)\n", f"### 4.6 Đánh giá trên bốn kịch bản thử nghiệm thực tế khắc nghiệt (Stress-Test)\n\n{fig_g9}\n")

with open('report/FINAL_REPORT_VI.md', 'w', encoding='utf-8') as f:
    f.write(text)

print("Updated FINAL_REPORT_VI.md with all 11 authentic figures.")

