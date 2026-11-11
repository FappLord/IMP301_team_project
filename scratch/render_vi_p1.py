import pymupdf as fitz

doc = fitz.open('report/FINAL_REPORT_VI.pdf')
p1 = doc[0]
pix = p1.get_pixmap(dpi=150)
pix.save('scratch/vi_page_1.png')
print("Rendered vi_page_1.png successfully.")

