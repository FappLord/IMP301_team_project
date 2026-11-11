import pymupdf as fitz

doc = fitz.open('report/FINAL_REPORT.pdf')
# Page 5 has Eq 2, 3, 4
p5 = doc[4] # 0-indexed
pix5 = p5.get_pixmap(dpi=150)
pix5.save('scratch/pdf_page_5.png')

# Page 7 has Eq 9, 10
p7 = doc[6]
pix7 = p7.get_pixmap(dpi=150)
pix7.save('scratch/pdf_page_7.png')

print("Saved page 5 and 7 pixmaps.")

