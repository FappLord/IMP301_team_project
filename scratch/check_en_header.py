import pymupdf as fitz
from PIL import Image

doc = fitz.open('report/FINAL_REPORT.pdf')
p1 = doc[0]
pix = p1.get_pixmap(dpi=150)
pix.save('scratch/en_page_1.png')

im = Image.open('scratch/en_page_1.png')
w, h = im.size
crop_top = im.crop((0, 0, w, int(h * 0.28)))
crop_top.save('scratch/en_header_crop.png')
print("Rendered and cropped en_header_crop.png successfully.")

