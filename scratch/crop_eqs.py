from PIL import Image

im5 = Image.open('scratch/pdf_page_5.png')
w, h = im5.size
# Crop around Eq 2, 3, 4 (middle of page 5)
eq_crop5 = im5.crop((int(w * 0.1), int(h * 0.2), int(w * 0.9), int(h * 0.6)))
eq_crop5.save('scratch/crop_eq_page5.png')

im7 = Image.open('scratch/pdf_page_7.png')
w7, h7 = im7.size
eq_crop7 = im7.crop((int(w7 * 0.1), int(h7 * 0.4), int(w7 * 0.9), int(h7 * 0.8)))
eq_crop7.save('scratch/crop_eq_page7.png')

print("Cropped equation sections successfully.")

