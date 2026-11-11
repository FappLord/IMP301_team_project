from PIL import Image

im7 = Image.open('scratch/pdf_page_7.png')
w7, h7 = im7.size
eq_crop7 = im7.crop((int(w7 * 0.05), 0, int(w7 * 0.95), int(h7 * 0.25)))
eq_crop7.save('scratch/crop_eq10_page7.png')
print("Saved crop of Eq 10.")

