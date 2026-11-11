from PIL import Image

im = Image.open('scratch/vi_page_1.png')
w, h = im.size
crop_top = im.crop((0, 0, w, int(h * 0.28)))
crop_top.save('scratch/vi_header_crop.png')
print("Cropped header successfully.")

