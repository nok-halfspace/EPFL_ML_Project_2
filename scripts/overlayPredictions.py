from PIL import Image, ImageEnhance, ImageOps
from constants import *

for k in range(1, NR_TEST_IMAGES + 1):
    background = Image.open("../Datasets/predictions/satImage_"+str(k).zfill(3)+".png")
    overlay = Image.open("../Datasets/test_set_images/test_" + str(k) + ".png")

    background = background.convert("RGBA")
    overlay = overlay.convert("RGBA")

    width, height = background.size

    for i in range(width):
        for j in range(height):
            pixel = background.getpixel((i,j))
            red   = pixel[0]
            green = pixel[1]
            blue  = pixel[2]
            newpixel = background.putpixel((i,j), (red*2,0,0))

    new_img = Image.blend(background, overlay, 0.7)
    enhancer = ImageEnhance.Brightness(new_img)
    enhanced_im = enhancer.enhance(1.8)
    enhanced_im.save("../Datasets/overlayPredictions/overlay"+str(k)+".png")
