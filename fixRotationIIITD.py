from helper.functions import getFilesInPath
from PIL import Image as im

if __name__ == '__main__':
    files = getFilesInPath('generated_images_iiitd_nd')
    for f in files:
        a = im.open(f).rotate(180)
        a.save(f)