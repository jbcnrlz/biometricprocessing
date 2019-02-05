from helper.functions import generateData
from PIL import Image as im
import argparse, shutil, os, numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Deep Models')
    parser.add_argument('-p', '--pathBase', help='Path for faces', required=True)
    parser.add_argument('-o','--output', help='Output Folder', required=True)
    args = parser.parse_args()

    if os.path.exists(args.output):
        shutil.rmtree(args.output)

    print('Criando diretorio')
    os.makedirs(args.output)

    print('Carregando dados')
    imageData, classesData = generateData(args.pathBase)
    for i in imageData:
        image = np.array(im.open(i))[:,:,0:3]
        fileName = i.split(os.path.sep)[-1]
        fileName = fileName.split('.')[0]
        image = im.fromarray(image)
        image.save(os.path.join(args.output,fileName+'.bmp'))