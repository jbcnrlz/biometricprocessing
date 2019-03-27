from helper.functions import generateData
from PIL import Image as im
import argparse, shutil, os, numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Deep Models')
    parser.add_argument('-p', '--pathBase', help='Path for faces', required=True)
    parser.add_argument('-o','--output', help='Output Folder', required=True)
    parser.add_argument('-l', '--layers', help='Layers to keep (like 1_2_3)', required=True)
    args = parser.parse_args()

    layers = [int(l) for l in args.layers.split('_')]
    if os.path.exists(args.output):
        shutil.rmtree(args.output)

    print('Criando diretorio')
    os.makedirs(args.output)

    print('Carregando dados')
    imageData, classesData = generateData(args.pathBase)
    for i in imageData:
        image = np.array(im.open(i))
        shape = list(image.shape)
        shape[-1] = len(layers)
        if shape[-1] == 1:
            shape = shape[:-1]
        newImage = np.zeros(shape)
        if len(shape) == 2:
            newImage[:,:] = image[:,:,layers[0]]
        else:
            newImage[:, :, :] = image[:, :, layers]
        fileName = i.split(os.path.sep)[-1]
        fileName = fileName.split('.')[0]
        if len(shape) == 2:
            image = im.fromarray(image).convert('L')
        else:
            image = im.fromarray(image)
        image.save(os.path.join(args.output,fileName+'.bmp'))