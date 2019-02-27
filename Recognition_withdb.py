import random, numpy as np, sys, cv2, argparse, os
import re

from sklearn.metrics.pairwise import cosine_similarity
from helper.functions import loadPatternFromFiles

def loadFileFeatures(pathFile):
    dataFile = None
    with open(pathFile,'r') as f:
        dataFile = f.readlines()

    returnFeatures = []
    for d in dataFile:
        d = d.split(' ')
        returnFeatures.append([float(x) for x in d[:-2]] + [int(d[-2])] + [d[-1].strip()])

    return returnFeatures

def generateDatabase(pathFile):
    dataFile = None
    with open(pathFile,'r') as f:
        dataFile = f.readlines()

    sizeProbe = int(len(dataFile)/10)
    dataFile = np.array([list(map(float,x.strip().split(' '))) for x in dataFile])

    foldChoices = random.sample([i for i in range(len(dataFile))], sizeProbe)
    probe = dataFile[foldChoices]
    gallery = []
    for i, d in enumerate(dataFile):
        if i not in foldChoices:
            gallery.append(d)

    return probe, np.array(gallery)

def gerarEspacoFace(faces):
    espaco = []
    faceMedia = np.zeros(faces.shape[1])
    for fa in faces:
        imgraw = fa
        espaco.append(imgraw)
        faceMedia = faceMedia + imgraw
    
    espaco = np.array(espaco)
    faceMedia = np.array(faceMedia) / float(espaco.shape[0])
    espaco = espaco - faceMedia
    return espaco, faceMedia

def generateFolds(data,pattern):
    experiment = []
    for f in pattern:
        probe = []
        gallery = []
        probePattern = f[1].split('__')
        galleryPattern = f[0].split('__')
        for d in data:
            features = d[:-1]
            fileName = d[-1]
            for p in probePattern:
                if re.match(p,fileName):
                    probe.append(features)

            for g in galleryPattern:
                if re.match(g,fileName):
                    gallery.append(features)

        experiment.append((gallery,probe))

    return experiment

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run recognition with db')
    parser.add_argument('--path', help='Path for features file', required=True)
    parser.add_argument('--folds', help='Path for folds file', required=True)
    args = parser.parse_args()

    features = loadFileFeatures(args.path)
    patterns = loadPatternFromFiles(args.folds)

    experiments = generateFolds(features,patterns)

    for fnum, e in enumerate(experiments):
        print('Doing fold ' + str(fnum))
        resultado = np.zeros(2)
        for snum, p in enumerate(e[1]):
            p = np.array(p)
            pdone = (snum / len(e[1])) * 100
            cClass = p[-1]
            p = p[0:-1]
            temp_max = -10
            temp_index = 0
            temp_max = -1000
            for gnum, j in enumerate(e[0]):
                j = np.array(j)
                print('\r [%.2f Completed] --- Checking subject %d from class %d against gallery subject %d from class %d' % (pdone, snum, cClass, gnum, j[-1]), end='\r', flush=True)
                temp_similarity = cosine_similarity(p.reshape(1, -1), j[:-1].reshape(1, -1))
                if temp_max < temp_similarity:
                    temp_max = temp_similarity
                    temp_index = j[-1]

            resultado[int(temp_index == cClass)] += 1
        resultado = resultado / len(e[1])
        print("\nAcertos %.2f Erro %.2f" % (resultado[1] * 100, resultado[0] * 100))

    print('opa')
    '''
    folds = getDirectoriesInPath(args.path)

    for f in folds:
        gl = loadFileFeatures(os.path.join(args.path,f,'gallery.txt'))
        pb = loadFileFeatures(os.path.join(args.path,f,'probe.txt'))
        print('Doing fold '+f)
        resultado = np.zeros(2)
        for snum, p in enumerate(pb):
            pdone = (snum/len(pb))*100
            cClass = p[-1]
            p = p[0:-1]
            temp_max = -10
            temp_index = 0
            temp_max = -1000
            for gnum, j in enumerate(gl):
                print('\r [%.2f Completed] --- Checking subject %d from class %d against gallery subject %d from class %d' % (pdone,snum,cClass,gnum,j[-1]), end='\r',flush=True)
                temp_similarity = cosine_similarity(p.reshape(1, -1), j[:-1].reshape(1, -1))
                if temp_max < temp_similarity:
                    temp_max = temp_similarity
                    temp_index = j[-1]

            resultado[int(temp_index == cClass)] += 1
        resultado = resultado / len(pb)
        print("\nAcertos %.2f Erro %.2f" % (resultado[1] * 100, resultado[0] * 100))
    '''