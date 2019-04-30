import random, numpy as np, argparse, re
from sklearn.metrics.pairwise import cosine_similarity
from helper.functions import loadPatternFromFiles, loadFileFeatures
from sklearn.decomposition import PCA

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
    parser.add_argument('--pca', help='Learn an apply PCA to the data', required=False, default=None)
    args = parser.parse_args()

    features = loadFileFeatures(args.path)
    patterns = loadPatternFromFiles(args.folds)

    experiments = generateFolds(features,patterns)

    for fnum, e in enumerate(experiments):
        gallery = np.array(e[0])
        probe = np.array(e[1])
        if args.pca is not None:
            try:
                componentesSize = float(args.pca)
                print('Generating PCA model -- initial feature size = %d' % (len(gallery[0])))
                pca = PCA(n_components=componentesSize, svd_solver='full')
                pca.fit(gallery[:,:-1])
                print('Applying PCA model')
                gallery = np.concatenate((pca.transform(gallery[:,:-1]),gallery[:,-1].reshape((-1,1))),axis=1)
                probe = np.concatenate((pca.transform(probe[:,:-1]),probe[:,-1].reshape((-1,1))),axis=1)
                print('Final feature size = %d' % (len(gallery[0])))
            except:
                from joblib import load
                pca = load(args.pca)
                print('Applying PCA model')
                gallery = np.concatenate((pca.transform(gallery[:,:-1]),gallery[:,-1].reshape((-1,1))),axis=1)
                probe = np.concatenate((pca.transform(probe[:,:-1]),probe[:,-1].reshape((-1,1))),axis=1)
                print('Final feature size = %d' % (len(gallery[0])))
        print('Doing fold %d with %d fold subjects' % (fnum,len(e[1])))
        resultado = np.zeros(2)
        for snum, p in enumerate(probe):
            pdone = (snum / len(probe)) * 100
            cClass = p[-1]
            p = p[0:-1]
            temp_max = -10
            temp_index = 0
            temp_max = -1000
            for gnum, j in enumerate(gallery):
                print('\r[%.2f Completed] --- Checking subject %d from class %d against gallery subject %d from class %d' % (pdone, snum, cClass, gnum, j[-1]), end='\r', flush=True)
                temp_similarity = cosine_similarity(p.reshape(1, -1), j[:-1].reshape(1, -1))
                if temp_max < temp_similarity:
                    temp_max = temp_similarity
                    temp_index = j[-1]

            resultado[int(temp_index == cClass)] += 1
        resultado = resultado / len(e[1])
        print("\nRight %.2f Wrong %.2f" % (resultado[1] * 100, resultado[0] * 100))