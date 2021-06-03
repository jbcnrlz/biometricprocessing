import random, numpy as np, argparse, re
from helper.functions import loadPatternFromFiles, loadFileFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def generateDatabase(pathFile):
    dataFile = None
    with open(pathFile, 'r') as f:
        dataFile = f.readlines()

    sizeProbe = int(len(dataFile) / 10)
    dataFile = np.array([list(map(float, x.strip().split(' '))) for x in dataFile])

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


def generateFolds(data, pattern):
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
                if re.match(p, fileName):
                    probe.append(features)

            for g in galleryPattern:
                if re.match(g, fileName):
                    gallery.append(features)

        experiment.append((gallery, probe))

    return experiment


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run recognition with db')
    parser.add_argument('--path', help='Path for features file', required=True)
    parser.add_argument('--folds', help='Path for folds file', required=True)
    parser.add_argument('--pca', help='Learn an apply PCA to the data', required=False, default=None)
    parser.add_argument('--saveScores', help='Path to save scores', required=False, default=None)
    args = parser.parse_args()

    features = loadFileFeatures(args.path)
    patterns = loadPatternFromFiles(args.folds)

    experiments = generateFolds(features, patterns)

    finalResults = []

    for fnum, e in enumerate(experiments):
        gallery = np.array(e[0])
        probe = np.array(e[1])
        neigh = KNeighborsClassifier(n_neighbors=3)
        #neigh = SVC()
        neigh.fit(gallery[:, :-1], gallery[:, -1])
        preds = neigh.predict(probe[:,:-1])
        labels = probe[:, -1]
        rs = preds == labels
        print(np.count_nonzero(rs))
        acertou = np.count_nonzero(rs) / probe.shape[0]
        errou = 1 - acertou
        print("Right %.2f Wrong %.2f" % (acertou * 100, errou * 100))

