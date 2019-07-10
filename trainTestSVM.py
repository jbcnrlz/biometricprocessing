import random, numpy as np, argparse, re, os, shutil
from sklearn.metrics.pairwise import cosine_similarity
from helper.functions import loadPatternFromFiles, loadFileFeatures
from sklearn.decomposition import PCA
from subprocess import check_output

def saveSVMTorchFile(pathFile, data, lineEnding='\n'):
    nf = open(pathFile, 'w')
    for d in data:
        nf.write(d + lineEnding)

    nf.close()

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
        if args.pca is not None:
            try:
                componentesSize = float(args.pca)
                print('Generating PCA model -- initial feature size = %d' % (len(gallery[0])))
                pca = PCA(n_components=componentesSize, svd_solver='full')
                pca.fit(gallery[:, :-1])
                print('Applying PCA model')
                gallery = np.concatenate((pca.transform(gallery[:, :-1]), gallery[:, -1].reshape((-1, 1))), axis=1)
                probe = np.concatenate((pca.transform(probe[:, :-1]), probe[:, -1].reshape((-1, 1))), axis=1)
                print('Final feature size = %d' % (len(gallery[0])))
            except:
                from joblib import load

                pca = load(args.pca)
                print('Applying PCA model')
                gallery = np.concatenate((pca.transform(gallery[:, :-1]), gallery[:, -1].reshape((-1, 1))), axis=1)
                probe = np.concatenate((pca.transform(probe[:, :-1]), probe[:, -1].reshape((-1, 1))), axis=1)
                print('Final feature size = %d' % (len(gallery[0])))
        print('Doing fold %d with %d fold subjects' % (fnum, len(e[1])))
        resultado = np.zeros(2)
        scoresCurrFold = np.zeros((len(probe), len(gallery)))
        labelsWhatever = np.zeros((len(probe), 1)).flatten()

        if os.path.exists(os.path.join('svmtorch','experiment_files')):
            shutil.rmtree(os.path.join('svmtorch','experiment_files'))

        os.makedirs(os.path.join('svmtorch','experiment_files'))
        os.makedirs(os.path.join('svmtorch', 'experiment_files', 'faces'))

        gallery = [str(len(gallery)) + ' ' + str(len(gallery[0]))] + [' '.join(list(map(str,gData[:-1]))) + ' ' + str(int(gData[-1])) for gData in gallery]
        probe = [str(len(probe)) + ' ' + str(len(probe[0]))] + [' '.join(list(map(str, pData[:-1]))) + ' ' + str(int(pData[-1])) for pData in probe]

        saveSVMTorchFile(os.path.join('svmtorch','experiment_files','gallery.txt'), gallery)
        saveSVMTorchFile(os.path.join('svmtorch', 'experiment_files', 'probe.txt'), probe)

        print('Training SVM')
        output = check_output('./svmtorch/SVMTorch -multi ' + os.path.join('svmtorch','experiment_files','gallery.txt') + ' ' + os.path.join('svmtorch', 'experiment_files', 'faces','face'), shell=True)
        print('Testing SVM')
        output = check_output('./svmtorch/SVMTest -multi ' + os.path.join('svmtorch', 'experiment_files', 'faces','face') + ' ' + os.path.join('svmtorch','experiment_files','probe.txt'), shell=True)
        classLine = output.__str__().split("\\n")[-4]
        currResult = float(classLine[classLine.index('[') + 1:classLine.index('%')])
        finalResults.append(currResult)

    print('Resultados: ')
    pr = [100 - finalResults[r] for r in range(len(finalResults))]
    print(pr)
