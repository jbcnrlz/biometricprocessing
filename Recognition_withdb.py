import random, numpy as np, argparse, re, os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from helper.functions import loadPatternFromFiles, loadFileFeatures
from sklearn.decomposition import PCA
from helper.functions import plot_confusion_matrix

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
    parser.add_argument('--saveScores', help='Path to save scores', required=False, default=None)
    parser.add_argument('--typeDecision', help='Which classifier to use as to make a decision', required=False, default='cosine_similarity')
    args = parser.parse_args()

    features = loadFileFeatures(args.path)
    patterns = loadPatternFromFiles(args.folds)

    experiments = generateFolds(features,patterns)

    finalResults = []

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
        print('Doing fold %d with %d fold subjects, gallery size %d' % (fnum,len(e[1]),len(gallery)))
        resultado = np.zeros(2)
        scoresCurrFold = np.zeros((len(probe), len(gallery)))
        labelsWhatever = np.zeros((len(probe), 1)).flatten()
        labelsPredicted = np.zeros((len(probe), 1)).flatten()
        if args.typeDecision == 'knn':
            kns = KNeighborsClassifier(n_neighbors=5)
            kns.fit(gallery[:,:-1],gallery[:,-1])
            decisions = kns.predict(probe[:,:-1])
            for idxD, d in enumerate(decisions):
                resultado[int(d==probe[idxD,-1])] += 1
        else:
            for snum, p in enumerate(probe):
                pdone = (snum / len(probe)) * 100
                cClass = p[-1]
                labelsWhatever[snum] = int(cClass)
                p = p[0:-1]
                temp_max = -10
                temp_index = 0
                temp_max = -1000
                for gnum, j in enumerate(gallery):
                    print('\r[%.2f Completed] --- Checking subject %d from class %d against gallery subject %d from class %d' % (pdone, snum, cClass, gnum, j[-1]), end='\r', flush=True)
                    temp_similarity = cosine_similarity(p.reshape(1, -1), j[:-1].reshape(1, -1))
                    scoresCurrFold[snum,gnum] = temp_similarity
                    if temp_max < temp_similarity:
                        temp_max = temp_similarity
                        temp_index = j[-1]

                resultado[int(temp_index == cClass)] += 1
                labelsPredicted[snum] = temp_index

            if args.saveScores is not None:
                if not os.path.exists(os.path.join(args.saveScores,str(fnum))):
                    os.makedirs(os.path.join(args.saveScores,str(fnum)))

                np.save(os.path.join(args.saveScores,str(fnum),'scores'),scoresCurrFold)
                np.save(os.path.join(args.saveScores, str(fnum), 'labels'), labelsWhatever)

            a = [i for i in range(int(max(labelsWhatever)) + 1)]
            confMat = plot_confusion_matrix(labelsWhatever, labelsPredicted, ['Subject ' + str(lnm) for lnm in a])
            confMat.savefig('cmatrix_fold_'+str(fnum)+'.png')

        resultado = resultado / len(e[1])
        print("\nRight %.2f Wrong %.2f" % (resultado[1] * 100, resultado[0] * 100))
        finalResults.append(resultado[1] * 100)

    print('Final result:')
    print(finalResults)