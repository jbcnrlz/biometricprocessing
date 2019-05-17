import numpy as np, argparse
from sklearn.decomposition import PCA

def outputFile(data,original,pathName):
    with open(pathName,'w') as fW:
        for x, i in enumerate(original):
            classNumber = i.split('_')[0]
            featureLine = ' '.join(list(map(str,data[x]))) + ' ' + classNumber + ' ' + i + '.png\n'
            fW.write(featureLine)


def loadFileFeatures(pathFile):
    dataFile = None
    with open(pathFile,'r') as f:
        dataFile = f.readlines()

    returnFeatures = []
    for d in dataFile:
        d = d.split(' ')
        returnFeatures.append([float(x) for x in d[:-2]] + [int(d[-2])] + [d[-1].strip()])

    return returnFeatures

def normedCrossCorrelation(a,b):
    return (1 / len(a)) * np.sum((a - np.mean(a)) * (b - np.mean(b))) / (np.sqrt(np.var(a) * np.var(b)))
    #a = (a - np.mean(a)) / (np.std(a) * len(a))
    #b = (b - np.mean(b)) / (np.std(b))
    #return np.correlate(a, b)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate fused models with the inverse of normed cross-correlation')
    parser.add_argument('--files', help='Original files with features', required=True)
    parser.add_argument('--modelOutput', help='Output features', required=True)
    parser.add_argument('--typeJoin', help='Type of join that is suppose to happen', required=False, default='sum')
    parser.add_argument('--pca', help='Models to lear PCA from', required=False, default=None)
    args = parser.parse_args()

    featuresSame = {}
    filesLoad = args.files.split('__')
    for f in filesLoad:
        features = loadFileFeatures(f)
        for fe in features:
            fileNoExt = fe[-1].split('.')[0]
            if fileNoExt in featuresSame.keys():
                featuresSame[fileNoExt].append(fe[:-2])
            else:
                featuresSame[fileNoExt] = [fe[:-2]]

    if args.typeJoin == 'sum':

        newFeatures = np.zeros((len(featuresSame),len(featuresSame[list(featuresSame.keys())[0]][0])))
        for idx, f in enumerate(featuresSame):
            weights = []
            for i in range(len(featuresSame[f])-1):
                for j in range(i+1,len(featuresSame[f])):
                    if (len(featuresSame[f]) == 2):
                        weights.append(normedCrossCorrelation(featuresSame[f][i], featuresSame[f][j]))
                    else:
                        weights.append(np.correlate(featuresSame[f][i],featuresSame[f][j]))

            if (len(featuresSame[f]) == 2):
                weights = (np.array(weights).flatten() + 1) / 2
            else:
                weights = np.array(weights).flatten()
                weights = weights / np.linalg.norm(weights)

            idxW = 0
            for i in range(len(featuresSame[f])-1):
                for j in range(i+1,len(featuresSame[f])):
                    sumVectors = np.array(featuresSame[f][i])+np.array(featuresSame[f][j]).flatten()
                    newFeatures[idx] += sumVectors*(1-weights[idxW])
                    idxW += 1

        outputFile(newFeatures,featuresSame,args.modelOutput)
    else:
        if args.pca is not None:
            features = {}
            for f in args.pca.split('__'):
                features[f] = loadFileFeatures(f)

            pcaModel = {}
            for f in features:
                currFeatures = np.array(features[f])
                currFeatures = currFeatures[:,:-2].astype(np.float64)
                pca = PCA(n_components=0.99, svd_solver='full')
                pca.fit(currFeatures)
                pcaModel[f] = pca

            newDatabase = []
            featuresKeys = list(features.keys())
            for idx, f in enumerate(featuresSame):
                newFeature = []
                for j in range(len(featuresSame[f])):
                    pcaFeatureModel = pcaModel[featuresKeys[j]].transform(np.array(featuresSame[f][j]).reshape(1, -1))
                    newFeature = newFeature + pcaFeatureModel.flatten().tolist()

                if len(newDatabase) > 0 and len(newDatabase[0]) != len(newFeature):
                    continue

                newDatabase.append(newFeature)

        else:
            newDatabase = []
            for idx, f in enumerate(featuresSame):
                newFeature = []
                for j in range(len(featuresSame[f])):
                    newFeature = newFeature + featuresSame[f][j]

                newDatabase.append(newFeature)

        outputFile(np.array(newDatabase),featuresSame,args.modelOutput)


