import numpy as np

def loadFile(pathFile):
    dados = []
    with open(pathFile,'r') as pf:        
        for p in pf:
            feature = None
            splittedFeats = p.split(' ')
            feature = list(map(float,splittedFeats[:-1]))
            feature.append(int(splittedFeats[-1]))
            dados.append(feature)

    return np.array(dados)

def outputFeatureFiles(feats,classes,filePath):
    with open(filePath,'w') as fp:
        for idx, f in enumerate(feats):
            fp.write(' '.join(list(map(str,f))) + ' ' + str(int(classes[idx])) + '\n')

def main():
    for i in range(3):
        print(i)
        fileA = loadFile('bosphorus/3dlbp_bosphorus_giogio_%d.txt' % (i))
        fileB = loadFile('bosphorus/sigmoid_bosphorus_giogio_%d.txt' % (i))
        fileC = loadFile('bosphorus/newdepth_bosphorus_giogio_%d.txt' % (i))
        jointFeats = (fileA[:,:-1] * 0.6) + (fileB[:,:-1] * 0.2) + (fileC[:,:-1] * 0.2)
        outputFeatureFiles(jointFeats,fileA[:,-1],'fusion_bosphorus_3dlbp_sigmoid_newdepth_giogio_%d.txt' % (i))


if __name__ == '__main__':
    main()
