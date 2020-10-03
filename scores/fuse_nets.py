import numpy as np
from sklearn.preprocessing import MinMaxScaler
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
    mscaler = MinMaxScaler()
    for i in range(3):
        print(i)
        fileA = loadFile('3dlbp_eurecom_giogio_%d.txt' % (i))
        fileB = loadFile('sigmoid_eurecom_giogio_%d.txt' % (i))
        fileC = loadFile('newdepth_eurecom_giogio_%d.txt' % (i))
        jointFeats = (mscaler.fit_transform(fileA[:,:-1]) * 0.6) + (mscaler.fit_transform(fileB[:,:-1]) * 0.3) + (mscaler.fit_transform(fileC[:,:-1]) * 0.3)
        outputFeatureFiles(jointFeats,fileA[:,-1],'fusion_eurecom_3dlbp_sigmoid_newdepth_giogio_minmax_%d.txt' % (i))


if __name__ == '__main__':
    main()
