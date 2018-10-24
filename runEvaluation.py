import random, numpy as np, sys, argparse, os
from helper.functions import getDirectoriesInPath
from sklearn.metrics.pairwise import cosine_similarity

def loadCharFile(pathFile):
    dataFile = loadTextFile(pathFile)
    return np.array([list(map(float,x.strip().split(' '))) for x in dataFile])

def loadTextFile(pathFile):
    df = None
    with open(pathFile,'r')as f:
        df = f.readlines()

    return df

def loadGalleryAndProbe(galFile,probFile,fullDatabase):
    gf = loadTextFile(galFile)
    pf = loadTextFile(probFile)

    dataSeparate = [gf,pf]
    dataReturn =[[],[]]

    for i, cData in enumerate(dataSeparate):
        for c in cData:
            currLine = int(c.split(' ')[-1])
            dataReturn[i].append(fullDatabase[currLine])

    return dataReturn[0], dataReturn[1]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run Evaluation')
    parser.add_argument('-p','--pathdatabase',help='Path for the database',required=True)
    parser.add_argument('-f', '--fileData', help='File of the full database', required=True)
    args = parser.parse_args()

    allDataset = loadCharFile(os.path.join(args.pathdatabase,args.fileData))

    fullResultados = []

    for d in getDirectoriesInPath(args.pathdatabase):
        gal, probe = loadGalleryAndProbe(
            os.path.join(args.pathdatabase,d,'gallery%02d.txt') % int(d),
            os.path.join(args.pathdatabase,d,'probe%02d.txt') % int(d),
            allDataset
        )
        print("Probe size %d Gallery size %d"%(len(probe),len(gal)))
        resultado = np.zeros(2)
        for p in probe:
            cClas = p[-1]
            p = p[0:-1]
            temp_index = 0
            temp_max = -1000
            for g in gal:
                ts = cosine_similarity(p.reshape(1,-1), g[0:-1].reshape(1,-1))
                if temp_max < ts:
                    temp_max = ts
                    temp_index = g[-1]

            resultado[int(temp_index == cClas)] += 1

        resultado = resultado / len(probe)
        print("Acertos %.2f Erro %.2f" % (resultado[1] * 100,resultado[0] * 100))
        fullResultados.append(resultado[1])

    print("Media %.2f Desvio Padrao %.2f" % (np.average(fullResultados) * 100, np.std(fullResultados) * 100))






