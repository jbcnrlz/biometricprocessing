import argparse, numpy as np, os
from helper.functions import getDirectoriesInPath

def readFile(pathFile,typeData=float):
    ranks = []
    with open(pathFile,'r') as pf:
        for p in pf:
            ranks.append(list(map(typeData,p.strip().split(' '))))

    return ranks

def main():
    parser = argparse.ArgumentParser(description='Generate CMC data with scores')
    parser.add_argument('--folderWithResults', help='Score file name', required=True)
    parser.add_argument('--scoresFile',help='Score file name', required=True)
    parser.add_argument('--labelsFile',help='Label file name',required=True)
    parser.add_argument('--numberOfRanks', type=int, default=20, help='Ranks', required=False)
    args = parser.parse_args()

    paths = getDirectoriesInPath(args.folderWithResults)

    for p in paths:
        scoreFiles = np.array(readFile(os.path.join(args.folderWithResults,p,args.scoresFile)))
        labelFiles = readFile(os.path.join(args.folderWithResults,p,args.labelsFile),int)

        ranks = np.zeros((args.numberOfRanks,2))

        for idxProbe, sc in enumerate(scoreFiles):
            indexes = sc[:-1].argsort()[::-1]
            for i in range(args.numberOfRanks):
                if labelFiles[idxProbe][indexes[i]] == int(sc[-1]):
                    for j in range(i,args.numberOfRanks):
                        ranks[j][1] += 1
                    break
                else:
                    ranks[i][0] += 1

        textPrint = []
        for r in ranks:
            textPrint.append(r[1]/sum(r))

        textPrint = ' '.join(list(map(str,textPrint)))
        print('[ %s ]' % (textPrint))

if __name__ == '__main__':
    main()