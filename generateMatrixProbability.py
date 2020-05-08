from helper.functions import readMatrixFile
import argparse, numpy as np, os

def outputMatrixRelation(pathFile,matrix):
    with open(pathFile,'w') as pf:
        for m in matrix:
            pf.write(' '.join(list(map(str,m))) + '\n')

def main():
    parser = argparse.ArgumentParser(description='Calculate FRR and FAR from score files')
    parser.add_argument('--filesOrigin',help='Path for original files', required=True)
    parser.add_argument('--filesOutput',help='Path for output files', required=True)
    args = parser.parse_args()
    groupsInput = args.filesOrigin.split('__')
    outputInput = args.filesOutput.split('__')
    for idxGroup, filesGroup in enumerate(groupsInput):
        scores = np.array(readMatrixFile(filesGroup))
        trueLabel = scores[:,-1].flatten()
        scores = scores[:,:-1]
        outpuMatrix = np.zeros(scores.shape)
        for idx, t in enumerate(trueLabel):
            outpuMatrix[idx][int(t)] = 1

        outputFiles = outputInput[idxGroup].split('|')

        if os.path.sep in outputFiles[0]:
            folderCreate = os.path.sep.join(outputFiles[0].split(os.path.sep)[:-1])
            if not os.path.exists(folderCreate):
                os.makedirs(folderCreate)

        outputMatrixRelation(outputFiles[0],outpuMatrix)
        outputMatrixRelation(outputFiles[1],scores)

if __name__ == '__main__':
    main()