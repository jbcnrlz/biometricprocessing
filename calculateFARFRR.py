from helper.functions import readMatrixFile
import argparse, numpy as np

def calcFAR(scores,labels,trueLabel):
    thres = np.linspace(0.79457,scores[0].max())
    results = []
    for t in thres:
        acepted = 0
        for idxScores, score in enumerate(scores):
            idxScoresImp = score[labels[idxScores] != trueLabel[idxScores]]
            if any(idxScoresImp >= t):
                acepted += 1

        results.append(acepted / len(scores))

    return results

def calcFRR(scores,labels,trueLabel):
    thres = np.linspace(0.79457,scores[0].max())
    results = []
    for t in thres:
        acepted = 0
        for idxScores, score in enumerate(scores):
            idxScoresImp = score[labels[idxScores] == trueLabel[idxScores]]
            if all(idxScoresImp < t):
                acepted += 1

        results.append(acepted / len(scores))

    return results


def main():
    parser = argparse.ArgumentParser(description='Calculate FRR and FAR from score files')
    parser.add_argument('--scoreFile',help='Path for score file', required=True)
    parser.add_argument('--labelFile',help='Path for label file', required=True)
    args = parser.parse_args()
    scores = np.array(readMatrixFile(args.scoreFile))
    trueLabel = scores[:,-1].flatten()
    scores = scores[:,:-1]
    labels = np.array(readMatrixFile(args.labelFile))
    far = calcFAR(scores,labels,trueLabel)
    frr = calcFRR(scores,labels,trueLabel)
    print('far = [%s];' % (','.join(list(map(str,far)))))
    print('frr = [%s];' % (','.join(list(map(str,frr)))))
    print('oi')

if __name__ == '__main__':
    main()