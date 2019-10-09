import argparse, numpy as np, pyperclip

def loadScoreFile(path):
    scores = []
    classes = []
    with open(path,'r') as fr:
        for f in fr:
            cscore = list(map(float, f.split(' ')[:-1]))
            scores.append(cscore)
            classes.append(int(f.split(' ')[-1]))

    return scores, classes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate CMC Data')
    parser.add_argument('--scoreFiles',help='Score file', required=True)
    parser.add_argument('--maxRankShow', help='Max score to show', required=False, default=15)
    parser.add_argument('--formating', help='How to present the data', required=False, default='matlab')
    args = parser.parse_args()

    finalRankResults = []

    for sf in args.scoreFiles.split('__'):

        scores, classes = loadScoreFile(sf)
        scores = np.array(scores)

        ranksResults = np.zeros((max(classes) + 1,1)).flatten()
        for idx, s in enumerate(scores):
            rank = np.where((-s).argsort() == classes[idx])[0][0]
            for i in range(rank,len(ranksResults)):
                ranksResults[i] += 1

        ranksResults = ranksResults / len(scores)
        finalRankResults.append(ranksResults)

    finalCopy=''
    for idx, f in enumerate(finalRankResults):
        if args.formating == 'matlab':
            listRanks = f[0:args.maxRankShow].tolist()
            finalCopy +='r%d = [ %s ];' % (idx,' '.join(list(map(str,listRanks)))) + '\n'
        else:
            print(ranksResults)

    #pyperclip.copy(finalCopy)
    print(finalCopy)


    '''
    for i in range(args.maxRank):
        print('oi')

    '''