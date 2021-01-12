import argparse, numpy as np
from sklearn.preprocessing import MinMaxScaler

def loadScoreFile(path):
    mscaler = MinMaxScaler()
    scores = []
    classes = []
    with open(path,'r') as fr:
        for f in fr:
            cscore = list(map(float, f.split(' ')[:-1]))
            #scores.append(mscaler.fit_transform(np.array(cscore).reshape(-1,1)).flatten())
            scores.append(cscore)
            classes.append(int(f.split(' ')[-1]))

    return np.array(scores), np.array(classes)

def main():
    parser = argparse.ArgumentParser(description='Generate impostor distribution')
    parser.add_argument('--scoreFile',help='Score file', required=True)
    args = parser.parse_args()

    scs, cls = loadScoreFile(args.scoreFile)
    impostores = []
    genuinos = []
    for idx, s in enumerate(scs):
        impostores = impostores + np.concatenate((s[:cls[idx]],s[cls[idx]+1:])).tolist()
        genuinos.append(s[cls[idx]])


    occsimp, binsimp = np.histogram(impostores)
    occsgen, binsgen = np.histogram(genuinos)

    print(occsimp)
    print(binsimp)
    print(occsgen)
    print(binsgen)

    print("OYEAH!")


if __name__ == "__main__":
    main()