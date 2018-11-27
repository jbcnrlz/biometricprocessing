import random, numpy as np, sys, cv2, argparse
from sklearn.metrics.pairwise import cosine_similarity

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run recognition with db')
    parser.add_argument('-p', '--pathdatabase', help='Path for the database', required=True)
    parser.add_argument('--PCA',
        help='Apply PCA to the data. The PCA is computed using the gallery and the probe characteristics are projeted into its space',
        required=False,
        default=False,
        type=bool
    )
    args = parser.parse_args()


    avg = []
    for foldNumber in range(10):
        pb, gl = generateDatabase(args.pathdatabase)

        if args.PCA:
            #espaco, faceMedia = gerarEspacoFace(gl)
            mean, eigenvector = cv2.PCACompute(gl,np.mean(gl,axis=0).reshape(1,-1))
            espaco = espaco * eigenvector

            novoEspaco, faceMedia = gerarEspacoFace(pb)
            novoEspaco = novoEspaco * eigenvector

        else:
            resultado =np.zeros(2)
            for p in pb:
                cClass = p[-1]
                p = p[0:-1]

                temp_max = -10
                temp_index = 0
                temp_max = -1000
                for j in gl:
                    temp_similarity = cosine_similarity(p.reshape(1,-1), j[0:-1].reshape(1,-1))
                    if temp_max < temp_similarity:
                        temp_max = temp_similarity
                        temp_index = j[-1]

                resultado[int(temp_index == cClass)] += 1

            resultado = resultado / len(pb)
            print("Acertos %.2f Erro %.2f" % (resultado[1] * 100,resultado[0] * 100))
            avg.append(resultado[1])

    print("Media %.2f Desvio Padrao %.2f" % (np.average(avg)*100,np.std(avg)*100))
