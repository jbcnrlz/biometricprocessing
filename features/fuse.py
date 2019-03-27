def loadFileFeatures(pathFile):
    dataFile = None
    with open(pathFile,'r') as f:
        dataFile = f.readlines()

    returnFeatures = []
    returnClasses = []
    returnPaths = []
    for p in dataFile:
        breaked = p.split(' ')
        returnFeatures.append(list(map(float,breaked[:-2])))
        returnClasses.append(breaked[-2])
        returnPaths.append(breaked[-1])

    return returnFeatures, returnClasses, returnPaths

if __name__ == '__main__':
    featsgiogio, classesgiogio, pathsgiogio = loadFileFeatures('giogio_eurecom_3layers.txt')
    featsvgg, classesvgg, pathsgiogio = loadFileFeatures('eurecom_3layers_medium_bigger.txt')

    with open('fused_eurecom.txt', 'w') as dk:
        for i, data in enumerate(featsgiogio):
            dk.write(' '.join(list(map(str, data))) + ' ' + ' '.join(list(map(str, featsvgg[i]))) + ' ' + str(classesgiogio[i]) + ' ' + pathsgiogio[i])


    print('eita')
