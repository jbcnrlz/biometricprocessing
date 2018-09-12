import glob, os, sys

def limparImagemEspecifica(path,filePoseCurrent):
    files = glob.glob(path + '/*'+filePoseCurrent+'*.png')
    for f in files:
        print('Removendo arquivo ' + f)
        os.remove(f)

def limparImagemEurecom(path,filePoseCurrent):
    session = ['s1','s2']
    folders = [os.path.join('Depth','DepthBMP')]
    os.chdir(path)
    files = os.listdir('.')
    for f in files:
        for s in session:
            for fo in folders:
                os.chdir(os.path.join(f,s,fo))
                filesDelete = glob.glob('*'+filePoseCurrent+'*.*')
                for fd in filesDelete:
                    os.remove(fd)
                os.chdir(path)

if __name__ == '__main__':
    limparImagemEurecom(sys.argv[1],'_OcclusionPaper_newdepth')