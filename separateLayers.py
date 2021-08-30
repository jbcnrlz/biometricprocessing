import argparse, os, cv2, numpy as np

def main():
    parser = argparse.ArgumentParser(description='Separate Layers')
    parser.add_argument('--filePath', help='Path for files to separate channels', required=True)
    parser.add_argument('--output', help='Folder to output to', required=True)
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    filesToOpen = args.filePath.split('|')
    channels = ['b','g','r','a']
    for f in filesToOpen:
        fileName = f.split(os.path.sep)[-1]
        imFile = cv2.imread(f,cv2.IMREAD_UNCHANGED)        
        for idx, nChan in enumerate(channels):
            if nChan == 'a':
                saveImg = np.zeros(imFile.shape[:-1])
                saveImg[:,:] = imFile[:,:,idx]   
            else:
                saveImg = np.zeros((imFile.shape[0],imFile.shape[1],3))
                saveImg[:,:,idx] = imFile[:,:,idx]            
            cv2.imwrite(os.path.join(args.output,fileName[:-4]+'_%s_channel' % (nChan) + fileName[-4:]),saveImg)
            




if __name__ == '__main__':
    main()