from torchvision.transforms import transforms
from datasetClass.structures import loadFolder, loadFolderDepthDI
from helper.functions import getFilesInPath
import argparse, networks.PyTorch.jojo as jojo
import torch.utils.data, os, numpy as np, shutil, matplotlib.pyplot as plt
from joblib import load
from finetuneRESNET import initialize_model
from torch import nn

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Extract Deep Features utilizing GioGio Network')
    parser.add_argument('--loadFromFolder', default=None, help='Load folds from folders', required=True)
    parser.add_argument('--fineTuneWeights', default=None, help='Do fine tuning with weights', required=True)
    parser.add_argument('--output', default=None, help='Features output files', required=True)
    parser.add_argument('--network', help='Joestar network to use', required=False, default='giogio')
    parser.add_argument('--pca', help='PCA', required=False, default=False)
    parser.add_argument('--modeLoadFile', help='Mode to load', required=False, default='auto')
    parser.add_argument('--meanImage', help='Mean image', nargs='+', required=False, type=float)
    parser.add_argument('--stdImage', help='Std image', nargs='+', required=False, type=float)
    parser.add_argument('--depthFolder', help='Folder with the depth', required=False, default=None)
    parser.add_argument('--batch', type=int, default=50, help='Size of the batch', required=False)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataTransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=args.meanImage,
                             std=args.stdImage)
    ])

    if args.network != 'giogioinputkerneldepthdi' and args.depthFolder is None:
        paths = getFilesInPath(args.loadFromFolder)
        foldFile = loadFolder(args.loadFromFolder, args.modeLoadFile, dataTransform)
        gal_loader = torch.utils.data.DataLoader(foldFile, batch_size=args.batch, shuffle=False)
    else:
        depthTransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.6928382911600398],std=[0.18346924017986496])
        ])
        paths = getFilesInPath(args.loadFromFolder)
        validateFiles = []
        if 'iiitd' in paths[0]:
            for fD in paths:
                fileName = fD.replace('_depthnocolor','').split(os.path.sep)[-1]
                if 'newdepth' not in fileName:
                    fileName = fileName[:-4] + '_newdepth.bmp'
                else:
                    fileName = fileName[:-4] + '.bmp'

                if os.path.exists(os.path.join(args.depthFolder,fileName)):
                    validateFiles.append(fD)

            paths  = validateFiles

        folds = loadFolderDepthDI(args.loadFromFolder, args.modeLoadFile, dataTransform, args.depthFolder,depthTransform,filePaths=paths)
        gal_loader = torch.utils.data.DataLoader(folds, batch_size=args.batch, shuffle=False)

    checkpoint = torch.load(args.fineTuneWeights)
    if args.network == 'giogio':
        muda = jojo.GioGio(checkpoint['state_dict']['softmax.1.weight'].shape[0],in_channels=checkpoint['state_dict']['features.0.weight'].shape[1]).to(device)
    elif args.network == 'jolyne':
        channelsIn = 4 if args.modeLoadFile == 'RGBA' else 3
        muda = jojo.Jolyne(checkpoint['state_dict']['softmax.2.weight'].shape[0],
                           in_channels=channelsIn).to(device)
    elif args.network == 'resnet':
        if args.depthFolder is not None:
            channelsIn = 5
        else:
            channelsIn = 4 if args.modeLoadFile == 'RGBA' else 3
        muda, _ = initialize_model(checkpoint['state_dict']['fc.weight'].shape[0],channelsIn)
    elif args.network.lower() == 'giogioinputkernel':
        muda = jojo.GioGioModulateKernelInput(checkpoint['state_dict']['softmax.2.weight'].shape[0]).to(device)
    elif args.network.lower() == 'maestro':
        muda = jojo.MaestroNetwork(checkpoint['state_dict']['softmax.1.weight'].shape[0]).to(device)
    elif args.network.lower() == 'giogioinputkerneldepth':
        muda = jojo.GioGioModulateKernelInputDepth(checkpoint['state_dict']['softmax.2.weight'].shape[0]).to(device)
    elif args.network.lower() == 'giogioinputkerneldepthdi':
        muda = jojo.GioGioModulateKernelInputDepthDI(checkpoint['state_dict']['softmax.2.weight'].shape[0]).to(device)
    elif args.network == 'vanillapaper':
        muda = jojo.VanillaNetworkPaper(checkpoint['state_dict']['softmax.2.weight'].shape[0]).to(device)
    elif args.network == 'attentionIN':
        muda = jojo.AttentionDINet(checkpoint['state_dict']['softmax.2.weight'].shape[0]).to(device)
    elif args.network == 'attentionINCN':
        muda = jojo.AttentionDICrossNet(checkpoint['state_dict']['softmax.2.weight'].shape[0]).to(device)
    elif args.network == 'depthPaper':
            muda = jojo.DepthAM(checkpoint['state_dict']['softmax.2.weight'].shape[0]).to(device)
    elif args.network == 'mobilenet':
        if args.depthFolder is not None:
            channelsIn = 5
        else:
            channelsIn = 4 if args.modeLoadFile == 'RGBA' else 3
        muda, _ = initialize_model(checkpoint['state_dict']['classifier.3.weight'].shape[0],channelsIn,modelName=args.network)
    muda.load_state_dict(checkpoint['state_dict'])
    
    if args.network == 'resnet':
        modules=list(muda.children())[:-1]
        muda=nn.Sequential(*modules)
        for p in muda.parameters():
            p.requires_grad = False

        muda = muda.to(device)
    elif args.network == 'mobilenet':
        muda.classifier = muda.classifier[:-1]
        muda = muda.to(device)
    '''
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = [o.detach() for o in output]

        return hook

    muda.enFeat.register_forward_hook(get_activation('enFeat'))
    if not os.path.exists('attention_maps'):
        os.makedirs('attention_maps')
    '''
    muda.eval()
    galleryFeatures = []
    galleryClasses = []
    filePathNameIdx = 0
    with open(args.output, 'w') as dk:
        with torch.no_grad():
            for bIdx, (currBatch, currTargetBatch) in enumerate(gal_loader):
                if args.network != 'giogioinputkerneldepthdi':
                    print("Extracting features from batch %d"%(bIdx))
                    if args.network in ['resnet','mobilenet']:
                        if args.depthFolder is not None:
                            cbt, cdb = currBatch
                            cbt, cdb = cbt.to(device), cdb.to(device)
                            output = muda(torch.cat((cbt,cdb[:,:1,:,:]),1))
                        else:
                            output = muda(currBatch.to(device))
                        if args.network == 'resnet':
                            output = output.reshape((-1,2048))
                    else:
                        _, output = muda(currBatch.to(device))
                    galleryFeatures = output.tolist()
                    galleryClasses = currTargetBatch.tolist()

                    for i, data in enumerate(galleryFeatures):
                        filesName = paths[filePathNameIdx].split(os.path.sep)[-1]
                        filePathNameIdx += 1
                        dk.write(' '.join(list(map(str, data))) + ' ' + str(galleryClasses[i]) + ' ' + filesName +'\n')
                else:
                    print("Extracting features from batch %d" % (bIdx))
                    currBatch, depthBatch = currBatch
                    _, output = muda(currBatch.to(device),depthBatch.to(device))
                    galleryFeatures = output.tolist()
                    galleryClasses = currTargetBatch.tolist()

                    #act = activation['enFeat']

                    for i, data in enumerate(galleryFeatures):
                        filesName = paths[filePathNameIdx].split(os.path.sep)[-1]
                        filePathNameIdx += 1
                        '''
                        if not os.path.exists(os.path.join('attention_maps', filesName)):
                            os.makedirs(os.path.join('attention_maps', filesName))

                        for imsFes in range(5):
                            if not os.path.exists(os.path.join('attention_maps', filesName,str(imsFes))):
                                os.makedirs(os.path.join('attention_maps', filesName,str(imsFes)))

                            for imF in range(act[imsFes].shape[1]):
                                #plt.imshow(act[imsFes][i, imF, :].cpu())
                                plt.imsave(os.path.join('attention_maps',filesName, str(imsFes),str(imF) + '.png'),act[imsFes][i, imF, :].cpu())
                        '''
                        dk.write(' '.join(list(map(str, data))) + ' ' + str(galleryClasses[i]) + ' ' + filesName + '\n')

    if args.pca:
        pca = load(args.pca)
        galleryFeatures=np.array(galleryFeatures)
        galleryFeatures=pca.transform(galleryFeatures).tolist()
    '''
    print("Quantidade de características: %d" % (len(galleryFeatures[0])))
    print('Writing feature file')
    with open(args.output, 'w') as dk:
        for i, data in enumerate(galleryFeatures):
            filesName = paths[i].split(os.path.sep)[-1]
            dk.write(' '.join(list(map(str, data))) + ' ' + str(galleryClasses[i]) + ' ' + filesName +'\n')
    '''