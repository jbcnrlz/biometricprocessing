from torchvision import transforms
from torch.autograd import Variable
from torch.nn import functional as F
from datasetClass.structures import loadFolder, loadFolderDepthDI
from helper.functions import getFilesInPath
import argparse, networks.PyTorch.jojo as jojo, os, cv2, torch, numpy as np

def return_CAM(feature_conv, weight):
    # generate the class -activation maps upsample to 256x256
    size_upsample = (100, 100)
    nc, h, w = feature_conv.shape
    output_cam = []    
    beforeDot =  feature_conv.reshape((nc, h*w))
    cam = np.matmul(weight, beforeDot)
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)    
    return cv2.resize(cam_img, size_upsample)

def main():
    parser = argparse.ArgumentParser(description='Generate CAMs')
    parser.add_argument('--loadFromFolder', default=None, help='Load folds from folders', required=True)
    parser.add_argument('--fineTuneWeights', default=None, help='Do fine tuning with weights', required=True)
    parser.add_argument('--output', default=None, help='CAM output', required=True)
    parser.add_argument('--network', help='Joestar network to use', required=False, default='giogio')
    parser.add_argument('--meanImage', help='Mean image', nargs='+', required=False, type=float)
    parser.add_argument('--stdImage', help='Std image', nargs='+', required=False, type=float)
    parser.add_argument('--depthFolder', help='Folder with the depth', required=False)
    parser.add_argument('--batch', type=int, default=50, help='Size of the batch', required=False)
    parser.add_argument('--modeLoadFile', help='Mode to load', required=False, default='auto')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataTransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=args.meanImage,
                             std=args.stdImage)
    ])

    if args.network != 'giogioinputkerneldepthdi':
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

    muda.load_state_dict(checkpoint['state_dict'])

    
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = [o.detach() for o in output]

        return hook

    muda.enFeat.register_forward_hook(get_activation('enFeat'))
    
    muda.eval()

    wts = list(muda.enFeat.parameters())
    wts = np.squeeze(wts[-1].cpu().data.numpy())

    galleryFeatures = []
    galleryClasses = []
    filePathNameIdx = 0
    with torch.no_grad():
        for bIdx, (currBatch, currTargetBatch) in enumerate(gal_loader):
            if args.network != 'giogioinputkerneldepthdi':
                print("Extracting features from batch %d"%(bIdx))
                _, output = muda(currBatch.to(device))

                galleryFeatures = output.tolist()
                galleryClasses = currTargetBatch.tolist()

            else:
                layers = ['r','g','b','a','depth']
                print("Extracting features from batch %d" % (bIdx))
                currBatch, depthBatch = currBatch
                _, output = muda(currBatch.to(device),depthBatch.to(device))
                galleryFeatures = output.tolist()
                galleryClasses = currTargetBatch.tolist()
                
                act = activation['enFeat']

                for j in range(act[0].shape[0]):
                    filesName = paths[filePathNameIdx].split(os.path.sep)[-1]
                    filePathNameIdx += 1
                    #imageOriginal = cv2.imread(os.path.join(args.loadFromFolder,filesName),cv2.IMREAD_UNCHANGED)
                    camImage = np.zeros((100,100,4))
                    for i in range(len(act)-1):
                        camImage[:,:,i] = return_CAM(act[i][j,:,:,:].cpu().numpy(),wts)
                        #hmap = cv2.applyColorMap(camImage,cv2.COLORMAP_JET)
                    cv2.imwrite(os.path.join(args.output,layers[i]+'_'+filesName),camImage)


if __name__ == '__main__':
    main()