from helper.functions import generateData, generateFoldsOfData, generateImageData, loadFoldFromFolders
import networks.PyTorch.jojo as jojo, argparse, numpy as np, torch, torch.optim as optim, torch.nn.functional as F
import torch.utils.data, shutil, os

def save_checkpoint(state,is_best,filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Deep Models')
    parser.add_argument('-p', '--pathBase', default='generated_images_lbp_frgc', help='Path for faces', required=False)
    parser.add_argument('-b', '--batch', type=int, default=500, help='Size of the batch', required=False)
    parser.add_argument('-c', '--classNumber', type=int, default=466, help='Quantity of classes', required=False)
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Epochs to be run', required=False)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Carregando dados')
    imageData, classesData = generateData(args.pathBase)
    imageData = np.array(generateImageData(imageData)) / 255.0
    classesData = np.array(classesData)

    if os.path.exists('training_full_pytorch'):
        shutil.rmtree('training_full_pytorch')

    print('Criando diretorio')
    os.makedirs('training_full_pytorch')
    muda = jojo.GioGio(args.classNumber)
    muda.to(device)

    qntBatches = imageData.shape[0] / args.batch

    if np.amin(classesData) == 1:
        foldGalleryClasses = classesData - 1

    print('Criando tensores')
    imageData = torch.from_numpy(np.rollaxis(imageData, 3, 1)).float()
    classesData = torch.from_numpy(classesData)
    tdata = torch.utils.data.TensorDataset(imageData, classesData)
    train_loader = torch.utils.data.DataLoader(tdata, batch_size=args.batch, shuffle=True)

    print('Criando otimizadores')
    optimizer = optim.SGD(muda.parameters(), lr=0.01, momentum=0.5)

    print('Iniciando treino')
    for ep in range(args.epochs):
        for bIdx, (currBatch, currTargetBatch) in enumerate(train_loader):
            optimizer.zero_grad()
            output = muda(currBatch.to(device))
            loss = F.cross_entropy(output, currTargetBatch.to(device))
            loss.backward()
            optimizer.step()
            print('[%d, %05d de %05d] loss: %.3f' % (ep + 1, bIdx + 1, qntBatches, loss.item()))

            if (ep + 1) % 20 == 0:
                fName = '%s_checkpoint_%05d.pth.tar' % ('GioGio',ep)
                fName = os.path.join('training_full_pytorch', fName)
                save_checkpoint({
                    'epoch': ep + 1,
                    'arch': 'GioGio',
                    'state_dict': muda.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, False, fName)


    print('Terminou')