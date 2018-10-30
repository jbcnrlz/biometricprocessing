from networks.PyTorch import jojo
from helper.functions import getDirectoriesInPath
import torch, argparse, os, torch.optim as optim

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Deep Models')
    parser.add_argument('-n', '--network', default=None, help='Name for the model', required=False)
    parser.add_argument('-c', '--classNumber', type=int, default=466, help='Quantity of classes', required=False)
    args = parser.parse_args()

    muda = jojo.GioGio(args.classNumber)
    optimizer = optim.SGD(muda.parameters(), lr=0.01, momentum=0.5)

    dirs = getDirectoriesInPath('training_pytorch')
    for d in dirs:
        fName = '%s_checkpoint_%05d.pth.tar' % (args.network, d)
        cpoint = torch.load(os.path.join('training_pytorch',d,fName))
        muda.load_state_dict(cpoint['state_dict'])
        optimizer.load_state_dict(cpoint['optimizer'])