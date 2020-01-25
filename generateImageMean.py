import torch.utils.data, argparse
from datasetClass.structures import loadFullSet
from torchvision import transforms

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Deep Models')
    parser.add_argument('--pathBase', default='generated_images_lbp_frgc', help='Path for faces', required=False)
    parser.add_argument('--batch', type=int, default=500, help='Size of the batch', required=False)
    args = parser.parse_args()

    print('Carregando dados')

    dataTransform = transforms.Compose([
        transforms.ToTensor()
    ])

    files = loadFullSet(args.pathBase,dataTransform)

    gal_loader = torch.utils.data.DataLoader(files, batch_size=args.batch)
    
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data, _ in gal_loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples
    
    mean /= nb_samples
    std /= nb_samples
    print("Media = %s Std = %s" % (mean,std))