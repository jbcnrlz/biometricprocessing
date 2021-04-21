import matplotlib.pyplot as plt
import argparse, cv2, os

def main():
    parser = argparse.ArgumentParser(description='Extract average from layers')
    parser.add_argument('--pathBase',default='generated_images_lbp_frgc',help='Path for faces', required=False)
    parser.add_argument('--pathBase3DLBP', default='generated_images_lbp_frgc', help='Path for faces', required=False)
    parser.add_argument('--pathBaseRGB', default='generated_images_lbp_frgc', help='Path for faces', required=False)
    args = parser.parse_args()

    #files = getFilesInPath(args.pathBase)
    #filesRGB = getFilesInPath(args.pathBaseRGB)
    #files3DLBP = getFilesInPath(args.pathBase3DLBP)
    #print(files[0])
    #print(filesRGB[0])
    #print(files3DLBP[0])
    imDI = cv2.imread(os.path.join(args.pathBase,'50_s2_depth_0050_s2_Neutral.png'),cv2.IMREAD_UNCHANGED)
    imRGB = cv2.imread(os.path.join(args.pathBaseRGB,'rgb_0051_s2_Neutral.bmp'))
    im3DLBP = cv2.imread(os.path.join(args.pathBase3DLBP,'50_s2_depth_0050_s2_Neutral.png'),cv2.IMREAD_UNCHANGED)

    fig, axes = plt.subplots(3,3, sharex=True, sharey=True)
    axes[0, 0].set_title('RGB data')
    axes[0, 1].set_title('Sigmoid data')
    axes[0, 2].set_title('3DLBP data')
    xrgb = imRGB[:, :, 0]
    xdi = imDI[:, :, 0]
    x3dlbp = im3DLBP[:, :, 0]
    for i in range(1,3):
        yrgb = imRGB[:, :, i]
        axes[i-1,0].scatter(xrgb,yrgb)
        ydi = imDI[:, :, i]
        axes[i-1,1].scatter(xdi,ydi)
        y3dlbp = im3DLBP[:,:,i]
        axes[i-1,2].scatter(x3dlbp,y3dlbp)

    ydi = imDI[:, :, 3]
    axes[2,1].scatter(xdi,ydi)
    y3dlbp = im3DLBP[:,:,3]
    axes[2,2].scatter(x3dlbp,y3dlbp)

    plt.savefig('corr_graph.png')

if __name__ == '__main__':
    main()