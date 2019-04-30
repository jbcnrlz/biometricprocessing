import argparse, os, numpy as np
from helper.functions import loadFileFeatures
from sklearn.decomposition import PCA
from joblib import dump

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate PCA model')
    parser.add_argument('--originalFile', help='Original file with features', required=True)
    parser.add_argument('--modelOutput', help='Output PCA model', required=True)
    args = parser.parse_args()

    if os.path.sep in args.modelOutput:
        path = args.modelOutput.split(os.path.sep)[:-1]
        path = os.path.sep.join(path)
        if not os.path.exists(path):
            os.makedirs(path)

    features = np.array([f[:-2] for f in loadFileFeatures(args.originalFile)])
    pca = PCA(n_components=0.99, svd_solver='full')
    pca.fit(features)

    if os.path.exists(args.modelOutput+'.joblib'):
        os.remove(args.modelOutput+'.joblib')

    dump(pca,args.modelOutput+'.joblib')