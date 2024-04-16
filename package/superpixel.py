import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import MiniBatchKMeans
from scipy.io import loadmat, savemat
import h5py


# Data is loaded from FinalMM.mat.
# MMpbp is loaded from your selection pbps.mat
# Clustern is number of superpixel.
# output variation: km_labels, groups


def superpixel(file_path, clustern=1024):
    # Load Mueller Matrix Data
    data = loadmat(file_path+'/FinalMM.mat')
    mmpbp = loadmat(file_path+'/pbps.mat')['pbps']

    temp = []
    mm_elements = ['FinalM11', 'FinalM12', 'FinalM13', 'FinalM14',
                   'FinalM21', 'FinalM22', 'FinalM23', 'FinalM24',
                   'FinalM31', 'FinalM32', 'FinalM33', 'FinalM34',
                   'FinalM41', 'FinalM42', 'FinalM43', 'FinalM44']
    for element in mm_elements:
        temp.append(data[element])
        MM = np.stack(temp, axis=2)

    # Invariant of Mueller Matrix elements
    # Edge
    DL = np.sqrt(MM[:, :, 1] ** 2 + MM[:, :, 2] ** 2)
    PL = np.sqrt(MM[:, :, 4] ** 2 + MM[:, :, 8] ** 2)
    rL = np.sqrt(MM[:, :, 7] ** 2 + MM[:, :, 11] ** 2)
    qL = np.sqrt(MM[:, :, 13] ** 2 + MM[:, :, 14] ** 2)
    # B
    b = 0.5 * (MM[:, :, 5] + MM[:, :, 10])
    Bhls = MM[:, :, 5] * MM[:, :, 10] - MM[:, :, 6] * MM[:, :, 9]
    Bfs = np.sqrt(MM[:, :, 5] ** 2 + MM[:, :, 6] ** 2 + MM[:, :, 9] ** 2 + MM[:, :, 10] ** 2)
    
    invariant = np.stack((DL, PL, rL, qL, b, Bhls, Bfs, MM[:, :, 3], MM[:, :, 12], MM[:, :, 15]), axis=2)

    # Calculate superpixel
    ss = StandardScaler()
    xx = ss.fit_transform(invariant.reshape(invariant.shape[0] * invariant.shape[1], invariant.shape[2]))
    minibkm = MiniBatchKMeans(n_clusters=clustern, batch_size=10*clustern, n_init='auto', random_state=2021214521)
    minibkm.fit(xx)
    km_labels = minibkm.labels_.reshape(MM.shape[:2])

    # result and output
    groups = []
    for i in range(clustern):
        groups.append(mmpbp[minibkm.labels_.reshape(MM.shape[:2]) == i].mean(axis=0))

    savemat(file_path+'/groups.mat', {'groups': groups})
    savemat(file_path+'/km_labels.mat', {'km_labels': km_labels})
    return groups, km_labels


def superpixel_pbps(file_path, clustern=1024):
    # Load Mueller Matrix Data
    print(file_path)
    mmpbp = h5py.File(file_path+'/pbps.mat')['pbps']
    mmpbp = np.transpose(mmpbp)
    print(np.shape(mmpbp))
    pbps = mmpbp.reshape(mmpbp.shape[0] * mmpbp.shape[1], mmpbp.shape[2])

    # Discretization pbps
    data = np.zeros(np.shape(pbps))
    for i in range(pbps.shape[1]):
        discret = pd.cut(pbps[:, i], bins=clustern, labels=False)
        data[:, i] = discret

    # Calculate superpixel
    #ss = StandardScaler()
    #xx = ss.fit_transform(data)
    #xx_ = xx.reshape(mmpbp.shape[0], mmpbp.shape[1], mmpbp.shape[2])
    mm = MinMaxScaler()
    xxx = mm.fit_transform(data)
    xxx_ = xxx.reshape(mmpbp.shape[0], mmpbp.shape[1], mmpbp.shape[2])
    minibkm = MiniBatchKMeans(n_clusters=clustern, batch_size=10*clustern, n_init='auto', random_state=2021214521)
    minibkm.fit(xxx)
    km_labels = minibkm.labels_.reshape(mmpbp.shape[:2])

    # result and output
    groups = []
    for i in range(clustern):
        groups.append(xxx_[minibkm.labels_.reshape(mmpbp.shape[:2]) == i].mean(axis=0))

    plt.imshow(km_labels)
    savemat(file_path+'/groups.mat', {'groups': groups})
    savemat(file_path+'/km_labels.mat', {'km_labels': km_labels})
    return groups, km_labels


if __name__ == '__main__':
    fp = ['data/canceer/ROI2']
    for file_path in fp:
        groups, km_labels = superpixel_pbps(file_path)
        print(groups)
