import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

# this is superpixel extension method, include: weight_expand, contrast_expand, counts_expand.
# plot different percent of superpixel, 10%,20%,30%...100%.
# you can use variable[i] check superpixel of different percent.
# all_sp = mask * label, interest_sp = tissue * label.


def contrast_expand(all_sp, interest_sp):
    a, countsa = np.unique(all_sp, return_counts=True)
    count_a = np.stack((a, countsa, np.zeros(np.shape(a))), axis=1)

    b, countsb = np.unique(interest_sp, return_counts=True)
    count_b = np.stack((b, countsb), axis=1)

    for i in range(a.size):
        for j in range(b.size):
            if count_a[i, 0] == count_b[j, 0]:
                count_a[i, 2] = count_b[j, 1]

    iou = []
    for i in range(a.size):
        iou.append(count_a[i, 2] / (count_a[i, 1]))

    temp = np.stack((a, iou), axis=1)
    temp = temp[np.lexsort(-temp.T)]
    contribute = temp[~np.any(temp == 0, axis=1)]
    return contribute


def counts_expand(interest_sp):
    b, countsb = np.unique(interest_sp, return_counts=True)
    contribute = np.stack((b, countsb), axis=1)
    contribute = contribute[np.lexsort(-contribute.T)]
    contribute = contribute[1:, :]
    return contribute


def weight_expand(all_sp, interest_sp):
    a, countsa = np.unique(all_sp, return_counts=True)
    count_a = np.stack((a, countsa, np.zeros(np.shape(a))), axis=1)

    b, countsb = np.unique(interest_sp, return_counts=True)
    count_b = np.stack((b, countsb), axis=1)

    for i in range(a.size):
        for j in range(b.size):
            if count_a[i, 0] == count_b[j, 0]:
                count_a[i, 2] = count_b[j, 1]

    iou = []
    for i in range(a.size):
        iou.append(count_a[i, 2] / (count_a[i, 1]))

    count_iou = []
    for i in range(a.size):
        count_iou.append(iou[i] * count_a[i, 2])

    temp = np.stack((a, count_iou), axis=1)
    temp = temp[np.lexsort(-temp.T)]
    contribute = temp[~np.any(temp == 0, axis=1)]
    return contribute


def positive_negative(positive, negative):
    a, countsa = np.unique(negative, return_counts=True)
    count_a = np.stack((a, countsa, np.zeros(np.shape(a))), axis=1)

    b, countsb = np.unique(positive, return_counts=True)
    count_b = np.stack((b, countsb), axis=1)

    for i in range(a.size):
        for j in range(b.size):
            if count_a[i, 0] == count_b[j, 0]:
                count_a[i, 2] = count_b[j, 1]

    iou = []
    for i in range(a.size):
        iou.append(count_a[i, 2] / (count_a[i, 1]+count_a[i, 2]))

    contribute = np.stack((a, iou), axis=1)
    contribute = contribute[np.lexsort(-contribute.T)]
    return contribute


def plot_percent(contribute, file_path):         # contribute is extension methods output;
    m11 = loadmat(file_path+'/m11.mat')['CalibratedM11']
    km_labels = loadmat(file_path+'/km_labels.mat')['km_labels']
    percent10 = contribute[0:int(0.1 * np.shape(contribute)[0]), 0]
    percent20 = contribute[0:int(0.2 * np.shape(contribute)[0]), 0]
    percent30 = contribute[0:int(0.3 * np.shape(contribute)[0]), 0]
    percent40 = contribute[0:int(0.4 * np.shape(contribute)[0]), 0]
    percent50 = contribute[0:int(0.5 * np.shape(contribute)[0]), 0]
    percent60 = contribute[0:int(0.6 * np.shape(contribute)[0]), 0]
    percent70 = contribute[0:int(0.7 * np.shape(contribute)[0]), 0]
    percent80 = contribute[0:int(0.8 * np.shape(contribute)[0]), 0]
    percent90 = contribute[0:int(0.9 * np.shape(contribute)[0]), 0]
    percent100 = contribute[:, 0]
    result = np.zeros(m11.shape)
    percent = ['percent10', 'percent20', 'percent30', 'percent40', 'percent50',
               'percent60', 'percent70', 'percent80', 'percent90', 'percent100', ]

    for per in percent:
        print(per)
        print(np.shape(eval(per)))
        for num in eval(per):
            result[km_labels == num] = 1
        enhence_result = 2 * result + m11
        plt.imshow(enhence_result)
        plt.show()
        result = np.zeros(m11.shape)

    return percent10, percent20, percent30, percent40, percent50, percent60, percent70, percent80, percent90, percent100
