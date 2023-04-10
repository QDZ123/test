import matplotlib.pyplot as plt
from scipy import interpolate
import os
import torch
import numpy as np
import torch.nn as nn
import sklearn.metrics as metrics
import predict_kmeans as pk
import GridSearchCV as GS
from scipy import optimize



def merge(fpr, tpr):
    sum = []
    mean_fpr = np.linspace(0, 1, 100)
    l = len(fpr)
    for i in range(l):
        eer = optimize.brentq(lambda x: 1. - x - interpolate.interp1d(fpr[i], tpr[i])(x), 0., 1.)
        plt.plot(fpr[i], tpr[i], color='gray', label="用户{},{:7f}".format(i, eer))
        f = interpolate.interp1d(fpr[i], tpr[i], kind='nearest')
        mean_tpr = f(mean_fpr)
        sum.append(mean_tpr)
    tpr_row = len(sum)
    tpr_col = len(sum[0])
    temp = 0
    temp_tpr = []
    for t in range(tpr_col):
        for k in range(tpr_row):
            temp = temp + sum[k][t]
        temp = temp / tpr_row  # 求相同fpr的tpr平均值
        temp_tpr.append(temp)
        temp = 0
    opti_eer = optimize.brentq(lambda x: 1. - x - interpolate.interp1d(mean_fpr, temp_tpr)(x), 0., 1.)
    plt.plot(mean_fpr, temp_tpr, label="mergeROC eer{:7f}".format(opti_eer))
    plt.legend()
    plt.savefig("kmeans_ROC_raw")
    plt.clf()


def plot_ROC(preds, labels):
    fpr, tpr, thresholds = metrics.roc_curve(labels, preds, pos_label=0)
    # 插值法平滑ROC曲线
    mean_fpr = np.linspace(0, 1, 100)
    f = interpolate.interp1d(fpr, tpr, kind='linear')
    mean_tpr = f(mean_fpr)
    return mean_fpr, mean_tpr


def plot_ROC_1(path, feature_extractor):
    user_list = os.listdir(path)
    s_fpr = []
    s_tpr = []
    for u in range(len(user_list)):
        print("用户", u)
        OC_model = pk.KmeansClassifier(feature_extractor)
        train_path = path + '/' + str(u) + '/1/train/0'
        test_path = path + '/' + str(u) + '/1/test'
        OC_model.fit(train_path)
        val_loader = GS.load_data(test_path)
        score, target = OC_model.roc_val(val_loader)
        y_true = []
        l = len(target)
        for k in range(0, l):
            if target[k] == '0':
                y_true.append(0)
            else:
                y_true.append(1)

        fpr, tpr = plot_ROC(score, y_true)
        s_fpr.append(fpr)
        s_tpr.append(tpr)
    merge(s_fpr, s_tpr)


def main():
    # kmeansROC曲线
    resume = '/home/DZ/densenet-pytorch/runs/DenseNet-40-12/model_best_siam.pth.tar'
    model = GS.load_model(resume)
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    usr_path = './Dataset/experiment_raw'
    plot_ROC_1(usr_path, feature_extractor)
    # usr_path = './Dataset/experiment_tracked'


if __name__== '__main__':
    main()
