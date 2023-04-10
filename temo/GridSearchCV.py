
import matplotlib.pyplot as plt
import sys
import time
from scipy import interpolate
import os
import torch
import numpy as np
import model.densenet as dn
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
from sklearn.model_selection import GridSearchCV
import cv2
from torchvision.models.feature_extraction import create_feature_extractor
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import make_scorer
import sklearn.metrics as metrics
import predict_SVM as ps
from scipy.interpolate import make_interp_spline
import predict_kmeans as pk
from scipy import optimize


def load_model(resume):
    checkpoint = torch.load(resume)
    start_epoch = checkpoint['epoch']
    # 加载模型
    m_dn = dn.DenseNet3(40, 66, reduction=1.0, bottleneck=False)
    m_dn.load_state_dict(checkpoint['state_dict'])
    return m_dn


def f1_scorer(y_true, y_pred):
    # print("y_true:", y_true, "y_pred:", y_pred)
    # print("f1_score(y_true, y_pred)", f1_score(y_true, y_pred))
    return f1_score(y_true, y_pred)


def make_score():
    return make_scorer(f1_scorer, greater_is_better=True)


def plot_ROC(preds, labels):
    fpr, tpr, thresholds = metrics.roc_curve(labels, preds, pos_label=0)
    # 插值法平滑ROC曲线
    mean_fpr = np.linspace(0, 1, 100)
    f = interpolate.interp1d(fpr, tpr, kind='linear')
    mean_tpr = f(mean_fpr)

    return mean_fpr, mean_tpr

# def plot_ROC(preds, labels, nu, g):
#     fpr, tpr, thresholds = metrics.roc_curve(labels, preds, pos_label=0)
#     # print("thresholds:", thresholds)
#     # roc_auc1 = metrics.auc(fpr, tpr)  # 计算auc的值，auc就是曲线包围的面积，越大越好
#
#     # 插值法平滑ROC曲线
#     # mean_fpr = np.linspace(0, 1, 100)
#     # f = interpolate.interp1d(fpr, tpr, kind='linear')
#     # mean_tpr = f(mean_fpr)
#     # plt.plot(mean_fpr, mean_tpr)
#     # # plt.title('nu ={:.2f}, g ={:.3f} ROC'.format(nu, g))
#     # plt.xlabel('fpr')
#     # plt.ylabel('tpr')
#     #
#     # plt.savefig('nu ={:.2f}, g ={:.3f}.png'.format(nu, g))
#     # plt.clf()
#     return fpr, tpr

    # 保存进log文件中
    # f = open("SVMGCVlog.txt", "a")
    # f = open("kmeansROC.txt", "a")
    # eer = optimize.brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
    # thresh = interpolate.interp1d(fpr, thresholds)(eer)
    # print("nu:", nu, "g:", g, "roc_auc1:", roc_auc1, "eer:", eer, "thresh:", thresh, file=f, flush=False)
    # print("nu:", nu, "g:", g, "roc_auc1:", roc_auc1, "eer:", eer, "thresh:", thresh, file=sys.stdout)


def sum_ROC(fpr, tpr, nu, g):
    mean_fpr = np.linspace(0, 1, 100)
    sum = []
    l = len(fpr)
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    for i in range(l):
        # plt.plot(fpr[i], tpr[i], color='gray', label="用户{}".format(i))
        f = interpolate.interp1d(fpr[i], tpr[i], kind='nearest')
        mean_tpr = f(mean_fpr)
        sum.append(mean_tpr)
    print("sum:", sum)
    tpr_row = len(sum)
    tpr_col = len(sum[0])
    print("tpr_row:", tpr_row, "tpr_col:", tpr_col)  # 66, 100
    temp = 0
    temp_tpr = []
    for t in range(tpr_col):
        for k in range(tpr_row):
            temp = temp + sum[k][t]
        temp = temp/tpr_row    # 求相同fpr的tpr平均值
        temp_tpr.append(temp)
        temp = 0
    # print("temp_tpr:", temp_tpr)
    eer = optimize.brentq(lambda x: 1. - x - interpolate.interp1d(mean_fpr, temp_tpr)(x), 0., 1.)
    # plt.plot(mean_fpr, temp_tpr, label='g ={:.3f}, nu ={:.2f}, EER={:7f}'.format(g, nu, eer))
    plt.plot(mean_fpr, temp_tpr, label='{:.3f},{:.2f},{:7f}'.format(g, nu, eer))
    plt.legend()
    # plt.savefig('g ={:.3f}, nu ={:.2f}.png'.format(g, nu))
    # plt.savefig('cubic_ROC')
    # plt.clf()


def load_data(path):
    data = ps.DataProcess(path)
    data_loader = DataLoader(data, batch_size=len(data), shuffle=True)
    return data_loader


# 要算对于每个用户的最优ROC
def compute_eer_2(path, g_list, nu_list, feature_extractor):
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    user_list = os.listdir(path)
    user_fpr = []
    user_tpr = []
    for user in range(len(user_list)):
        s_fpr = []   # 当前用户所有超参数组的ROC
        s_tpr = []
        s_eer = []
        for g in g_list:
            for nu in nu_list:
                clf = ps.SVMClassifier(feature_extractor, nu, g)
                train_path = path + '/' + str(user) + '/1/train/0'
                clf.fit(train_path)
                test_path = path + '/' + str(user) + '/1/test'
                val_loader = load_data(test_path)
                y_true = []   # y_pred 是score  y_true是label
                y_pred, target = clf.roc_val(val_loader)
                l = len(target)
                for k in range(0, l):
                    if target[k] == '0':
                        y_true.append(0)
                    else:
                        y_true.append(1)

                # fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=0)
                fpr, tpr = plot_ROC(y_pred, y_true)
                eer = optimize.brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
                plt.plot(fpr, tpr, label='{:.3f}, {:.2f}, {:7f}'.format(g, nu, eer))
                plt.legend()
                s_fpr.append(fpr)
                s_tpr.append(tpr)
                s_eer.append(eer)
        plt.savefig("user{}".format(user))
        plt.clf()
        print("用户", user)

        # 选出最小的eer, 以及fpr tpr
        min_eer = min(s_eer)  # 求列表最小值
        min_idx = s_eer.index(min_eer)  # 求最小值对应索引
        user_fpr.append(s_fpr[min_idx])
        user_tpr.append(s_tpr[min_idx])
    # 求平均eer
    sum = []
    mean_fpr = np.linspace(0, 1, 100)
    l = len(user_fpr)
    for i in range(l):
        plt.plot(user_fpr[i], user_tpr[i], color='gray')
        f = interpolate.interp1d(user_fpr[i], user_tpr[i], kind='nearest')
        mean_tpr = f(mean_fpr)
        sum.append(mean_tpr)
    tpr_row = len(sum)
    tpr_col = len(sum[0])
    # print("tpr_row:", tpr_row, "tpr_col:", tpr_col)  # 66, 100
    temp = 0
    temp_tpr = []
    for t in range(tpr_col):
        for k in range(tpr_row):
            temp = temp + sum[k][t]
        temp = temp / tpr_row  # 求相同fpr的tpr平均值
        temp_tpr.append(temp)
        temp = 0
    # print("temp_tpr:", temp_tpr)
    opti_eer = optimize.brentq(lambda x: 1. - x - interpolate.interp1d(mean_fpr, temp_tpr)(x), 0., 1.)
    print("opti_eer:", opti_eer)
    # plt.plot(mean_fpr, temp_tpr, label='g ={:.3f}, nu ={:.2f}, EER={:7f}'.format(g, nu, eer))
    plt.plot(mean_fpr, temp_tpr, label="mergeROC eer{:7f}".format(opti_eer))
    plt.legend()
    plt.savefig("ROC_raw_2")
    plt.clf()


#def kmeans_ROC(feature_extractor, path):
    # kmeansROC曲线
    # user_list = os.listdir(usr_path)
    # for u in range(len(user_list)):
    #     OC_model = pk.KmeansClassifier(feature_extractor)
    #     OC_model.fit(train_path)
    #     val_loader = load_data(test_path)
    #     score, target = OC_model.roc_val(val_loader)
    #     y_true = []
    #     l = len(target)
    #     for k in range(0, l):
    #         if target[k] == '0':
    #             y_true.append(0)
    #         else:
    #             y_true.append(1)
    #
    #     plot_ROC(score, y_true, 1, 1)

def main():
    resume = '/home/DZ/densenet-pytorch/runs/DenseNet-40-12/model_best_siam.pth.tar'
    model = load_model(resume)
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    # usr_path = './experiment_raw'
    usr_path = './experiment_tracked'
    g_list = [0.001, 0.005, 0.01, 0.1]
    n_list = [0.01, 0.03, 0.04, 0.05, 0.06]
    compute_eer_2(usr_path, g_list, n_list, feature_extractor)

    # SVM网格搜索
    # usr_path = './experiment_raw'
    # usr_path = './experiment_tracked'
    # user_list = os.listdir(usr_path)
    # g_list = [0.001, 0.005, 0.01, 0.1]
    # n_list = [0.01, 0.03, 0.04, 0.05, 0.06]
    # for i in g_list:
    #     for j in n_list:
    #         sum_f = []
    #         sum_t = []
    #         clf = ps.SVMClassifier(feature_extractor, j, i)
    #         for k in range(len(user_list)):  # 每个用户
    #             print("用户：", k, "len(user_list):", len(user_list))
    #             train_path = usr_path + '/' + str(k) + '/1/train/0'
    #             clf.fit(train_path)
    #             test_path = usr_path + '/' + str(k) + '/1/test'
    #             val_loader = load_data(test_path)
    #             y_true = []
    #             y_pred, target = clf.roc_val(val_loader)
    #             l = len(target)
    #             for k in range(0, l):
    #                 if target[k] == '0':
    #                     y_true.append(0)
    #                 else:
    #                     y_true.append(1)
    #
    #             fpr, tpr = plot_ROC(y_pred, y_true, j, i)
    #             sum_f.append(fpr)
    #             sum_t.append(tpr)
    #         print("sum_f:", len(sum_f), "sum_t:", len(sum_t))
    #         sum_ROC(sum_f, sum_t, j, i)
    #         plt.savefig("ROC_tracked")
    #         print("*************************************************")


if __name__== '__main__':
    main()
    # # 线性插值
    # linear_interp = interpolate.interp1d(fpr, tpr)
    # fpr_new = np.linspace(fpr.min(), fpr.max(), 100)
    # tpr_new_linear = linear_interp(fpr_new)

    # B样条插值
    # fpr_unique, idx = np.unique(fpr, return_index=True)
    # tpr_unique = []
    # for i in idx:
    #     tpr_unique.append(tpr[i])
    # # print("fpr_unique:", fpr_unique)
    # # print("tpr_unique:", tpr_unique)
    # spl = make_interp_spline(fpr_unique, tpr_unique)
    # tpr_new_spline = spl(fpr_new)

    # param_dict = [
    #     {'kernel': ['rbf'], 'gamma': [0.01, 0.001, 0.1, 0.005], 'nu': [0.03, 0.06, 0.04, 0.035, 0.05]},
    #     {'kernel': ['linear'], 'nu': [0.03, 0.06, 0.04, 0.035, 0.05]}
    #
    # ]