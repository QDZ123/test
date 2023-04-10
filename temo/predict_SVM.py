import torch
from PIL import Image
import sys
import os
import model.densenet as dn
import numpy as np
from os import listdir

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
from sklearn.model_selection import GridSearchCV
import torchvision.transforms as transforms

import cv2
from torch.utils.data import Dataset, DataLoader
from sklearn import svm


def load_model(resume):
    checkpoint = torch.load(resume)
    start_epoch = checkpoint['epoch']
    # 加载模型
    m_dn = dn.DenseNet3(40, 66, reduction=1.0, bottleneck=False)
    m_dn.load_state_dict(checkpoint['state_dict'])
    return m_dn


class DataProcess(Dataset):
    def __init__(self, path):
        self.data_info = []
        fileList = listdir(path)  # 获取当前文件夹下所有文件
        for child_dir in fileList:   # 类别名称  1 zheng
            digit = child_dir
            child_path = os.path.join(path, child_dir) # 小类别目录
            img_list = listdir(child_path)   # 小目录下所有文件
            for i in img_list:
                self.data_info.append((child_path + '/' + i, digit))

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        filepath, label = self.data_info[index]
        return filepath, label


class SVMClassifier(nn.Module):
    def __init__(self, feature_extractor, temp_nu, temp_g):
        super(SVMClassifier, self).__init__()
        self.feature_extractor = feature_extractor
        clf = svm.OneClassSVM(nu=temp_nu, kernel="rbf", gamma=temp_g)
        self.clf = clf
        self.normal_data = torch.tensor([])
        self.threshold = 0

    def feature_data(self, x):
        out = self.feature_extractor(x)
        temp = F.avg_pool2d(out, 8)
        temp = temp.view(-1, 456)  # 最终的特征
        return temp

    def img_process(self, path):
        data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # padding大小
        row, col = data.shape
        _max = max(col, row)
        temp = np.zeros((_max, _max), np.uint8)
        temp[0:row, 0:col] = data
        img = cv2.resize(temp, (48, 48), interpolation=cv2.INTER_CUBIC)
        img = img.reshape(1, 1, 48, 48)
        return img

    def folder_process(self, path):
        fileList = listdir(path)
        count = 0
        for child_dir in fileList:
            child_path = os.path.join(path, child_dir)
            count = count + 1
            img_data = self.img_process(child_path)
            img_data = torch.from_numpy(img_data).float()
            img_feature = self.feature_data(img_data)
            if count == 1:
                result = img_feature
            else:
                result = torch.cat((result, img_feature), dim=0)

        # print("总共数目：", count)
        result = result.detach().numpy()
        return result

    def fit(self, train_path):
        self.normal_data = self.folder_process(train_path)
        self.clf.fit(self.normal_data)
        dst = self.clf.decision_function(self.normal_data)
        dst = torch.from_numpy(dst)
        result = dst.sort()
        # print("result:", result)
        # self.threshold = result[0][0].item()
        self.threshold = -0.010827628165859322
        # print("阈值：", self.threshold)
        # print("result[0]：", result[0][0])

    def predict(self, temp_input):
        img_data = self.img_process(temp_input)
        img_data = torch.from_numpy(img_data).float()
        img_feature = self.feature_data(img_data)
        img_feature = img_feature.detach().numpy()

        # test_feature = self.folder_process(img_test)
        score = self.clf.decision_function(img_feature)
        return score[0]
        # print("score:", score)
        # if score[0] < self.threshold:   # 异常
        #     return 0
        # else:
        #     return 1
        # 正常
            # print("Anomalies detected:", score)

    def upgrade(self, new_feature):
        self.normal_data = np.append(self.normal_data, new_feature, axis=0)
        self.clf.fit(self.normal_data)
        new_dst = self.clf.decision_function(self.normal_data)
        new_dst = torch.from_numpy(new_dst)
        result = new_dst.sort()
        return result


    def val(self, val_loader):
        TP = 0
        TN = 0
        correct_count = 0
        for input, target in val_loader:
            n = len(input)
            for i in range(0, n):
                temp_input = input[i]
                temp_target = target[i]
                # print("temp_input:", temp_input, "temp_target:", temp_target)
                img_data = self.img_process(temp_input)
                img_data = torch.from_numpy(img_data).float()
                img_feature = self.feature_data(img_data)
                img_feature = img_feature.detach().numpy()
                score = self.clf.decision_function(img_feature)
                # print("temp_target:", temp_target, "score[0]:", score[0])

                if score[0] < self.threshold and temp_target == '1':   # 异常
                    correct_count = correct_count + 1
                    TN = TN + 1
                if score[0] > self.threshold and temp_target == '0':   # 正常
                    correct_count = correct_count + 1
                    TP = TP + 1
                    # new_result = self.upgrade(img_feature)
                    # # print("self.normal_data.shape[0]:", self.normal_data.shape[0])   # 116
                    # number = int(self.normal_data.shape[0] * (1 - rate))
                    # # print("number:", number)
                    # self.threshold = new_result[0][number]
                    # print("更新后：", self.threshold)
            return TP, TN
    def roc_val(self, val_loader):
        y_score = []
        for input, target in val_loader:
            n = len(input)
            for i in range(0, n):
                temp_input = input[i]
                y_score.append(self.predict(temp_input))
                # y_pred.append(self.predict(temp_input))
            # print("y_pred:", y_pred)
            return y_score, target


def main():   # 调用保存的最佳模型的准确率输出

    resume = '/home/DZ/densenet-pytorch/runs/DenseNet-40-12/model_best_siam.pth.tar'
    model = load_model(resume)

    # SVM使用示例
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    one = SVMClassifier(feature_extractor, 0.001, 0.06)

    f = open("SVMlog.txt", "a")
    #
    path = './experiment_raw'
    user_list = os.listdir(path)

    for i in user_list:  # 每个用户
        if i == '0':
            print("用户：", i)
            user_path = os.path.join(path, i)
            exper = os.listdir(user_path)
            for j in exper:  # 每次实验
                train_path = user_path + '/' + j + '/' + 'train' + '/' + '0'
                # print('train_path:', train_path)
                one.fit(train_path)
                # 加载test数据集
                test_path = user_path + '/' + j + '/' + 'test'
                p_path = test_path + '/' + '0'  # 正类
                n_path = test_path + '/' + '1'  # 负类
                testdata = DataProcess(test_path)
                # print("测试数据个数：", len(testdata))
                val_loader = DataLoader(testdata, batch_size=len(testdata), shuffle=True)

                # one.predict(test_path, threshold=-2)
                TP, TN = one.val(val_loader)
                sum_p = len(os.listdir(p_path))
                sum_n = len(os.listdir(n_path))
                print("sum_p:", sum_p, "TP:", TP)
                print("sum_n:", sum_n, "TN:", TN)
                # 错误接受率  横轴
                far = 1 - (TN / sum_n)
                # 错误拒绝率   1 - 纵轴
                frr = 1 - (TP / sum_p)
                print("用户", i, "的第", j, "次实验的错误接受率为：", far, "错误拒绝率为：", frr, file=f, flush=False)
                print("用户", i, "的第", j, "次实验的错误接受率为：", far, "错误拒绝率为：", frr, file=sys.stdout)
    f.close()


if __name__== '__main__':
    main()
