import os
import utils.data as ud
import model.densenet as dn
import numpy as np
from os import listdir

import torch
import torch.nn as nn
import torch.nn.parallel
from torchvision.datasets import ImageFolder
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms

import cv2
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


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


class KmeansClassifier(nn.Module):
    def __init__(self, feature_extractor):
        super(KmeansClassifier, self).__init__()
        self.feature_extractor = feature_extractor
        self.kmeans = KMeans(n_clusters=1)
        self.distances = 0
        self.normal_data = torch.tensor([])
        # self.threshold = 0

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
        print(result.shape)
        result = result.detach().numpy()
        return result

    def fit(self, train_path):
        self.normal_data = self.folder_process(train_path)
        self.kmeans.fit(self.normal_data)
        self.distances = np.linalg.norm(self.normal_data - self.kmeans.cluster_centers_, axis=1)

    def predict(self, temp_input):
        img_data = self.img_process(temp_input)
        img_data = torch.from_numpy(img_data).float()
        img_feature = self.feature_data(img_data)
        img_feature = img_feature.detach().numpy()
        dis_test = np.linalg.norm(img_feature - self.kmeans.cluster_centers_, axis=1)
        return dis_test

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

def main():
    resume = '/home/DZ/densenet-pytorch/runs/DenseNet-40-12/model_best_siam.pth.tar'
    model = load_model(resume)

    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    one = KmeansClassifier(feature_extractor)
    train_path = './Dataset/experiment_raw/0/1/train'
    test_path = './Dataset/experiment_raw/0/1/test'
    one.fit(train_path)
    # train_data = ImageFolder(root=train_path, transform=ud.transform_train)
    # print(len(train_data))
    # dataloader_train = DataLoader(train_data, batch_size=len(train_data), shuffle=True,
    #                               pin_memory=True, pin_memory_device='cuda')
    # for i, (input, target) in enumerate(dataloader_train):
    #     out = feature_extractor(input)
    #
    # print('out.shape:', out.shape)
    # temp = F.avg_pool2d(out, 8)
    # print("temp.shape:", temp.shape)
    # temp = temp.view(-1, 456)  # 最终的特征
    # print("temp.shape:", temp.shape)

    # out = self.feature_extractor(x)
    # temp = F.avg_pool2d(out, 8)
    # temp = temp.view(-1, 456)  # 最终的特征



if __name__== '__main__':
    main()
