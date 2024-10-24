import numpy as np
import pandas as pd
import h5py
import heapq
import matplotlib.pyplot as plt
from scipy.stats import norm
import gmpy2
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import (RBF, WhiteKernel, ConstantKernel,
                                              Matern, RationalQuadratic, DotProduct,
                                              ExpSineSquared,PairwiseKernel)
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from hyperopt import fmin, tpe, hp
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import time

import warnings
import sklearn.utils.validation as validation
warnings.filterwarnings("ignore", category=validation.DataConversionWarning)
from sklearn.exceptions import ConvergenceWarning
# 忽略 ConvergenceWarning 警告
warnings.filterwarnings("ignore", category=ConvergenceWarning)

data = pd.read_excel(r"D:\课程资料\本科毕设\本科毕业设计\Ms数据集\interation_0.xlsx")
X = data.iloc[:,0:5].values
y = data.iloc[:,5].values


# 几种效能函数
#########################################################################################
# 1、期望提升
def EI(regressor, X_train, y_train, X_space, y_space):
    regressor.fit(X_train, y_train)
    y_max = max(y_train)
    stds = []
    y_preds = []
    EIs = []
    for x in X_space:
        y_pred, std = regressor.predict(x.reshape(1, -1), return_std = True)      #GPR可计算不确定度
        y_preds.append(y_pred)
        stds.append(std)
        z = (y_pred - y_max) / std
        EI = std * (norm.pdf(z) + z * norm.cdf(z)) # pdf-概率密度函数；cdf-累计分布函数
        EIs.append(EI)
    # print(y_preds,'\n',stds)
    query_idx = np.argmax(EIs)
    return X_space[query_idx], y_space[query_idx], query_idx, y_preds[query_idx], stds[query_idx], EIs[query_idx]
# 2、不确定度
def std(regressor, X_train, y_train, X_space, y_space):
    regressor.fit(X_train, y_train)
    stds = []
    for x in X_space:
        _, std = regressor.predict(x.reshape(1, -1), return_std = True)
        stds.append(std)
    query_idx = np.argmax(stds)
    return X_space[query_idx], y_space[query_idx], query_idx
# 3、不确定度和预测值加权
def value_weighting(regressor, X_train, y_train, X_space, y_space):
    regressor.fit(X_train, y_train)
    values = []
    stds = []
    y_preds = []
    for x in X_space:
        y_pred, std = regressor.predict(x.reshape(1, -1), return_std=True)  # GPR可计算不确定度
        stds.append(std)
        y_preds.append(y_pred)
        value = 0.5 * std + 0.5 * y_pred
        values.append(value)
    # print(y_preds,'\n',stds)
    query_idx = np.argmax(values)
    return X_space[query_idx], y_space[query_idx], query_idx, y_preds[query_idx], stds[query_idx], values[query_idx]
# 4、不确定度和预测值的排名加权
def rank_weighting(regressor, X_train, y_train, X_space, y_space):
    regressor.fit(X_train, y_train)
    rank_values=[]
    stds= []
    y_preds = []
    for x in X_space:
        y_pred, std = regressor.predict(x.reshape(1, -1), return_std=True)  # GPR可计算不确定度
        stds.append(std)
        y_preds.append(y_pred)
    # print(y_preds,'\n',stds)
    sorted_stds = sorted(stds, reverse=True)
    sorted_y_preds = sorted(y_preds, reverse=True)
    rank_stds = [None]*len(stds)
    rank_y_preds = [None]*len(y_preds)
    for index, value in enumerate(sorted_stds):
        rank_stds[stds.index(value)] = index + 1
    for index, value in enumerate(sorted_y_preds):
        rank_y_preds[y_preds.index(value)] = index + 1
    for i in range(len(rank_stds)):
        rank_values.append(rank_stds[i]*0.1+rank_y_preds[i]*0.9)
    query_idx = np.argmin(rank_values)
    return X_space[query_idx], y_space[query_idx], query_idx, y_preds[query_idx], stds[query_idx], rank_values[query_idx]
#############################################################################################

kernel = ConstantKernel(constant_value=1) * (Matern(length_scale=1, nu=1.5) + WhiteKernel(noise_level=1))

num = 0  #执行一百次
i = 0  #随机种子
interaions = []
while num < 100:
    X_data, X_space, y_data, y_space = train_test_split(X, y, test_size=0.8, random_state=i)
    X_data_temp = X_data.reshape(-1, 5)
    y_data_temp = y_data.reshape(-1, 1)
    X_space_temp = X_space.reshape(-1, 5)
    y_space_temp = y_space.reshape(-1, 1)
    constant1 = max(y_data_temp)
    constant2 = max(y_space_temp)
    if constant1 <= constant2:
        num += 1
        interation = 0   #迭代次数
        gpr = GPR(kernel=kernel, n_restarts_optimizer=10)
        gpr.fit(X_data_temp, y_data_temp)
        regressor = GPR(kernel=gpr.kernel_, alpha=gpr.alpha, n_restarts_optimizer=10)
        print(regressor)
        regressor.fit(X_data_temp, y_data_temp)
        while True:
            a, b, c, d, e, f = rank_weighting(regressor, X_data_temp, y_data_temp, X_space_temp, y_space_temp)
            interation += 1
            if b > constant1:
                X_data_temp = np.append(X_data_temp, a).reshape(-1, 5)
                y_data_temp = np.append(y_data_temp, b).reshape(-1, 1)
                X_space_temp = np.delete(X_space_temp, c, axis=0).reshape(-1, 5)
                y_space_temp = np.delete(y_space_temp, c).reshape(-1, 1)
                print(f"随机种子为{i},迭代{interation}次,已执行{num}次")
                interaions.append(interation)
                break
            else:
                X_data_temp = np.append(X_data_temp, a).reshape(-1, 5)
                y_data_temp = np.append(y_data_temp, b).reshape(-1, 1)
                X_space_temp = np.delete(X_space_temp, c, axis=0).reshape(-1, 5)
                y_space_temp = np.delete(y_space_temp, c).reshape(-1, 1)
        i += 1
    else:
        i += 1
df = pd.DataFrame(interaions)
# 指定输出的 Excel 文件路径
output_file_path = 'D:\课程资料\本科毕设\本科毕业设计\小论文\GPR-rank.xlsx'
# 将 DataFrame 写入 Excel 文件
df.to_excel(output_file_path, index=False, sheet_name='Interactions')