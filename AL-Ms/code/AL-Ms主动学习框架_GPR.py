import numpy as np
import pandas as pd
import h5py
import heapq
import matplotlib.pyplot as plt
from scipy.stats import norm
import gmpy2
from sklearn import preprocessing #预处理归一化
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import (RBF, WhiteKernel, ConstantKernel,
                                              Matern, RationalQuadratic, DotProduct,
                                              ExpSineSquared,PairwiseKernel)
from sklearn.decomposition import PCA
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import time
import warnings
import sklearn.utils.validation as validation
warnings.filterwarnings("ignore", category=validation.DataConversionWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

data = pd.read_excel(r"D:\课程资料\本科毕设\本科毕业设计\Ms数据集\5.xlsx")
X = data.iloc[:,0:5].values
y = data.iloc[:,5].values.reshape(-1,1)

# 1、期望提升
def EI(regressor, X_train, y_train, X_space):
    regressor.fit(X_train, y_train)
    y_max = max(y_train)
    stds = []
    y_preds = []
    EIs = []
    for x in X_space:
        y_pred, std = regressor.predict(x.reshape(1, -1), return_std=True)  # GPR可计算不确定度
        y_preds.append(y_pred)
        stds.append(std)
        z = (y_pred - y_max) / std
        EI = std * (norm.pdf(z) + z * norm.cdf(z))  # pdf-概率密度函数；cdf-累计分布函数
        EIs.append(EI)
    idx = heapq.nlargest(3, range(len(EIs)), EIs.__getitem__)
    return idx, y_preds, stds, EIs
# 2、不确定度
def std(regressor, X_train, y_train, X_space, y_space):
    regressor.fit(X_train, y_train)
    stds = []
    y_preds = []
    for x in X_space:
        y_pred, std = regressor.predict(x.reshape(1, -1), return_std = True)
        stds.append(std)
        y_preds.append(y_pred)
    idx = heapq.nlargest(3, range(len(stds)), stds.__getitem__)
    return idx, y_preds, stds, stds
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
        value = 0.7 * std + 0.3 * y_pred
        values.append(value)
    idx = heapq.nlargest(3, range(len(values)), values.__getitem__)
    return idx, y_preds, stds, values
# 4、不确定度和预测值的排名加权
def rank_weighting(regressor, X_train, y_train, X_space):
    regressor.fit(X_train, y_train)
    rank_values=[]
    stds= []
    y_preds = []
    for x in X_space:
        y_pred, std = regressor.predict(x.reshape(1, -1), return_std=True)  # GPR可计算不确定度
        stds.append(std)
        y_preds.append(y_pred)
    sorted_stds = sorted(stds, reverse=True)
    sorted_y_preds = sorted(y_preds, reverse=True)
    rank_stds = [None]*len(stds)
    rank_y_preds = [None]*len(y_preds)
    for index, value in enumerate(sorted_stds):
        rank_stds[stds.index(value)] = index + 1
    for index, value in enumerate(sorted_y_preds):
        rank_y_preds[y_preds.index(value)] = index + 1
    for i in range(len(rank_stds)):
        rank_values.append(rank_stds[i]*0.2+rank_y_preds[i]*0.8)
    idx = heapq.nsmallest(3, range(len(rank_values)), rank_values.__getitem__)
    return idx, y_preds, stds, rank_values
#############################################################################################

# random、boostapping、cross_validation 求解均方根误差RMSE
#############################################################################################
def random_RMSE(model, X, y):   #test_size=0.3
    RMSEs = []
    for i in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
        RMSEs.append(RMSE)
    mean_RMSE = np.mean(RMSEs)
    return mean_RMSE
def boostapping_RMSE(model, X, y):
    RMSEs = []
    n_samples = X.shape[0]
    for i in range(100):
        X_bootstrap, y_bootstrap = resample(X, y, n_samples=n_samples, random_state=i)
        model.fit(X_bootstrap, y_bootstrap)
        y_pred = model.predict(X)
        RMSE = np.sqrt(mean_squared_error(y, y_pred))
        RMSEs.append(RMSE)
    mean_RMSE = np.mean(RMSEs)
    return mean_RMSE
def cross_validation_RMSE(X, y):
    RMSEs = []
    for i in range(10):
        model = GPR(kernel=kernel, n_restarts_optimizer=5)
        kf = KFold(n_splits=10, shuffle=True, random_state=i)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)
            best_model = model
            y_pred = best_model.predict(X_test)
            RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
            RMSEs.append(RMSE)
    mean_RMSE = np.mean(RMSEs)
    return  mean_RMSE
###############################################################################################


kernel = ConstantKernel(constant_value=1) * (Matern(length_scale=1, nu=1.5) + WhiteKernel(noise_level=1))
gpr = GPR(kernel=kernel, n_restarts_optimizer=20)
gpr.fit(X, y)
regressor = GPR(kernel=gpr.kernel_, alpha=gpr.alpha,n_restarts_optimizer=20)
regressor.fit(X, y)

x_test0 = [[80,14,5,0,1],[43,26,28,1,2],[44,33,20,1,2],[66,2,26,4,2],[65,24,1,2,8],[70,25,4,0,1],[67,17,10,4,2],[66,28,3,1,2],[66,17,14,2,1]]
x_test = x_test0
for i in range(9):
    for j in range(5):
        x_test[i][j] = x_test[i][j]/100
validation_X= x_test
validation_y = [218.49,143.84,162.86,50.52,164.95,225.76,185.14,219.33,199.65]
y_pred = regressor.predict(validation_X, return_std=False)
print(y_pred)
RMSE = np.sqrt(mean_squared_error(validation_y, y_pred))
print(f'RMSE: {RMSE}')

# file_path = "D:\JetBrains\python\pythonProject\space_FeCoNiAlSi.h5"
# with h5py.File(file_path, 'r') as f:
#     for key in f.keys():
#         data = f[key]
#     X_space = [x / 100 for x in data]
#     pre, std = regressor.predict(X_space, return_std=True)
#     print(f'alloy1_1成分为{data[103651]}预测值: {pre[103651]},不确定度: {std[103651]}')
#     print(f'alloy1_2成分为{data[32986]}预测值: {pre[32986]},不确定度: {std[32986]}')
#     print(f'alloy1_3成分为{data[35661]}预测值: {pre[35661]},不确定度: {std[35661]}')
#     print(f'alloy2_1成分为{data[83031]}预测值: {pre[83031]},不确定度: {std[83031]}')
#     print(f'alloy2_2成分为{data[82512]}预测值: {pre[82512]},不确定度: {std[82512]}')
#     print(f'alloy2_3成分为{data[91397]}预测值: {pre[91397]},不确定度: {std[91397]}')
#     print(f'alloy3_1成分为{data[85851]}预测值: {pre[85851]},不确定度: {std[85851]}')
#     print(f'alloy3_2成分为{data[84640]}预测值: {pre[84640]},不确定度: {std[84640]}')
#     print(f'alloy3_3成分为{data[84022]}预测值: {pre[84022]},不确定度: {std[84022]}')
#     print(f'Fe65Co35成分为{data[82859]}预测值: {pre[82859]},不确定度: {std[82859]}')
#     print(f'预测值最大为{pre[np.argmax(pre)]}成分为{data[np.argmax(pre)]},不确定度: {std[np.argmax(pre)]}')
#     idx, a, b, c = EI(regressor, X, y, X_space)
#     print(f'效能函数查询得到了{len(idx)}组成分,预测值,不确定度,效能函数值如下:')
#     for i in idx:
#         print(f'成分: {data[i]},预测值: {a[i]},不确定度: {b[i]},效能函数值: {c[i]}')

