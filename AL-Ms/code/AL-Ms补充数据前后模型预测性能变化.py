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

data = pd.read_excel(r"D:\课程资料\本科毕设\本科毕业设计\Ms数据集\interation_3.xlsx")
X = data.iloc[:,0:5].values
y = data.iloc[:,5].values.reshape(-1,1)
print(X)
print(y)
def boostapping_RMSE(X, y):
    RMSEs = []
    n_samples = X.shape[0]
    for i in range(100):
        model = GPR(kernel=kernel, n_restarts_optimizer=20)
        X_bootstrap, y_bootstrap = resample(X, y, n_samples=n_samples, random_state=i)
        model.fit(X_bootstrap, y_bootstrap)
        best_model = GPR(kernel=model.kernel_, alpha=model.alpha, n_restarts_optimizer=20)
        best_model.fit(X_bootstrap, y_bootstrap)
        y_pred = model.predict(X)
        RMSE = np.sqrt(mean_squared_error(y, y_pred))
        RMSEs.append(RMSE)
    mean_RMSE = np.mean(RMSEs)
    return mean_RMSE
def boostapping_R2(X, y):
    R2s = []
    n_samples = X.shape[0]
    for i in range(100):
        model = GPR(kernel=kernel, n_restarts_optimizer=20)
        X_bootstrap, y_bootstrap = resample(X, y, n_samples=n_samples, random_state=i)
        model.fit(X_bootstrap, y_bootstrap)
        best_model = GPR(kernel=model.kernel_, alpha=model.alpha, n_restarts_optimizer=20)
        best_model.fit(X_bootstrap, y_bootstrap)
        y_pred = model.predict(X)
        R2 = r2_score(y, y_pred)
        R2s.append(R2)
    mean_R2 = np.mean(R2s)
    return mean_R2

def cross_validation_RMSE(X, y):
    RMSEs = []
    for i in range(10):
        model = GPR(kernel=kernel, n_restarts_optimizer=20)
        kf = KFold(n_splits=5, shuffle=True, random_state=i)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)
            best_model = GPR(kernel=model.kernel_, alpha=model.alpha, n_restarts_optimizer=20)
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
            RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
            RMSEs.append(RMSE)
    mean_RMSE = np.mean(RMSEs)
    return  mean_RMSE

def cross_validation_R2(X,y):
    R2s = []
    for i in range(10):
        model = GPR(kernel=kernel, n_restarts_optimizer=20)
        kf = KFold(n_splits=5, shuffle=True, random_state=i)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)
            best_model = GPR(kernel=model.kernel_, alpha=model.alpha, n_restarts_optimizer=20)
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
            R2 = r2_score(y_test, y_pred)
            R2s.append(R2)
    mean_R2 = np.mean(R2s)
    return  mean_R2

kernel = ConstantKernel(constant_value=1) * (Matern(length_scale=1, nu=1.5) + WhiteKernel(noise_level=1))
# gpr = GPR(kernel=kernel, n_restarts_optimizer=20)
# gpr.fit(X, y)
# regressor = GPR(kernel=gpr.kernel_, alpha=gpr.alpha,n_restarts_optimizer=20)
# regressor.fit(X, y)
print(cross_validation_RMSE(X, y))