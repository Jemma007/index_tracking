import random
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import SparsePCA
from sklearn.preprocessing import StandardScaler

SEED = 2022
random.seed(SEED)

# 399108.SZ

def split_train_test(df, ratio=0.7):
    idxs = list(range(df.shape[0]))
    train_idxs = idxs[:int(df.shape[0]*ratio)]
    test_idxs = idxs[int(df.shape[0]*ratio):]
    train_data = df.iloc[train_idxs, :].copy()
    test_data = df.iloc[test_idxs, :].copy()
    train_X = train_data.loc[:, ~df.columns.isin(['399108.SZ', 'time'])]
    train_Y = train_data.loc[:, df.columns.isin(['399108.SZ'])]
    test_X = test_data.loc[:, ~df.columns.isin(['399108.SZ', 'time'])]
    test_Y = test_data.loc[:, df.columns.isin(['399108.SZ'])]
    return train_X, train_Y, test_X, test_Y

def use_lasso(train_X, train_Y, test_X, test_Y, alpha=0.05):
    model = Lasso(alpha)  # 通过调整alpha，可以调整使用的股票数量
    # alpha增大，股票数量变少，同时MSE也会变大
    model.fit(train_X, train_Y)

    # 选出系数超出阈值的股票
    threshold = 1e-5
    all_stocks = szb_data.columns[~szb_data.columns.isin(['399108.SZ', 'time'])]
    flag = abs(model.coef_) > threshold
    left_stocks = all_stocks[flag]

    Y_pre = np.dot(test_X[:, flag], model.coef_[flag].reshape(-1, 1)) + model.intercept_
    # Y_pre = Y_scale.inverse_transform(Y_pre)
    test_Y = Y_scale.transform(test_Y)
    test_MSE = mean_squared_error(Y_pre, test_Y)

    train_Y_pre = model.predict(train_X)
    # train_Y_pre = Y_scale.inverse_transform(train_Y_pre.reshape(-1, 1))
    # train_Y = Y_scale.inverse_transform(train_Y)
    train_MSE = mean_squared_error(train_Y_pre, train_Y)

    print('使用股票个数：', len(left_stocks))
    print('训练集均方误差', train_MSE)
    print('测试集均方误差：', test_MSE)

def use_pca(train_X, train_Y, test_X, test_Y, n=10):
    select = SparsePCA(n_components=n, random_state=SEED)
    transformed_train_X = select.fit_transform(train_X)
    transformed_test_X = select.transform(test_X)
    model = LinearRegression()
    model.fit(transformed_train_X, train_Y)

    Y_pre = model.predict(transformed_test_X)
    # Y_pre = Y_scale.inverse_transform(Y_pre)
    test_Y = Y_scale.transform(test_Y)
    test_MSE = mean_squared_error(Y_pre, test_Y)

    train_Y_pre = model.predict(transformed_train_X)
    # train_Y_pre = Y_scale.inverse_transform(train_Y_pre)
    # train_Y = Y_scale.inverse_transform(train_Y)
    train_MSE = mean_squared_error(train_Y_pre, train_Y)

    print('使用股票个数：', n)
    print('训练集均方误差', train_MSE)
    print('测试集均方误差：', test_MSE)

if __name__ == '__main__':
    szb_data = pd.read_csv('./data/szb_data.csv', index_col=0)
    train_X, train_Y, test_X, test_Y = split_train_test(szb_data)

    X_scale = StandardScaler()
    train_X = X_scale.fit_transform(train_X.values)
    test_X = X_scale.transform(test_X.values)
    Y_scale = StandardScaler()
    train_Y = Y_scale.fit_transform(train_Y.values)

    ## 使用LASSO进行指数追踪
    train_X_lasso, train_Y_lasso, test_X_lasso, test_Y_lasso = train_X.copy(), train_Y.copy(), test_X.copy(), test_Y.copy()
    use_lasso(train_X_lasso, train_Y_lasso, test_X_lasso, test_Y_lasso)

    ## 使用稀疏主成分进行指数追踪
    train_X_pca, train_Y_pca, test_X_pca, test_Y_pca = train_X.copy(), train_Y.copy(), test_X.copy(), test_Y.copy()
    use_pca(train_X_pca, train_Y_pca, test_X_pca, test_Y_pca)