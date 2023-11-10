import numpy as np
import matplotlib.pyplot as plt

# 员工数据
data = np.array([
    [25, 2, 3, 4.5],
    [30, 5, 8, 7.2],
    [35, 8, 12, 9.6],
    [28, 3, 4, 5.0],
    [32, 6, 10, 8.3],
    [40, 10, 15, 12.5],
    [45, 12, 20, 15.2],
    [38, 9, 13, 10.1],
    [29, 4, 6, 6.8],
    [33, 7, 11, 8.9]
])

# 特征缩放（标准化）
def feature_normalize(X, mean=None, std=None):
    if mean is None and std is None:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
    X_normalized = (X - mean) / std
    return X_normalized, mean, std

# 添加偏置项
def add_bias(X):
    m = X.shape[0]
    ones = np.ones((m, 1))
    X_bias = np.concatenate((ones, X), axis=1)
    return X_bias

# 梯度下降法求解模型参数
def gradient_descent(X, y, learning_rate, num_iterations):
    m, n = X.shape
    w = np.zeros(n)  # 初始化权重向量
    history_cost = []  # 记录每次迭代的损失函数值
    for i in range(num_iterations):
        y_pred = np.dot(X, w)  # 预测值
        error = y_pred - y  # 误差
        cost = np.sum(error**2) / (2 * m)  # 均方误差
        history_cost.append(cost)
        dw = np.dot(X.T, error) / m  # 权重向量梯度
        w -= learning_rate * dw  # 更新权重向量
    return w, history_cost

# 数据预处理
X = data[:, :-1]  # 特征矩阵
y = data[:, -1]  # 标签向量
X_normalized, mean, std = feature_normalize(X)
X_bias = add_bias(X_normalized)

# 设置超参数
learning_rate = 0.01  # 学习率
num_iterations = 1000  # 迭代次数

# 使用梯度下降法求解模型参数
w, history_cost = gradient_descent(X_bias, y, learning_rate, num_iterations)

# 绘制拟合曲线
x = np.linspace(-2, 2, 100)
x_normalized = (x - mean[0]) / std[0]  # 标准化x
X_test = np.column_stack((np.ones(100), x_normalized))  # 添加偏置项
y_pred = np.dot(X_test, w)
plt.scatter(X[:, 0], y, label='Actual')  # 原始数据散点图
plt.plot(x, y_pred, color='red', label='Fitted')  # 拟合曲线
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()
