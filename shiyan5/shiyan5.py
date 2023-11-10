import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 定义计算伪逆矩阵的函数
def calculate_pseudo_inverse(X):
    Xt = matrix_transpose(X)
    XtX_inv = matrix_inverse(matrix_multiply(Xt, X))
    pseudo_inv = matrix_multiply(XtX_inv, Xt)
    return pseudo_inv

# 矩阵转置函数
def matrix_transpose(X):
    # 手动实现矩阵转置操作
    Xt = np.array([[X[j][i] for j in range(len(X))] for i in range(len(X[0]))])
    return Xt

# 矩阵相乘函数
def matrix_multiply(A, B):
    # 手动实现矩阵相乘操作
    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                C[i][j] += A[i][k] * B[k][j]
    return C

# 矩阵求逆函数
# 矩阵求逆函数（不使用np.dot）
def matrix_inverse(X):
    n = X.shape[0]
    I = np.eye(n)
    inv = np.zeros((n, n))
    for i in range(n):
        # 手动实现高斯消元法求解线性方程组 X * inv[:,i] = I[:,i]，求解inv的第i列
        A = X.copy()
        b = I[:, i]
        for j in range(n):
            if A[j, j] == 0:
                # 如果对角线上的元素为0，则需要进行行交换
                for k in range(j + 1, n):
                    if A[k, j] != 0:
                        A[[j, k]] = A[[k, j]]
                        b[[j, k]] = b[[k, j]]
                        break
            for k in range(j + 1, n):
                if A[k, j] != 0:
                    # 使用初等行变换将A变为上三角矩阵
                    ratio = A[k, j] / A[j, j]
                    A[k] -= ratio * A[j]
                    b[k] -= ratio * b[j]
        # 回代过程，求解出inv的第i列
        inv[n-1, i] = b[n-1] / A[n-1, n-1]
        for j in range(n - 2, -1, -1):
            s = 0
            for k in range(j + 1, n):
                s += A[j, k] * inv[k, i]
            inv[j, i] = (b[j] - s) / A[j, j]
    return inv


# 数据
X = np.array([[12], [11], [9], [6], [8], [10], [12], [7]])
Y = np.array([[3.54], [3.01], [3.09], [2.48], [2.56], [3.36], [3.18], [2.65]])

# 向X添加一列1
ones = np.ones((X.shape[0], 1))
X = np.hstack((ones, X))

# 计算伪逆矩阵
pseudo_inv = calculate_pseudo_inverse(X)

# 拟合直线系数
beta = np.dot(pseudo_inv, Y)

# 预测六岁半的尿肌酐含量
age_6_half = np.array([[1, 6.5]])  # 六岁半对应的输入特征值
predicted_creatinine = np.dot(age_6_half, beta)

print("六岁半的正常儿童尿肌酐含量预测值为：", predicted_creatinine[0][0])

# 绘制拟合直线
plt.scatter(X[:, 1], Y, color='b', label='实际数据')  # 绘制原始数据点
plt.plot(X[:, 1], np.dot(X, beta), color='r', label='拟合直线')  # 绘制拟合直线
plt.xlabel('年龄')
plt.ylabel('尿肌酐含量')
plt.title('年龄与尿肌酐的含量关系（最小二乘（伪逆矩阵））')
plt.legend()
plt.show()