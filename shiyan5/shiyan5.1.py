# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 输入数据
X = np.array([12, 11, 9, 6, 8, 10, 12, 7])
Y = np.array([3.54, 3.01, 3.09, 2.48, 2.56, 3.36, 3.18, 2.65])

# 定义损失函数的偏导数
def dL_dslope(slope, intercept):
    return -2 * np.sum(X * (Y - (slope*X + intercept)))

def dL_dintercept(slope, intercept):
    return -2 * np.sum(Y - (slope*X + intercept))

# 定义牛顿法迭代更新步骤
def newton_method(slope, intercept, learning_rate, iterations):
    for _ in range(iterations):
        slope -= learning_rate * dL_dslope(slope, intercept)
        intercept -= learning_rate * dL_dintercept(slope, intercept)
    return slope, intercept

# 初始值和迭代次数
initial_slope = 0
initial_intercept = 0
learning_rate = 0.001
iterations = 1000

# 使用牛顿法求解
slope_newton, intercept_newton = newton_method(initial_slope, initial_intercept, learning_rate, iterations)

# 预测6.5岁时的尿肌酐含量
x_pred = 6.5
y_pred_newton = slope_newton * x_pred + intercept_newton
print("牛顿法预测结果：", y_pred_newton)

# 绘制输入数据和拟合直线图像
plt.scatter(X, Y)
plt.plot(X, slope_newton*X+intercept_newton, 'r')
plt.xlabel('年龄')
plt.ylabel('尿肌酐含量')
plt.title('年龄与尿肌酐含量关系图')
plt.show()
