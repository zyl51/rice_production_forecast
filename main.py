import pandas as pd                              # 导入 pandas 库，用于数据处理和分析
import numpy as np                               # 导入 numpy 库，用于科学计算
import matplotlib.pyplot as plt                  # 导入 matplotlib 库，用于数据可视化
from sklearn.ensemble import RandomForestRegressor# 导入随机森林回归模型
from sklearn.model_selection import train_test_split # 导入用于数据集拆分的函数

# 读取数据
# 读取 csv 格式的文件，文件名为 'rice_production1.csv'，保存到变量 df 中
df = pd.read_csv('rice_production.csv')
# 将过小和过大的数据进行删除，以免结果具有特殊性
df = df.loc[df['production'] <= 1000]
df = df.loc[df['production'] >= 50]

# 可视化数据集特征分布
# 创建 2 行 3 列的图形窗口，并设置窗口大小为 (15, 10)，返回的子图对象保存在变量 axs 中
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# 在第 1 行 1 列的子图中画出温度特征的直方图，并分成 10 个区间
axs[0, 0].hist(df['avg_sunshine'], bins=10)
# 设置第 1 行 1 列的子图标题为 'Temperature'
axs[0, 0].set_title('avg_sunshine (hour/year)')

axs[0, 1].hist(df['avg_rain'], bins=10)
axs[0, 1].set_title('avg_rain (cm/year)')

axs[0, 2].hist(df['nitrogen'], bins=10)
axs[0, 2].set_title('nitrogen (g/ha)')

axs[1, 0].hist(df['potash'], bins=10)
axs[1, 0].set_title('potash (g/ha)')

axs[1, 1].hist(df['phosphate'], bins=10)
axs[1, 1].set_title('phosphate (g/ha)')

axs[1, 2].hist(df['production'], bins=10)
axs[1, 2].set_title('production (g/ha)')

# 归一化特征值, newValue = (oldValue - min) / (max - min)
# 将 df 进行备份
df_backup = df.copy()
# 取出数据部分
dataSet = df.values
minVals = dataSet.min(0)    # 求每一列的最小值，返回一维数组
maxVals = dataSet.max(0)    # 求每一列的最大值，返回一维数组
ranges = maxVals - minVals
# 创建一个和 dataSet 具有相同形状的零矩阵
# normDataSet = zeros(shape(dataSet))
m = dataSet.shape[0]
normDataSet = dataSet - np.tile(minVals, (m, 1))
normDataSet = normDataSet / np.tile(ranges, (m, 1))
df = pd.DataFrame(normDataSet, columns = df.columns.tolist())

# 拆分数据集为训练集和测试集
# 从数据集中去掉 production 特征，将其余特征保存到变量 X 中
X = df.drop('production', axis=1)
# 将 production 特征保存到变量 y 中
y = df['production']
# 函数返回值包含 4 个部分，分别是训练集的特征数据、测试集的特征数据、训练集的标签数据和测试集的标签数据
# 其顺序与传入的 X 参数一致。
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练随机森林模型， 用 n_estimators 设置决策树的数量
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# 传入训练数据集 X_train 和 训练目标值 y_train 拟合随机化森林回归模型
rf_model.fit(X_train, y_train)

# 预测测试集，调用自带函数对结果 production 进行预测
y_pred = rf_model.predict(X_test)

# 可视化预测结果
plt.figure()
# 绘制测试数据集中目标向量的真实 production 和预测p roduction，散点图
plt.scatter(y_test * ranges[-1] + minVals[-1], y_pred * ranges[-1] + minVals[-1])
# 绘制折线图，第一个参数为 x 数值范围，第二个参数为 y 的数值范围，`--k`表示黑色的虚线
plt.plot([0, 1000], [0, 1000], '--k')
# 分别在 x 轴和 y 轴写上标签
plt.xlabel('True Production')
plt.ylabel('Predicted Production')
plt.show()

# 可视化特征重要性
# 返回一个数组，表示随机森林模型中各特征对预测结果的重要性程度。是所有决策树中特征重要性的平均值。
importances = rf_model.feature_importances_
# 计算重要性标准差， rf_model.estimators_ 返回所有的决策树，
# 通过 np.std 函数对每个特征的重要性计算标准差，得到一个包含所有特征标准差的一维数组 std，
# 其中每个元素对应一个特征。
std = np.std([tree.feature_importances_ for tree in rf_model.estimators_], axis=0)
# 将特征的重要性从大到小排列，并返回排列后的特征索引
indices = np.argsort(importances)[::-1]
plt.figure()
# 设置柱状图标题
plt.title("Feature importances")
# 使用 plt.bar 绘制柱状图
plt.bar(range(X_train.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
# plt.xticks 函数用于设a置 x 轴的刻度标签
plt.xticks(range(X_train.shape[1]), X_train.columns[indices])
plt.show()

# 将预测值和真实值存入文件中，进行可持久化处理
with open('forecase_and_reality.csv', 'w') as file:
    # 先将标题输入
    file.write('forecase,reality,interpolation\n')
    # 将真实结果和预测结果输出
    for fcase, rel in zip(y_test, y_pred):
        # 将归一化的数据还原
        fcaset = fcase * ranges[-1] + minVals[-1]
        relt = rel * ranges[-1] + minVals[-1]
        # 取差值的绝对值t
        cha = fcaset - relt if fcaset - relt >=0 else relt - fcaset
        file.write(f'{fcaset},{relt},{cha}\n')
