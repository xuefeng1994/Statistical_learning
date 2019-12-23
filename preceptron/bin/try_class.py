import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt 


iris = sklearn.datasets.load_iris()
df = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
df['label'] = iris.target
# df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label']
df.columns

plt.scatter(df[:50]['sepal_length'], df[:50]['sepal_width'], label='0')
plt.scatter(df[50:100]['sepal_length'], df[50:100]['sepal_width'], label='1')
plt.xlabel('sepal_length')
plt.ylabel('sepal_width')
plt.legend()

plt.scatter(df[:]['sepal_length'], df[:]['petal_length'], label = '0')
plt.xlabel('sepal_length')
plt.show()

df_test = df[['sepal_length', 'petal_length', 'label']]
plt.scatter(df_test[:]['sepal_length'], df_test[:]['petal_length'], label = df_test[:]['label'] )
plt.xlabel('sepal_length')
plt.show()

data = np.array(df.iloc[:100, [0, 1, -1]])   
len(data)

###初始化数据，标记修改为1或-1
X, y = data[:,:-1], data[:,-1]
y = np.array([1 if i == 1 else -1 for i in y])


####感知机模型
class Perceptron ():
    """docstring for ClassName"""
    def __init__(self):
        self.w = np.ones(len(data[0]) - 1, dtype=np.float32)
        self.b = 0
        self.l_rate = 0.2
    def fit(self, X_train, y_train):
        is_fit = 0
        while not is_fit:
            flag = 0
            for index in range(0, len(X_train)):
                sample = X_train[index]
                label = y_train[index]
                ###误分类点
                if -label * (np.dot(sample, self.w) + self.b) > 0:
                    self.w = self.w + self.l_rate * np.dot(sample, label)
                    self.b = self.b + self.l_rate * label
                    flag = 1
            if not flag:
                is_fit = 1                    

train = Perceptron()
train.w

train.fit(X, y)

train.w

x_points = np.linspace(4, 7, 10)
y_ = -(train.w[0] * x_points + train.b) / train.w[1]
df_test = df[['sepal_length', 'petal_length', 'label']]
plt.scatter(df_test[:]['sepal_length'], df_test[:]['petal_length'], label = df_test[:]['label'] )
plt.xlabel('sepal_length')
plt.plot(x_points, y_)


