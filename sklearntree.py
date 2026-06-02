import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

x_train = np.array(list(range(1,11))).reshape(-1,1)
y_train = np.array([5.56,5.7,5.91,6.4,6.8,7.05,8.9,8.7,9,9.05])

#模型训练
#1.模型创建
estimator1=LinearRegression()   #线性回归
estimator2=DecisionTreeRegressor(max_depth=1)
estimator3=DecisionTreeRegressor(max_depth=3)
#2.模型训练
estimator1.fit(x_train,y_train)
estimator2.fit(x_train,y_train)
estimator3.fit(x_train,y_train)

#准备实验的特征数据
x_test = np.arange(0,10,0.1).reshape(-1,1)

#模型预测
y_pre1=estimator1.predict(x_test)
y_pre2=estimator2.predict(x_test)
y_pre3=estimator3.predict(x_test)

print(y_pre1)
print(y_pre2)
print(y_pre3)

plt.scatter(x_train,y_train)
plt.plot(x_test,y_pre1)
plt.plot(x_test,y_pre2)
plt.plot(x_test,y_pre3)
plt.legend()
plt.xlabel("date")
plt.ylabel("target")
plt.title("TREE REGRESSION")
plt.show()



