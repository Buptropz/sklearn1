import pandas as pd
from sklearn.preprocessing import LabelEncoder   #标签编码器
from sklearn.model_selection import train_test_split        #分割
from sklearn.tree import DecisionTreeClassifier            #决策树分类器
from sklearn.ensemble import AdaBoostClassifier         #adaBoost分类器（boost思想）
from sklearn.metrics import  accuracy_score

df_wine = pd.read_csv('date/wine0501.csv')

#数据预处理
#从标签中过滤掉1类别，剩下2，3类别
df_wine = df_wine[df_wine['Class label'] != 1]
#获得特征列和标签列
x=df_wine[['Alcohol','Hue']]
y=df_wine['Class label']
#通过标签编码器把标签列转换成数值列
le = LabelEncoder()
y = le.fit_transform(y)
#训练集和测试集分割
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=23,stratify=y)

#模型训练
#场景1：单一决策树为弱分类器
estimator1 = DecisionTreeClassifier(max_depth=3)
estimator1.fit(x_train,y_train)
y_pre1 = estimator1.predict(x_test)
print("场景1：单一决策树为弱分类器，准确率：",accuracy_score(y_test,y_pre1))

#场景2：AdaBoost分类器
#参一：弱分类器 参二：弱分类器的数量 参三：学习率 参四：算法
estimator2 = AdaBoostClassifier(estimator=estimator1,n_estimators=200,learning_rate=0.1)
estimator2.fit(x_train,y_train)
y_pre2 = estimator2.predict(x_test)
print("场景2：AdaBoost分类器，准确率：",accuracy_score(y_test,y_pre2))
