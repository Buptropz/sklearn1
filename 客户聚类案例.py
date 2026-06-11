from sklearn.cluster import KMeans     #聚类api
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, silhouette_score  # 评价指标

#1.定义函数，找聚类的质心数
def dm01_find_k():
    #读取数据
    data = pd.read_csv('date/customers.csv')
    #2.记录不同的k值的评估效果
    sse_list = []
    sc_list = []

    #抽取特征
    x = data.iloc[:, 3:5]

    #3.定义for，遍历k值，并计算评估效果
    for k in range(2,20):
        estimator = KMeans(n_clusters=k,max_iter=100,random_state=23)
        estimator.fit(x)
        y_pre = estimator.predict(x)
        sse_list.append(estimator.inertia_)
        sc_list.append(silhouette_score(x,y_pre))

    #4.绘制图像
    plt.figure(figsize=(8,6))
    plt.plot(range(2,20),sse_list, label='SSE')
    plt.show()

    plt.figure(figsize=(8,6))
    plt.plot(range(2,20),sc_list, label='SC')
    plt.show()

#2.定义函数，模型训练，模型预测
def dm02_train_model():
    df = pd.read_csv('date/customers.csv')

    #提取特征
    x = df.iloc[:,3:5]
