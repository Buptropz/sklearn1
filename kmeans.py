from sklearn.cluster import KMeans     #聚类api
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs         #默认按照高斯发布来生成数据集
from sklearn.metrics import calinski_harabasz_score          #评价指标

#参1：样本数 参2：特征数 参3：簇的位置 参4：标准差 参5：随机数种子
x,y = make_blobs(n_samples=1000,n_features=2,centers=[[-1,-1],[0,0],[1,1],[2,2]],cluster_std=1.0,random_state=0)

#2.绘制图像
#参1：x 参2：y 参3：颜色
plt.scatter(x[:,0],x[:,1])
#plt.show()

#3.创建KMeans模型
#参1：聚类数量 参2：随机数种子
estimatorK = KMeans(n_clusters=4,random_state=23)

#4.训练模型 和 预测
y_pre = estimatorK.fit_predict(x)

#5.绘制图像
plt.scatter(x[:,0],x[:,1],c=y_pre)
plt.show()

#6.评价指标
print(f'评价指标：{calinski_harabasz_score(x,y_pre)}')   #越大越好

