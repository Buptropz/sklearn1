from sklearn.cluster import KMeans     #聚类api
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs         #默认按照高斯发布来生成数据集
from sklearn.metrics import calinski_harabasz_score          #评价指标

#1.定义函数，演示SSE + 肘部法
def dm01_SSE():
    sse_list = []
    x,y=make_blobs(n_samples=1000,n_features=2,centers=[[-1,-1],[0,0],[1,1],[2,2]],cluster_std=[0.4,0.2,0.2,0.2],random_state=23)

    #遍历训练，获取每个k值，计算其SSE
    for k in range(1,100):
        estimator = KMeans(n_clusters=k,max_iter=100,random_state=23)
        estimator.fit(x)
        sse_list.append(estimator.inertia_)

    plt.figure(figsize=(8,6))
    plt.title('SSE')
    plt.xlabel(range(0,100,3))
    plt.ylabel('sse')
    plt.grid()
    #参1：x 参2：y 参3：颜色
    plt.plot(range(1,100),sse_list)
    plt.show()

if __name__ == '__main__':
    dm01_SSE()