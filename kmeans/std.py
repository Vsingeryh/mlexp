import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 读取数据
pointpath=[".\cluster_data_dataA_X.txt",
       ".\cluster_data_dataB_X.txt",
       ".\cluster_data_dataC_X.txt"]
labelpath=[".\cluster_data_dataA_Y.txt",
           ".\cluster_data_dataB_Y.txt",
           ".\cluster_data_dataC_Y.txt"]

point=np.genfromtxt(pointpath[0],dtype='float64')
label=np.genfromtxt(labelpath[0],dtype='int')
point=point[np.newaxis,:]
label=label[np.newaxis,:]
for i in range(1,len(pointpath)):
    x=np.genfromtxt(pointpath[i],dtype='float64')
    y=np.genfromtxt(labelpath[i],dtype='int')
    x=x[np.newaxis,:]
    y=y[np.newaxis,:]
    point=np.vstack((point, x))
    label=np.vstack((label, y))
# print(point.shape) (3,2,200)
# print(label.shape) (3,200)

#画图
colors = ["red","blue","green","yellow"]
markers = ["o","v","+","x"]
# for i in range(label.shape[0]):
#     for k in range(4):
#             plt.scatter(point[i,0,label[i]==(k+1)],point[i,1,label[i]==(k+1)],color=colors[k],marker=markers[k])
#     plt.show()

# 聚类数量
k = 4
# training
for idx in range(label.shape[0]):
    model = KMeans(n_clusters=k)
    #print(point[idx].T.shape)
    model.fit(point[idx].T)
    centers=model.cluster_centers_
    result=model.predict(point[idx].T)
    for k in range(4):
            plt.scatter(point[idx,0,result==k],point[idx,1,result==k],color=colors[k],marker=markers[k])
    #中心点
    for i, center in enumerate(centers):
        plt.plot(center[0], center[1], color="black",marker="*", markersize=15)

    plt.show()