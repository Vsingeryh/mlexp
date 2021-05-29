import numpy as np
import matplotlib.pyplot as plt

def kmeans(data, n, k):
    # 获取4个随机数
    rarray = np.random.random(size=k)
    # 乘以数据集大小——>数据集中随机的4个点
    rarray = np.floor(rarray*n)
    # 转为int
    rarray = rarray.astype(int)
    # 随机取数据集中的4个点作为初始中心点
    center = data[rarray]
    #标签集
    klabel = np.zeros([n], np.int)
    #print('初始center=\n', center)
    run = True
    time = 0
    while run:
        time = time + 1
        for i in range(n):
            # 求差
            tmp = data[i] - center
            # 求平方
            tmp = np.square(tmp)
            # axis=1表示按行求和
            tmp = np.sum(tmp, axis=1)
            # 取最小（最近）的给该点“染色”（标记每个样本所属的类(k[i])）
            klabel[i] = np.argmin(tmp)
        # 如果没有修改各分类中心点，就结束循环
        run = False
        # 计算更新每个类的中心点
        for i in range(k):
            # 找到属于该类的所有样本
            club = data[klabel==i]
            # axis=0表示按列求平均值，计算出新的中心点
            newcenter = np.mean(club, axis=0)
            # 如果新旧center的差距很小，看做他们相等，否则更新之。run置true，再来一次循环
            ss = np.abs(center[i]-newcenter)
            if np.sum(ss, axis=0) > 1e-5:
                center[i] = newcenter
                run = True
        #print('new center=\n', center)
    print('程序结束，迭代次数：', time)
    # 按类打印图表，因为每打印一次，颜色都不一样，所以可区分出来
    for k in range(4):
            plt.scatter(data[klabel==k,0],data[klabel==k,1],color=colors[k],marker=markers[k])
    #中心点
    for i, center in enumerate(center):
        plt.plot(center[0], center[1], color="black",marker="*", markersize=15)
    plt.show()

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
for i in range(label.shape[0]):
    for k in range(4):
            plt.scatter(point[i,0,label[i]==(k+1)],point[i,1,label[i]==(k+1)],color=colors[k],marker=markers[k])
    plt.show()

# 聚类数量
k = 4
n=label.shape[1]
# training
for idx in range(label.shape[0]):
    kmeans(point[idx].T,n,k)
