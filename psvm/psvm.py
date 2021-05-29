# _*_ coding: utf-8 _*_
import scipy.io as sio
import os
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def plot_point(dataArr, labelArr, Support_vector_index, W, b):
    for i in range(np.shape(dataArr)[0]):
        if labelArr[i] == 1:
            plt.scatter(dataArr[i][0], dataArr[i][1], c='b', s=20)
        else:
            plt.scatter(dataArr[i][0], dataArr[i][1], c='y', s=20)

    for j in Support_vector_index:
        plt.scatter(dataArr[j][0], dataArr[j][1], s=100, c='r', alpha=0.5, linewidth=1.5, edgecolor='red')

    x = np.arange(0, 4, 0.01)
    y = (W[0][0] * x + b) / (-1 * W[0][1])
    plt.scatter(x, y, s=5, marker='h')
    plt.title("linearSVM:   C=100")
    plt.show()

def plot_dataset(X, y, axes):
    plt.plot( X[:,0][y==0], X[:,1][y==0], "rs" )
    plt.plot( X[:,0][y==1], X[:,1][y==1], "y^" )
    plt.axis( axes )
    plt.grid( True, which="both" )
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")

def plot_predict(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    plt.contour(x0, x1, y_pred, cmap=plt.cm.winter, alpha=0.5)
    plt.contour(x0, x1, y_decision, cmap=plt.cm.winter, alpha=0.2)

if __name__ == "__main__":
    #线性
    load_plane1 = os.path.join(os.getcwd(), 'MLA2_data1.mat')  # mat文件路径
    plane1 = sio.loadmat(load_plane1)  # 使用scipy读入mat文件数据
    # print(plane1)
    # print(type(plane1))  # it's a dict
    X1=np.array(plane1["X"])
    y1=np.array(plane1["y"]).flatten()
    clf1 = SVC(C=100, kernel='linear')
    # fit训练数据
    clf1.fit(X1, y1)
    # 获取模型返回值
    n_Support_vector = clf1.n_support_  # 支持向量个数
    Support_vector_index = clf1.support_  # 支持向量索引
    W = clf1.coef_  # 方向向量W
    b = clf1.intercept_  # 截距项b

    # 绘制分类超平面
    plot_point(X1, y1, Support_vector_index, W, b)

    #高斯核
    load_plane2 = os.path.join(os.getcwd(), 'MLA2_data2.mat')  # mat文件路径
    plane2 = sio.loadmat(load_plane2)  # 使用scipy读入mat文件数据
    X2 = np.array(plane2["X"])
    y2 = np.array(plane2["y"]).flatten()
    rbf_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(C=0.01, kernel='rbf',gamma=10))
    ])
    rbf_kernel_svm_clf.fit(X2, y2)
    plot_dataset(X2, y2, [0, 1.1, 0.3, 1.1])
    plot_predict(rbf_kernel_svm_clf, [0, 1.1, 0.3, 1.1])
    plt.show()

