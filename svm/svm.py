import numpy as np
import matplotlib.pyplot as plt
import copy
import random

def poly(p):
    def _poly(x, y):
        return np.power(np.dot(x,y) + 1, p)
    return _poly

class SVC(object):
    def __init__(self, p,epochs=100, C=1.0, toler=1e-3, kernel=poly):
        """
        Parameters epochs: 最高迭代次数
        Parameters C: 惩罚系数
        Parameters toler:误差容忍度
        Parameters kernel:核函数
        """
        self.b = None
        self.alpha = None
        self.E = None
        self.epochs = epochs
        self.C = C
        self.tol = toler
        self.kernel = poly(p)
        # 支持向量
        self.support_vectors = None
        self.support_vector_x = []
        self.support_vector_y = []
        # 支持向量的alpha
        self.support_vector_alpha = []
    #预测结果的函数
    def f_pre(self, x):
        """
        Parameters x:
        Returns data: wx+b
        """
        x_np = np.asarray(x) #原地修改
        if len(self.support_vector_x) == 0:
            if x_np.ndim <= 1: #数组维度
                return 0
            else:
                return np.zeros((x_np.shape[:-1]))
        else:
            if x_np.ndim <= 1:
                wx = 0
            else:
                wx = np.zeros((x_np.shape[:-1]))
            for i in range(0, len(self.support_vector_x)):
                wx += self.kernel(x, self.support_vector_x[i]) * self.support_vector_alpha[i] * self.support_vector_y[i]
            return wx + self.b
    #初始化参数
    def init_params(self, X, Y):
        n_samples, n_features = X.shape
        self.b = .0
        self.alpha = np.zeros(n_samples)
        self.E = np.zeros(n_samples)
        for i in range(0, n_samples):
            self.E[i] = self.f_pre(X[i, :]) - Y[i]

    def select_j(self, best_i):
        """
        选择j
        Parameters best_i:
        """
        valid_j_list = [i for i in range(0, len(self.alpha)) if self.alpha[i] > 0 and i != best_i]
        best_j = -1
        # 优先选择使得|E_i-E_j|最大的j
        if len(valid_j_list) > 0:
            max_e = 0
            for j in valid_j_list:
                current_e = np.abs(self.E[best_i] - self.E[j])
                if current_e > max_e:
                    best_j = j
                    max_e = current_e
        else:
            # 随机选择
            l = list(range(len(self.alpha)))
            seq = l[: best_i] + l[best_i + 1:]
            best_j = random.choice(seq)
        return best_j

    def meet_kkt(self, x_i, y_i, alpha_i):
        if alpha_i < self.C:
            return y_i * self.f_pre(x_i) >= 1 - self.tol
        else:
            return y_i * self.f_pre(x_i) <= 1 + self.tol

    def fit(self, data, laber):
        y = copy.deepcopy(laber)
        y[y == 0] = -1
        # 初始化参数
        self.init_params(data, y)
        for _ in range(0, self.epochs):
            if_all_match_kkt = True
            for i in range(0, len(self.alpha)):
                x_i = data[i, :]
                y_i = y[i]
                alpha_i_old = self.alpha[i]
                E_i_old = self.E[i]
                # 外循环：选择违反KKT条件的点i
                if not self.meet_kkt(x_i, y_i, alpha_i_old):
                    if_all_match_kkt = False
                    # 内循环，选择使|Ei-Ej|最大的点j
                    best_j = self.select_j(i)
                    alpha_j_old = self.alpha[best_j]
                    x_j = data[best_j, :]
                    y_j = y[best_j]
                    E_j_old = self.E[best_j]

                    # 1.首先获取无裁剪的最优alpha_2
                    eta = self.kernel(x_i, x_i) + self.kernel(x_j, x_j) - 2.0 * self.kernel(
                        x_i, x_j)
                    # 如果x_i和x_j很接近，则跳过
                    if eta < 1e-3:
                        continue
                    alpha_j_unc = alpha_j_old + y_j * (E_i_old - E_j_old) / eta
                    # 2.裁剪并得到new alpha_2
                    if y_i == y_j:
                        L = max(0., alpha_i_old + alpha_j_old - self.C)
                        H = min(self.C, alpha_i_old + alpha_j_old)
                    else:
                        L = max(0, alpha_j_old - alpha_i_old)
                        H = min(self.C, self.C + alpha_j_old - alpha_i_old)

                    if alpha_j_unc < L:
                        alpha_j_new = L
                    elif alpha_j_unc > H:
                        alpha_j_new = H
                    else:
                        alpha_j_new = alpha_j_unc

                    # 如果变化不够大则跳过
                    if np.abs(alpha_j_new - alpha_j_old) < 1e-5:
                        continue
                    # 3.得到alpha_1_new
                    alpha_i_new = alpha_i_old + y_i * y_j * (alpha_j_old - alpha_j_new)
                    # 5.更新alpha_1,alpha_2
                    self.alpha[i] = alpha_i_new
                    self.alpha[best_j] = alpha_j_new
                    # 6.更新b
                    b_i_new = y_i - self.f_pre(x_i) + self.b
                    b_j_new = y_j - self.f_pre(x_j) + self.b
                    if self.C > alpha_i_new > 0:
                        self.b = b_i_new
                    elif self.C > alpha_j_new > 0:
                        self.b = b_j_new
                    else:
                        self.b = (b_i_new + b_j_new) / 2.0
                    # 7.更新E
                    for k in range(0, len(self.E)):
                        self.E[k] = self.f_pre(data[k, :]) - y[k]

                    # 8.更新支持向量相关的信息
                    self.support_vectors = np.where(self.alpha > 1e-3)[0]
                    self.support_vector_x = [data[i, :] for i in self.support_vectors]
                    self.support_vector_y = [y[i] for i in self.support_vectors]
                    self.support_vector_alpha = [self.alpha[i] for i in self.support_vectors]
            # 所有的点都满足KKT条件
            if if_all_match_kkt is True:
                break
    #分类
    def predict(self, x):
        """
        Parameters x:ndarray格式数据: m x n
        Returns data: m x 1
        """
        proba = self.f_pre(x)
        return (proba >= 0).astype(float)

def pltshow(data,labels,title):
    int_labels=map(int,list(labels))
    colors=['r','g']
    x,y = data.T
    for index,label in enumerate(int_labels):
        plt.scatter(x[index],y[index],color=colors[label])
    plt.xlabel('x')
    plt.ylabel("y")
    my_x_ticks = np.arange(-0.5, 2, 0.5)
    my_y_ticks = np.arange(-0.5, 2, 0.5)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
    plt.title(title, fontsize=15)
    plt.show()


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    #生成数据
    n=2
    sample_num=20
    sigma=0.008
    cov = sigma * np.identity(n)
    mean = [[1, 1], [1, 0], [0, 1], [0, 0]]

    posdata1= np.random.multivariate_normal(mean[0], cov, sample_num)
    posdata2 = np.random.multivariate_normal(mean[3], cov, sample_num)
    posd = np.vstack((posdata1,posdata2))
    negdata1 = np.random.multivariate_normal(mean[1], cov, sample_num)
    negdata2 = np.random.multivariate_normal(mean[2], cov, sample_num)
    negd = np.vstack((negdata1,negdata2))

    #数据集和标签集
    pos_num = posd.shape[0]
    neg_num = negd.shape[0]
    labels = np.ones((pos_num + neg_num, 1))
    labels[pos_num:] = 0
    train_data = np.vstack((posd, negd))
    DataMat = np.array(train_data, dtype='float32')
    Labels = np.array(labels.reshape(-1))
    pltshow(DataMat,Labels,title='STD_poly_svm')

    #预测
    for p in range(1,12):
        print("\n多项式核次方数p = ",p)
        svc = SVC(p)
        svc.fit(DataMat, Labels)
        print('标签集:')
        print(Labels)
        print("预测值:")
        pre=svc.predict(DataMat)
        print(pre)
        sum=0
        for en in range(len(Labels)):
            sum+=abs(np.float(Labels[en]-pre[en]))
        print('错误率err =',sum/len(Labels))
        pltshow(DataMat,pre,title='p={}时分类情况'.format(p))

