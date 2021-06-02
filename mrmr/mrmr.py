import numpy as np
import mifs

if __name__ == '__main__':
    #生成数据
    d=100
    sample_num=1000
    cov=np.identity(d)
    mean1=[]
    for i in range(d):
        mean1.append(1/(i+1))
    mean1=np.sqrt(mean1)
    mean2=-1*mean1
    m1 = np.random.multivariate_normal(mean1, cov, sample_num)
    m2 = np.random.multivariate_normal(mean2, cov, sample_num)
    data=np.vstack((m1,m2))
    print("数据集样本数: ",data.shape[0])
    print("数据集特征数: ",data.shape[1])
    label=[0]*sample_num
    label.extend([1]*sample_num)
    # task2
    for k in range(1,11):
        feat_selector = mifs.MutualInformationFeatureSelector(method='MRMR',n_features=k)
        feat_selector.fit(data, np.array(label))
        print("取{}个特征：".format(k),feat_selector.ranking_)
    # task3
    n1=10
    n2=10000
    m1 = np.random.multivariate_normal(mean1, cov, n1)
    m2 = np.random.multivariate_normal(mean2, cov, n1)
    data1 = np.vstack((m1, m2))
    label1 = [0] * n1
    label1.extend([1] * n1)
    m1 = np.random.multivariate_normal(mean1, cov, n2)
    m2 = np.random.multivariate_normal(mean2, cov, n2)
    data2 = np.vstack((m1, m2))
    label2 = [0] * n2
    label2.extend([1] * n2)

    feat_selector = mifs.MutualInformationFeatureSelector(method='MRMR', n_features=10)
    feat_selector.fit(data1, np.array(label1))
    print("样本数为{}个时取10个特征：".format(n1), feat_selector.ranking_)
    feat_selector = mifs.MutualInformationFeatureSelector(method='MRMR', n_features=10)
    feat_selector.fit(data2, np.array(label2))
    print("样本数为{}个时取10个特征：".format(n2), feat_selector.ranking_)