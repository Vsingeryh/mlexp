import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn import metrics

def get_data(path):
    df_list = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        df = pd.read_csv(file_path, error_bad_lines=False, encoding='ISO-8859-1')
        # print([column for column in df])
        df= df.drop('Unnamed: 0', axis=1)
        df_list.append(df)
    df = pd.concat(df_list, axis=1)
    return df

def get_len(path):
    len=[]
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        df = pd.read_csv(file_path, error_bad_lines=False, encoding='ISO-8859-1')
        df = df.drop('Unnamed: 0', axis=1)
        len.append(df.shape[1])
    return len

TEST_PATH = './data/'
# test_df = get_data(TEST_PATH)
# test_df.to_csv(path_or_buf="scrna.csv", index=False)  # 保存为CSV文件

# len=get_len(TEST_PATH)
# print(len)
len=[392, 251, 700, 459, 186]
labels=[]
for idx,l in enumerate(len):
    labels.extend([idx]*l)
path = '.\scrna.csv'
data = pd.read_csv(path, error_bad_lines=False, encoding='ISO-8859-1')
data = np.array(data)
# print(data.shape)
list =data.var(axis=1)
# print(list)
arridx=np.array(list).argsort()
newdata=(data[arridx[::-1]])[:2000]
# print(newdata.shape)
newdata=newdata.T
pca = PCA(n_components=50)
pca.fit(newdata)
PCADATA = pca.transform(newdata)
PCADATA2 = PCADATA[:, 0:2]
mark = ['or', 'ob', 'og', 'ok', 'oy', 'oc']
plt.title("PCA")
for idx,label in enumerate(labels):
    plt.plot(PCADATA2[idx, 0], PCADATA2[idx, 1], mark[label], markersize=2)
plt.show()

arrtesteps=np.arange(10,15,0.5)
maxNMI=0
for testeps in arrtesteps:
    for testmin in range(20,50,5):
        db = DBSCAN(eps=testeps, min_samples=testmin).fit(PCADATA)
        newlabel = db.labels_
        result_NMI = metrics.normalized_mutual_info_score(labels, newlabel)
        if maxNMI<result_NMI:
            maxNMI=result_NMI
            maxeps=testeps
            maxmin=testmin
        print("eps=",testeps,"\tmin_samples=",testmin,"\tresult_NMI:", result_NMI)
print("最接近的划分为:")
print("eps=",maxeps,"\tmin_samples=",maxmin,"Max:",maxNMI)
db = DBSCAN(eps=maxeps, min_samples=maxmin).fit(PCADATA)
newlabel = db.labels_
c=np.zeros(max(newlabel)+1)
sol=0
for idx in newlabel:
    if idx==-1:
        sol+=1
    else:
        c[idx]+=1
for idx in range(max(newlabel)+1):
    print("第{}类有{}个".format(idx+1,c[idx]))
print("孤立点有{}个".format(sol))