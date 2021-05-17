import numpy as np
import matplotlib.pyplot as plt

class Perception(object):
    # 初始化权重
    def __init__(self):
        self.weights = np.random.random()
        self.bias = 0

    # 定义训练函数，训练集input_vecs,标签集labels,学习率rate
    def train(self, input_vecs, labels, rate):

        round = 0  # 迭代次数
        while(True):
            if(round>100):break;#陷入死循环
            round+=1
            round_errors=0

            for input_vec, label in zip(input_vecs, labels):
                if (sum(np.array(input_vec) * self.weights) + self.bias)*label <= 0:
                    round_errors+=1
                    # 更新权重
                    self.weights += rate * np.array(input_vec) * label
                    self.bias += rate*label
                    #print('g(x) = {} * x1 + {} * x2 + {}'.format(self.weights[0], self.weights[1], self.bias))

            print("第{}轮中，训练错误数：样本数量={}:{}".format(round, round_errors, len(input_vecs)))
            if round_errors==0:
                break;

        return self.weights, self.bias ,round

def show(x,y):
    plt.scatter(x, y, s=20)
    plt.title('perceptron', fontsize=24)

    plt.xlabel('x')
    plt.ylabel('y')
    my_x_ticks = np.arange(0,1.5, 0.5)
    my_y_ticks = np.arange(0,1.5, 0.5)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)

    plt.show()


if __name__ == '__main__':
    #生成数据
    n=2
    sample_num=300
    sigma=0.012;
    cov=sigma*np.identity(n)
    mean=[[1,1],[1,0],[0,1],[0,0]]
    inputt=[]
    for i in range(4):
        inputt.append(np.random.multivariate_normal(mean[i],cov,sample_num))
    inputdata=np.vstack((inputt[0],inputt[1],inputt[2],inputt[3]))

    x,y=inputdata.T
    #show(x,y)

    #生成标签
    labels=np.zeros(4*sample_num)
    for i in range(4):
        for j in range(sample_num):
            if i :
                labels[i*sample_num+j]=-1
            else:
                labels[i * sample_num + j] = 1

    rate=0.1#学习率（决定步长）
    p=Perception()
    p.train(inputdata,labels,rate)
    px=np.array([0,2])
    py=np.zeros(2)
    for i in range(len(px)):
        py[i]=(p.weights[0]/p.weights[1]*px[i]+p.bias/p.weights[1])*(-1)
    plt.plot(px,py)
    plt.text(0.5, 1, 'sigma={}'.format(sigma),fontsize=16)
    show(x,y)


