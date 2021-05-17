import numpy as np
import matplotlib.pyplot as plt

rate=0.1 #ѧϰ��
sample_num=4 #����������

class my_mlp:
    def __init__(self, input_size, hidden_size, output_size):
        self.w1 = np.random.normal(size=(hidden_size, input_size))#����㵽���ز�
        self.w2 = np.random.normal(size=(hidden_size,output_size))#���ز㵽�����
        self.b1 = np.random.normal(size=(hidden_size))
        self.b2 = np.random.normal(size=(output_size))
        self.h_out = np.zeros(1)
        self.out = np.zeros(1)

    @staticmethod
    def sigmoid(x):
        '''sigmoid������Ϊ�����'''
        return 1 / (1 + np.exp(-x))
    @staticmethod
    def d_sigmoid(x):
        '''��������������������'''
        return x * (1 - x)
    def forward(self,input):
        self.h_out = my_mlp.sigmoid(np.dot(input, self.w1)+self.b1)
        self.out = my_mlp.sigmoid(np.dot(self.h_out, self.w2)+self.b2)

    def backpropagation(self,input,output,lr=rate):
        self.forward(input)
        L2_delta=(output-self.out) * my_mlp.d_sigmoid(self.out)
        L1_delta = L2_delta.dot(self.w2.T) * my_mlp.d_sigmoid(self.h_out)
        d_w2 = rate * self.h_out.T.dot(L2_delta)
        d_w1 = rate * input.T.dot(L1_delta)
        self.w2 += d_w2
        self.w1 += d_w1
        d_b2 = np.ones((1,sample_num)).dot(L2_delta)
        d_b1 = np.ones((1,sample_num)).dot(L1_delta)
        self.b2 += rate*d_b2.reshape(d_b2.shape[0]*d_b2.shape[1],)
        self.b1 += rate*d_b1.reshape(d_b1.shape[0]*d_b1.shape[1],)



if __name__ == '__main__':
    mlp=my_mlp(2,2,1)
    # x_data x1,x2
    x_data = np.array([[0, 0],
                       [0, 1],
                       [1, 0],
                       [1, 1]])
    # y_data label
    y_data = np.array([[0],
                       [1],
                       [1],
                       [0]])

    for i in range(15000):
        mlp.backpropagation(x_data,y_data)
        out=mlp.out  # ����Ȩֵ
        if i % 500 == 0:
            plt.scatter(i, np.mean(np.abs(y_data - out)))
            #print('��ǰ���:',np.mean(np.abs(y_data - out)))
    plt.title('Error Curve')
    plt.xlabel('iteration')
    plt.ylabel('Error')
    plt.show()
    print('����㵽������Ȩֵ:\n',mlp.w1)
    print('����㵽������ƫ�ã�\n',mlp.b1)
    print('�����㵽�����Ȩֵ��\n',mlp.w2)
    print('�����㵽�����ƫ�ã�\n',mlp.b2)

    print('������:\n',out)
    print('�������������:')
    for i in out:
        print(0 if i<=0.5 else 1)
