import numpy as np
import matplotlib.pyplot as mp

train_x = [0.5,0.6,0.8,1.1,1.4]
train_y = [5.0,5.5,6.0,6.8,7.0]

#实现梯度下降
times = 1000
lrate = 0.01
w0,w1 = [1],[1]#记录每次梯度下降的参数
for i in range(1,times+1):
    #每次梯度下降过程需要求出w0与w1的修正值
    #球修正值需要推导loss哈市南湖在w0与w1方向的偏岛
    d0 = (w0[-1]+w1[-1]*train_x-train_y)*sum()

#绘制样本点
mp.figure('Linear Regression',facecolor='lightgray')
mp.title('Linear Regression')
mp.grid(linestyle=':')
mp.scatter(train_x,train_y,s=60,marker='o',c='orangered',label='Samples')
mp.legend()
mp.show()