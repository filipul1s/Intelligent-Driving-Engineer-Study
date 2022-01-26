# -*- coding:utf-8 -*-
# @Date :2022/1/20 23:04
# @Author:KittyLess
# @name: 2.Gradient Descent Code

import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

learnrate = 0.5
x = np.array([1,2])
y = np.array(0.5)

w = np.array([0.5,-0.5])

nn_output = sigmoid(np.dot(x,w))

error = y - nn_output

del_w = learnrate * error * nn_output * (1-nn_output) * x

print('Neural Network output:')
print(nn_output)
print('Amount of Error:')
print(error)
print('Change in Weights:')
print(del_w)