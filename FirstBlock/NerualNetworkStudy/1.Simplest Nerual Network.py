# -*- coding:utf-8 -*-
# @Date :2022/1/20 20:32
# @Author:KittyLess
# @name: 1.Simplest Nerual Network

import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

inputs = np.array([0.7,-0.3])
weights = np.array([0.1,0.8])
bias = -0.1

output = sigmoid(np.dot(weights, inputs) + bias)
#output = sigmoid(weights.transpose()*inputs + bias)

print('Output:',output)