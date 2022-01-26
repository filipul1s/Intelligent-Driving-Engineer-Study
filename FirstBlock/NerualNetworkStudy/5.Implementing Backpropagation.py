# -*- coding:utf-8 -*-
# @Date :2022/1/24 18:30
# @Author:KittyLess
# @name: 5.Implementing Backpropagation

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.array([0.5,0.1,-0.2])
target = 0.6
learnrate = 0.5

weights_input_hidden = np.array([[0.5, -0.6],
                                 [0.1, -0.2],
                                 [0.1, 0.7]])

weights_hidden_output = np.array([0.1, -0.3])

hidden_layer_input = np.dot(x,weights_input_hidden)
hidden_layer_output = sigmoid(hidden_layer_input)

output_layer_in = np.dot(hidden_layer_output,weights_hidden_output)
output = sigmoid(output_layer_in)

# Backwards
error = target - output

del_err_output = error * output * (1 - output)

del_err_hidden = np.dot(del_err_output,weights_hidden_output) * \
                hidden_layer_output * ( 1 - hidden_layer_output)

delta_w_h_o = learnrate * del_err_output * hidden_layer_output

delta_w_i_h = learnrate * del_err_hidden * x[:,None]

print('Change in weights for hidden layer to output layer:')
print(delta_w_h_o)
print('Change in weights for input layer to hidden layer:')
print(delta_w_i_h)