# -*- coding:utf-8 -*-
# @Date :2022/1/26 10:51
# @Author:KittyLess
# @name: 11.Gradient Descent

import random
from miniflow import gradient_descent_update
def f(x):
    """
    Quadratic function.

    It's easy to see the minimum value of the function
    is 5 when is x=0.
    """
    return x**2 + 5

def df(x):
    """
    Derivative of `f` with respect to `x`.
    """
    return 2*x

x = random.randint(0, 10000)
# TODO: Set the learning rate
learning_rate = 0.1
epochs = 100

for i in range(epochs+1):
    cost = f(x)
    gradx = df(x)
    print("EPOCH {}: Cost = {:.3f}, x = {:.3f}".format(i, cost, gradx))
    x = gradient_descent_update(x, gradx, learning_rate)