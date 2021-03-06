# -*- coding:utf-8 -*-
# @Date :2022/1/25 21:43
# @Author:KittyLess
# @name: 7.learning and loss

from miniflow import *

inputs,weights,bias = Input(),Input(),Input()

f = Linear(inputs,weights,bias)

feed_dict = {
    inputs : [6,14,3],
    weights : [0.5,0.25,1.4],
    bias : 2
}

graph = topological_sort(feed_dict)

output = forward_pass(f,graph)

print(output)