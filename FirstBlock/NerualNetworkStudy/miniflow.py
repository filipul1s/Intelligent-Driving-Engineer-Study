# -*- coding:utf-8 -*-
# @Date :2022/1/25 11:41
# @Author:KittyLess
# @name: miniflow

import numpy as np

class Node(object):
    def __init__(self,inbound_nodes=[]):
        self.inbound_nodes = inbound_nodes
        self.outbound_nodes = []
        self.value = None
        self.gradients = {}

        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)

    def forward(self):
        raise NotImplemented

    def backward(self):
        raise NotImplementedError

class Input(Node):
    def __init__(self):
        Node.__init__(self)

    def forward(self,value=None):
        if value is not None:
            self.value = value

    def backward(self):
        self.gradients = {self:0}
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            self.gradients[self] += grad_cost * 1

class Add(Node):
    def __init__(self,*inputs):
        Node.__init__(self,inputs)

    def forward(self):
        x = self.inbound_nodes[0].value
        y = self.inbound_nodes[1].value
        self.value = x + y

class Linear(Node):
    def __init__(self,X,W,b):
        Node.__init__(self,[X,W,b])

    def forward(self):
        self.value = np.dot(self.inbound_nodes[0].value,self.inbound_nodes[1].value) + self.inbound_nodes[2].value

    def backward(self):
        self.gradients = {n:np.zeros_like(n.value) for n in self.inbound_nodes}
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            self.gradients[self.inbound_nodes[0]] += np.dot(grad_cost,self.inbound_nodes[1].value.T)
            self.gradients[self.inbound_nodes[1]] += np.dot(self.inbound_nodes[0].value.T, grad_cost)
            self.gradients[self.inbound_nodes[2]] += np.sum(grad_cost, axis=0, keepdims=False)

class Sigmoid(Node):
    def __init__(self,node):
        node.__init__(self,[node])
    def _sigmoid(self,x):
        return 1. / (1. + np.exp(-x))
    def forward(self):
        input_value = self.inbound_nodes[0].value
        self.value = self._sigmoid(input_value)
    def backward(self):
        self.gradients = {n:np.zeros_like(n.value) for n in self.inbound_nodes}

        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            output_value = self._sigmoid(self.inbound_nodes[0].value)
            self.gradients[self.inbound_nodes[0]] += output_value * (1 - output_value) * grad_cost

class MSE(Node):
    def __init__(self,y,a):
        Node.__init__(self,[y,a])
    def forward(self):
        y = self.inbound_nodes[0].value.reshape(-1, 1)
        a = self.inbound_nodes[1].value.reshape(-1, 1)
        self.m = self.inbound_nodes[0].value.shape[0]
        self.diff = y - a
        self.value = np.mean((y - a) ** 2)
    def backward(self):
        self.gradients[self.inbound_nodes[0]] = (2 / self.m) * self.diff
        self.gradients[self.inbound_nodes[1]] = (-2 / self.m) * self.diff

def gradient_descent_update(x,gradx,learning_rate):
    x = x - learning_rate*gradx
    return x

def forward_and_backward(graph):
    for n in graph:
        n.forward()
    for n in graph[::-1]:
        n.backward()

def sgd_update(trainables,learning_rate=1e-2):
    """
        Updates the value of each trainable with SGD.

        Arguments:

            `trainables`: A list of `Input` Nodes representing weights/biases.
            `learning_rate`: The learning rate.
    """
    for t in trainables:
        t.value -= learning_rate * t.gradients[t]

def topological_sort(feed_dict):
    """
    Sort generic nodes in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` node and the value is the respective value feed to that node.

    Returns a list of sorted nodes.
    """

    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_pass(output_node, sorted_nodes):
    """
    Performs a forward pass through a list of sorted nodes.

    Arguments:

        `output_node`: A node in the graph, should be the output node (have no outgoing edges).
        `sorted_nodes`: A topologically sorted list of nodes.

    Returns the output Node's value
    """

    for n in sorted_nodes:
        n.forward()

    return output_node.value
