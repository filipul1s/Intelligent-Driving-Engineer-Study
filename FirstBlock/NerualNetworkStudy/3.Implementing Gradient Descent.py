# -*- coding:utf-8 -*-
# @Date :2022/1/23 15:22
# @Author:KittyLess
# @name: 3.Implementing Gradient Descent
import pandas as pd
import numpy as np

data_path = "binary.csv"

riders = pd.read_csv(data_path)
#print(riders.head())

# step1 data cleaning
# dummy variables
rand_dummy = pd.get_dummies(riders['rank'],prefix='rank')
riders = pd.concat([riders,rand_dummy],axis=1)
riders = riders.drop('rank',axis=1)

# standarize features
for filed in ['gre','gpa']:
    mean,std = riders[filed].mean(),riders[filed].std()
    riders.loc[:,filed] = (riders[filed] - mean) / std

# split off random 10% of the data for testing
np.random.seed(42)
sample = np.random.choice(riders.index,size=int(len(riders)*0.9),replace=False)
data,test_data = riders.loc[sample,:],riders.drop(sample)

# split into features and targets
features, targets = data.drop('admit', axis=1), data['admit']
features_test, targets_test = test_data.drop('admit', axis=1), test_data['admit']


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

np.random.seed(42)

n_records,n_features = features.shape
print(features.shape)
last_loss = None

# initialize weights
weights = np.random.normal(scale = 1/n_features**.5,size=n_features)

epoches = 1000
learnrate = 0.5

for e in range(epoches):
    del_w = np.zeros(weights.shape)
    for x,y in zip(features.values,targets):
        output = sigmoid(np.dot(x,weights))
        error = y - output
        del_w += error * output * (1 - output) * x
    weights += learnrate*del_w/n_records

    if e % (epoches / 10) == 0:
        out = sigmoid(np.dot(features,weights))
        loss = np.mean((out - targets) ** 2)
        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss

# Calculate accuracy on test data
tes_out = sigmoid(np.dot(features_test, weights))
predictions = tes_out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))
