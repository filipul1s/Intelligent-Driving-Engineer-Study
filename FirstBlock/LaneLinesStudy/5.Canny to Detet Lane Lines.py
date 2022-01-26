# -*- coding:utf-8 -*-
# @Date :2022/1/19 10:28
# @Author:KittyLess
# @name: 5.Canny to Detet Lane Lines

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np

image = mpimg.imread('test2.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# Define a kernel size for gaussian smoothing
kernel_size = 3
blur_gray = cv2.GaussianBlur(gray,(kernel_size,kernel_size),0)

low_threshod = 100
high_threshold = 200
edges = cv2.Canny(blur_gray,low_threshod,high_threshold)

plt.imshow(edges,cmap='Greys_r')
plt.show()
