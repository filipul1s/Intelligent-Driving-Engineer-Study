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
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size,kernel_size),0)

low_threshod = 50
high_threshold = 150
edges = cv2.Canny(blur_gray,low_threshod,high_threshold)

#plt.imshow(edges,cmap='Greys_r')
#plt.show()

rho = 1
theta = np.pi/180
threshold = 15
min_line_length = 40
max_line_gap = 20
line_image = np.copy(image) * 0 # creating a blank to draw lines on

lines = cv2.HoughLinesP(edges,rho,theta,threshold,np.array([]),
                        min_line_length,max_line_gap)
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

color_edges = np.dstack((edges,edges,edges))

combo = cv2.addWeighted(color_edges,0.8,line_image,1,0)
plt.imshow(combo)
plt.show()