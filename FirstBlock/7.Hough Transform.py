# -*- coding:utf-8 -*-
# @Date :2022/1/19 10:28
# @Author:KittyLess
# @name: 5.Canny to Detet Lane Lines

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np

# 思路：首先利用Canny算子获得轮廓之后，用fillPoly函数生成roi区域，
# 然后将edges图和mask图进行按位与相加运算即(bitwise_and)
# 之后调用cv2.HoughLinesP函数检测直线 然后输出到masked_edges中
# 最后调用cv2.addWeighted

image = mpimg.imread('test2.jpg')
imshape = [image.shape[0],image.shape[1]]

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# Define a kernel size for gaussian smoothing
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size,kernel_size),0)

low_threshod = 50
high_threshold = 150
edges = cv2.Canny(blur_gray,low_threshod,high_threshold)

mask = np.zeros_like(edges)
ignore_mask_color = 255
vertices = np.array([[(90,imshape[0]),(450, 290), (490, 290), (900,imshape[0])]], dtype=np.int32)
cv2.fillPoly(mask,vertices,ignore_mask_color)
masked_edges = cv2.bitwise_and(edges, mask)

rho = 1
theta = np.pi/180
threshold = 15
min_line_length = 40
max_line_gap = 20
line_image = np.copy(image) * 0 # creating a blank to draw lines on
region_select = np.copy(image)

lines = cv2.HoughLinesP(masked_edges,rho,theta,threshold,np.array([]),
                        min_line_length,max_line_gap)
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

color_edges = np.dstack((edges,edges,edges))

combo1 = cv2.addWeighted(color_edges,0.8,line_image,1,0)
plt.imshow(combo1)
plt.show()