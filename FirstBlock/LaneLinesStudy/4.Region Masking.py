# -*- coding:utf-8 -*-
# @Date :2022/1/18 17:23
# @Author:KittyLess
# @name: 3.Color Selection Code

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

image = mpimg.imread('./test.jpg')
print('This image is: ',type(image),
      'with dimensions:',image.shape)

ysize = image.shape[0]
xsize = image.shape[1]
color_select = np.copy(image)
region_select = np.copy(image)
red_threshold = 200
green_threshold = 200
blue_threshold = 200
rgb_threshold = [red_threshold,green_threshold,blue_threshold]

# pixel low as rgb_threshold will be set up 0
color_thresholds = (image[:,:,0] < rgb_threshold[0]) \
            | (image[:,:,1] < rgb_threshold[1]) \
            | (image[:,:,2] < rgb_threshold[2])

color_select[color_thresholds] = [0,0,0]

# region masking

left_bottom = [70, 320]
right_bottom = [470, 320]
apex = [275, 180]

# Fit lines (y=Ax+B) to identify the  3 sided region of interest
# np.polyfit() returns the coefficients [A, B] of the fit

fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

XX,YY = np.meshgrid(np.arange(0,xsize),np.arange(0,ysize))
region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
                    (YY > (XX*fit_right[0] + fit_right[1])) & \
                    (YY < (XX*fit_bottom[0] + fit_bottom[1]))

region_select[~ color_thresholds & region_thresholds] = [255,0,0]

plt.imshow(region_select)
plt.show()