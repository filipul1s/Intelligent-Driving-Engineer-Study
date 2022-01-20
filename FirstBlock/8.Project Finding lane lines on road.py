# -*- coding:utf-8 -*-
# @Date :2022/1/19 22:37
# @Author:KittyLess
# @name: 8.Project Finding lane lines on road

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

import imageio
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def Lane_Lines_Detection(currFrame):
    FrameShape = currFrame.shape # FrameShape[0] represent y,FrameShape[1] represent xf
    # step1 灰度化
    gray = cv2.cvtColor(currFrame,cv2.COLOR_BGR2GRAY)
    # step2 gaussianBlur
    kernerl_size = 5
    blur_gray = cv2.GaussianBlur(currFrame,(kernerl_size,kernerl_size),0)
    # step3 Canny detect edges
    low_threshold = 100
    high_threshold = 200
    edges = cv2.Canny(blur_gray,low_threshold,high_threshold)
    # step4 set regionMask
    mask = np.zeros_like(edges)
    ignore_mask_color = 255
    vertices = np.array([[(230,660),(550, 450), (700, 450), (1150,660)]], dtype=np.int32)
    cv2.fillPoly(mask,vertices,ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges,mask)
    # step5 HoughLineP
    rho = 1
    theta = np.pi/180
    threshold = 50
    min_line_length = 70
    max_line_gap = 100
    line_image = np.copy(currFrame) * 0

    lines = cv2.HoughLinesP(masked_edges,rho,theta,threshold,np.array([]),min_line_length,max_line_gap)

    for line in lines:
        for x1,y1,x2,y2 in line:
            angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi
            if ((angle > 10) | (angle < -10)):
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

    # plt.imshow(line_image)
    # plt.show()
    color_edges = np.dstack((edges,edges,edges))

    combo = cv2.addWeighted(currFrame,0.6,line_image,1,0)
    return combo

new_clip_output = 'test_output.mp4'
LaneVidio_clip = VideoFileClip("test.mp4")
new_clip = LaneVidio_clip.fl_image(lambda x: Lane_Lines_Detection(x))
new_clip.write_videofile(new_clip_output,audio=False)
