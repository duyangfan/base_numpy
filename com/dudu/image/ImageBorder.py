#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/10/15 10:18
#@Author: dudu
#@File  : ImageBorder.py

import cv2
import numpy as np

image_path="D://imageInfo//"
IMAGE_NAME = "apple.jpg"
SAVE_IMAGE_NAME = "res_"+IMAGE_NAME
'读取图片'
img = cv2.imread(image_path+IMAGE_NAME)
'转化图片转化'
img2gray = cv2.cvtColor(img,cv2.COLOR_RGB2BGRA)
'高斯平滑处理'
img2gray = cv2.GaussianBlur(img2gray,(3,3),0)
'边缘检测 参数一：单通道图像，参数二 min阙值，参数三：max阙值：'
c_canny_img = cv2.Canny(img2gray,50,160)
canny_arr=np.array(c_canny_img)
np.savetxt("D:\imageInfo\\arr.txt", canny_arr,fmt="%d")





cv2.imshow('mask',c_canny_img)
'等待毫秒数'
cv2.waitKey(2000)
'输出'
cv2.imwrite(image_path+SAVE_IMAGE_NAME,c_canny_img,[int(cv2.IMWRITE_JPEG_QUALITY),95])

