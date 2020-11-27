#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/10/15 14:59
#@Author: dudu
#@File  : ReadImage.py

import cv2 as cv

def read_img(path,name):
    img=cv.imread(path+name)

