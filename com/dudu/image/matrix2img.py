#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/10/16 9:09
#@Author: dudu
#@File  : matrix2img.py

from PIL import Image
import numpy as np

def saveImg(arr):
    if arr is  None:
        return
    matrix=np.matrix(arr)
    matrix=matrix.T
    img=Image.fromarray(matrix.astype(np.uint8))
    img.save("D://imageInfo//res_img.jpg")
'''
创建单位阵
'''
def createArr(height,width):
    arr=np.zeros((height,width))
    i=0
    j=0
    for i in range(0,height,1):
        for j in range(0, width, 1):
            if i==j:
                arr[i][j]=1
                continue
            if (i+j)<=255:
                arr[i][j]=i+j
                continue
            arr[i][j]=255
    return arr








if __name__ =='__main__':
    arr=createArr(500,500)
    saveImg(arr)
