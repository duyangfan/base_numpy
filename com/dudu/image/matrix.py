#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/10/16 9:57
#@Author: dudu
#@File  : matrix.py

from PIL import Image
from scipy import signal
import numpy as np
'''
转置
'''
def matrix_t(types):
    arr = np.random.randint(0,255,(429,500))
    i=0
    j=0
    for i in range(0,429,1):
        for j in range(0, 500, 1):
            if i==j:
                arr[i][j]=0

    img=Image.open("D://imageInfo//apple.jpg")
    r,g,b=img.split()
    r_arr=np.array(r)
    r2_arr=np.array(g)
    i=0
    j=0
    for i in range(0,429,1):
        for j in range(0, 500, 1):
            if r_arr[i][j]>245:
                r_arr[i][j]=255
                continue
            r_arr[i][j]=0
            arr[i][j]=0
    matrix_show(np.matrix(r_arr),"r_add")
    print(r_arr)
    arr=arr+r_arr;
    print(r_arr.shape)
    core_arr=np.matrix([[1/16,2/16,1/16],
                        [2/16,4/16,2/16],
                        [1/16,2/16,1/16]])
    con_arr=signal.fftconvolve(r2_arr,core_arr,'same')
    #np.absolute(con_arr)
    #arr=arr.T
    return np.matrix(con_arr)

def matrix_show(matrix,name):
    print(matrix)
    img=Image.fromarray(matrix.astype(np.uint8))
    img.save("D://imageInfo//"+name+".jpg")


def matrix_convonlve():
    img=Image.open("D://imageInfo//b.jpg")
    r,g,b=img.split()
    img_arr=np.array(r)
    core_arr = np.matrix([[1 / 16, 2 / 16, 1 / 16],
                          [2 / 16, 4 / 16, 2 / 16],
                          [1 / 16, 2 / 16, 1 / 16]])

    core_arr = np.matrix([[-1,-1,-1],
                          [0,0,0],
                          [1 ,1,1]])

    core_arr = np.matrix([[-1, 0, -1],
                          [-1, 0, -1],
                          [-1, 0, -1]])
    core_arr = np.matrix([[1, 1, 1],
                          [1, -8, 1],
                          [1, 1, 1]])
    con_arr = signal.fftconvolve(img_arr, core_arr, 'same')
    img_mat=np.matrix(con_arr)

    matrix_show(img_mat,"createImage_matrix")

if __name__ =='__main__':
    # matrix=matrix_t(500)
    # matrix_show(matrix,"res_img")


    matrix_convonlve()