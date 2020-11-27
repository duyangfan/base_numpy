#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/11/27 10:01
#@Author: dudu
#@File  : numpy_ces.py


import numpy as np

##创建数组
def defaut_careate():
    arr = np.array([[1,2,3,4,5],[5,6,7,8,9]])

    return arr;

def arange_create(begin,end,step):
    return np.arange(begin,end,step)


def num_create():
    return np.zeros((2,2),int)
def one_create():
    return np.ones(10,dtype='bool')







def print_msg(arr):
    print(arr, type(arr))
    print(arr * 10)
    print(arr + arr)
    print(arr * arr)
    print(arr.dtype)
    print(arr.ndim)
    print(arr.size)
    print(arr.shape)
    print(arr.data)




if __name__ =='__main__':
    print("numpy");
    arr=defaut_careate();
    print_msg(arr)
    print("=============")
    arr=arange_create(2,5,1)
    print_msg(arr)
    print("=============")
    arr = num_create()
    print_msg(arr)
    print("=============")
    arr = one_create()
    print_msg(arr)