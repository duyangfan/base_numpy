#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/11/27 11:05
#@Author: dudu
#@File  : numpy_prop.py

import numpy as np

##ndarray对象属性的基本操作 shape
def arr_prop():
    arr=np.array([1,2,3,4,5,6])
    print(arr)
    ###按照总元素的个数进行分割
    arr.shape = (2,3)
    print(arr)
    print('dtype:',arr.dtype)
    arr=arr.astype('int32')
    print('dtype:',arr.dtype,arr)
    ##len 是返回外层元素个数
    print(arr.size,len(arr))

def get_arr():
    print("=====================================")
    arr=np.arange(1,9)
    print(arr.size)
    arr.shape=(2,2,2)
    print(arr,arr.shape[0])
    for i in range(arr.shape[0]):
        for j in range(arr.shape[0]):
            for k in range(arr.shape[0]):
                print(arr[i][j][k])


#自定义复合类型数据
def  fuhe():
    data=[('ZS',[90,88,82],15),
          ('LS',[91,84,89],15),
          ('WU',[92,87,85],15)];
    #设置dtype的方式：U2表示：字符 2
    arr=np.array(data,dtype="U2,3int32,int32")
    print(arr)
    print(arr[0][2])
    print(arr.shape)
    #设置dtype别名方式：元组方式
    arr=np.array(data,dtype=[('name','str',2),
                             ('scores','int32',3),
                             ('age','int32',)])
    print(arr,'LS scores:',arr[2]['scores'])
    #设置dtype的别名方式：字典方式
    arr=np.array(data,dtype={'names':['name','scores','age'],
                             'formats':['U2','3int32','int32']})
    print(arr, 'ZS scores:', arr[1]['scores'])
    #测试日期类型数组 datetime64
    data=['2011','2012-11-25','2015-09-12','2015-09-20']
    arr=np.array(data)
    print(arr,arr.dtype)
    # 精确到day的date
    arr=arr.astype('M8[D]')
    print(arr,arr.dtype)
    print('days:',arr[3]-arr[2])
    '''
      类型字符码：
      np.bool                       ?
      np.str                        U<字符数>
      np.datetime74                 M8[Y]
      np.int8/int16/32/64           i1/i2/i4/i8
      np.float/16/32/64                f2/f4/f8
    '''

    #维度操作
def change_shape():
    arr=np.arange(1,21)
    print(arr.size)
    #视图变维 reshape
    arr_4_5=arr.reshape((4,5))
    print(arr_4_5)
    #扁平化
    arr_1=arr.ravel()
    print(arr_1)
    #复制变维 扁平化
    arr_clone=arr_4_5.flatten()
    arr_clone[0]=500
    print(arr_clone)
    print(arr)

'''
   ndarray数组切片
'''
def arr_qie():
    arr=np.arange(1,10)
    print(arr)
    arr_re=arr[::-1]
    print(arr_re)
    '''
        arr[起始下标:终止下标:步长]
        多维数组的切片
        arr[begin:end:step,begin:end:step]:表示二维数组的切片,有逗号分隔
    '''
    #多维数组切片
    arr=np.arange(1,21)
    arr=arr.reshape((4,5))
    print(arr)
    print(arr[:,:2])

'''
    ndarry数组的掩码操作
    arr[表达式]
    使用掩码排序
'''
def mask_fun():
    arr=np.arange(1,4)
    arr=arr.reshape((3,1))
    print(arr)
    arr=np.arange(1,4)
    mask=arr[(arr % 2==0)&(arr<3) ]
    print(mask)
    #掩码排序
    arr=np.array(['A','B','C','G'])
    mask=[1,2,3,0,2,1,3,1,0]
    print(arr[mask])


'''
   多维数组的组合和拆分
   垂直合并                 垂直拆分 2:表示份数
   np.vstack((a,b,c...))    np.vsplit((c,2))
   水平合并                 水平拆分
   no.hstach((a,b,c..))     np.hsplit((c,2))
   深度合并                 深度拆分
   np.dstack((a,b,c...))    np.dsplit((c,2))
'''









if __name__ =='__main__':
    arr_prop()
    get_arr()
    print("=========================================")
    fuhe()
    print("=========================================")
    change_shape()
    print("=========================================")
    arr_qie()
    print("=======ss==================================")
    mask_fun()