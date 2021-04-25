#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/12/10 15:55
#@Author: dudu
#@File  : ML.py
import numpy as np
import matplotlib.pyplot as mp
import sklearn.preprocessing as sp
'''
ML:
监督学习：   已知输出评估模型的性能
无监督学习： 没有已知输出的情况下，仅仅根据输入信息的相关性，进行类别划分
半监督学习： 通过无监督学习划分类别，再根据人工标记通过有监督学习预测输出。 先分类->人工标记（打标签）
强化学习：   通过对不同决策结果的奖励和惩罚，    
            实际去学习系统在经过足够长时间的训练，
            越来越倾向于给出接近期望结果的输出

批量学习：先训练->现实预测   判断是否满足  不满足重新训练

'''


'''
  数据预处理：
  均值移除(标准化):输入范围过大
    由于一个样本的不同特征值差异较大，不利于使用机器学习算法进行样本处理。
    均值移除可以让样本矩阵中的每一列的平均值为0，标准差为1
    均值为0:1.先计算样本的均值，2.计算每一个样本与均值的差作为结果，然后就可以得到样本的均值为0
    标准差为1：
    sp.scale(array): scale函数用于对函数进行预处理，实现均值移除
    array为原数组，返回A为均值移除后的结果
'''
def fun():
    arr=np.array([[17.,100.,4000],
                 [20.,80.,5000],
                 [23.,75.,5500]]);
    #均值移除 E=1 标准差=1
    r=sp.scale(arr)
    mms=sp.MinMaxScaler(feature_range=(0,1))
    res=mms.fit_transform(arr)
    print('结果：',res)
    s=sp.normalize(arr,norm='l1')
    print('归一化',s)
    print(r)
    print(r.mean(axis=0))
    print(r.std(axis=0))

'''
 范围缩放：
  将样本矩阵中的每一列的最小值和最大直是定为相同的区间，同一各列特征值的范围、一般情况下会把特征值缩放至[0,1]区间
  [17,20,23]->[0,20-17,23-17]->[0/6,3/6,6/6]=>[0,1/2,1] 缩放至[0,1]
  #创建MinMax缩放器
  mms=sp.MinMaxScaler(featrue_range)=(0,1))
  #调用mms对象的方法执行缩放操作，返回缩放过后的结果
  result=mms.fit_transform(原始样本矩阵)
'''

'''
 归一化：特征值具体并不重要，每个样本特征值的占比更加重要，每一样本的特征值绝对值之和为1
  s=sp.normalize(arr,norm='l1')
 二值化：给定阙值，用0和1表示特征值不高于或高于阙值。
 定义阙值，二值化器
 bin=sp.Binarizer(threshold=阙值)
 调用transform 方法对原始样本矩阵进行二值化预处理操作
 res=bin.transform(arr)
'''
'''
  独热编码(One-Hot)
  ：为样本特征的每个值建立一个由一个1和若干个0组成的序列，用该序列对所有的特征值进行编码，适用于离散值
   两个数   三个数    四个数
    1        3        2       
    7        5        4
    1        3        6
    7        3        9
   为每一个数字进行独热编码：
   1-10      3-100     2-1000
   7-01      5-010     4-0100
             8-001     6-0010
                       9-0001 
   编码完毕后得到最终经过独热编码后的样本矩阵：
   101001000
   010100100
   101000010
   011000001
   稀疏矩阵
   #创建一个独热编码器
   #sparse:是否使用紧缩格式(稀疏矩阵)
   #dtype:数据类型
   ohe=sp.OneHotEncoder(sparse,dtype)
   #对原始样本矩阵进行处理，返回独热编码后的样本矩阵。
   result=ohe.fit_transform(原始样本矩阵)
'''
def ohe_fun():
    arr=np.array([
        [1,3,2],
        [7,5,4],
        [1,8,6],
        [7,3,9]])
    print(arr)
    ohe=sp.OneHotEncoder(sparse=False)
    res=ohe.fit_transform(arr)
    print(res,type(res))
    ohe = sp.OneHotEncoder(sparse=True)
    res = ohe.fit_transform(arr)
    print(res,type(res))

'''
  标签编码：为其制定一个数字模型。
  #获取标签编码器
  lbe=sp.LabelEncoder()
  #调用标签编码器的fit_transform方法训练并且为原始样本矩阵进行标签编码
  result=lbe.fit_transform(原始特征列)
  #根据标签编码的结果矩阵反查字典，得到原始数据矩阵
  sampleslbe.inverse_transform(result)
  
  #根据数字模型---->转化为原始标签
  r=lbe.inverser_transform([])
'''

'''
 线性回归：
  输入    输出   
  0.5     5.0
  0.6     5.5
  0.8     6.0
  1.1     6.8
  1.4     7.0
  ...
  y=f(x)
  预测函数：y=kx+b
  输入：x
  输出：y
  模型参数：k,b
  所谓的模型训练就是根据已知的x和y,找到最佳的模型参数k和b,尽可能精确地描述出输入和输出的关系。
  
  5.0=0.5k+b
  5.5=0.6k+b
  单样本误差：
  根据预测函数求出输入为x时的预测值：y'=k*x+b,单体样本误差为1/2(y'-y)^2
  总体样本误差：
  把所有单体误差样本误差相加即是总体误差：1/2∑（y'-y)^2
  损失函数：
  loss=1/2Σ(b+kx-y)^2
  所以损失函数就是总样本误差关于模型参数的函数，该函数属于三维数学模型，即需要找到一组k,b使得loss取极小值
  梯度下降： 
'''
def get_loss_fun():

    n = 1000
    k, b = np.meshgrid(np.linspace(-100, 100, n), np.linspace(-100, 100, n))
    #计算损失函数
    xs=np.array([0.5,0.6,0.8,1.1,1.4])
    ys=np.array([5.0,5.5,6.0,6.8,7.0])

    loss=np.zeros_like(k)
    for x,y in zip(xs,ys):
        loss+=(b+k*x-y)**2/2

    mp.figure('qu_mian_tu')

    ax3d=mp.gca(projection='3d')
    ax3d.set_xlabel('k',fontsize=14)
    ax3d.set_ylabel('b', fontsize=14)
    ax3d.set_zlabel('loss', fontsize=14)
    ax3d.plot_surface(k,b,loss,cmap='jet',rstride=10,cstride=10)

    mp.tight_layout()

    mp.show()








if __name__ == '__main__':
    get_loss_fun()