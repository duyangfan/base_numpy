#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/11/30 15:30
#@Author: dudu
#@File  : mat_base.py

import matplotlib.pyplot as mp
import numpy as np

def base():
    xs=np.arange(6)
    ys=np.array([15,26,14,23,75,96])
    #传入点左边数组
    mp.plot(xs,ys)
    #垂直线(vals,min,max)
    mp.vlines([1,5],2,100)
    mp.hlines(20,2,6)

    mp.show()

#正弦曲线
def sin_line():
    x=np.linspace(-np.pi,np.pi,1000)#获取-π到π 1000个数
    sinx=np.sin(x)#矢量方法，返回每一个x对应的y
    cosx=np.cos(x)


    #绘制点plot(x,y,color='')
    mp.plot(x,sinx,linestyle='-.',label=r'$y=sin(x)$')
    mp.plot(x, cosx,label=r'$y=cos(x)$')

    #设置坐标做的范围xlim(min,max)
    # mp.xlim(0, np.pi)
    # mp.ylim(0,1)

    #设置坐标刻度  LaTex排版语法字符
    mp.xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi],[r'$-\pi$',r'$-\frac{\pi}{2}$','0',r'$\frac{\pi}{2}$',r'$\pi$'])
    mp.yticks([-1,1])

    #设置坐标轴
    ax=mp.gca()
    #获取某个坐标轴 ax.spines['坐标轴名'] left top right bottom
    #设置坐标轴值 set_position('data',val)
    ax.spines['left'].set_position(('data',0))
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position(('data',0))




    #绘制特殊点
    xarray=[1/4*np.pi]
    yarray=[np.sin(xarray[0])]
    mp.scatter(xarray,yarray,marker='x',s=50,color='green',label='Point',zorder=3)

    #备注
    mp.annotate(r'$\frac{\pi}{2}$',xycoords='data',xy=(xarray[0],yarray[0]),textcoords='offset points',xytext=(-np.pi,np.sin(-np.pi)),fontsize=14,arrowprops=dict())

    # 设置图例 :需要给plot定义label属性
    mp.legend()
    mp.show()
    print('finished')

'''
    plot函数参数：
    xarray,yarray,linestyle,linewidth,color,alpha
'''


if __name__=='__main__':
    sin_line()

