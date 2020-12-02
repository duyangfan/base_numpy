#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/11/30 10:13
#@Author: dudu
#@File  : kedu_dingwei.py

import matplotlib.pyplot as mp
import numpy as np

def shu_zhou():
    mp.figure('ding_wei',facecolor='lightgray')
    #获取当前坐标轴
    ax=mp.gca()
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position(('data',0.5))
    mp.xlim(1,10)
    mp.yticks([])

    ma_loc=mp.MultipleLocator(1)
    # 设置主刻度定位器
    ax.xaxis.set_major_locator(ma_loc)

    mi_loc=mp.MultipleLocator(0.1)
    # 设置次刻度定位器
    ax.xaxis.set_minor_locator(mi_loc)
    mp.show()



'''
    ax=mp.gca()
    ax.grid(which='' #k刻度类型：’major/minor' 主,次刻度
    axis='' x/y/both
'''
#刻度网格线
def grid_line():
    y=[1,10,100,1000,100,10,1]
    mp.figure('grid line',facecolor='lightgray')
    ax=mp.gca();
    x_ma_loc=mp.MultipleLocator(1)
    ax.xaxis.set_major_locator(x_ma_loc)
    x_mi_loc=mp.MultipleLocator(0.1)
    ax.xaxis.set_minor_locator(x_mi_loc)

    y_ma_loc = mp.MultipleLocator(250)
    ax.yaxis.set_major_locator(y_ma_loc)
    y_mi_loc = mp.MultipleLocator(50)
    ax.yaxis.set_minor_locator(y_mi_loc)

    #绘制网格刻度线
    ax.grid(which='major',axis='both',linewidth=0.25,color='lightgray',linestyle='-')
    #半对数坐标显示：mp.semilogy(y,'o-',color='dodgerblue')
    mp.plot(y,'o-',color='dodgerblue')
    mp.show()





#散点图
def san_dian():
    n=300
    x=get_random(173,5,n)
    y=get_random(65,15,n)
    mp.figure('persons', facecolor='lightgray')
    mp.title('person',fontsize=16)
    mp.xlabel('Height',fontsize=14)
    mp.ylabel('Weight',fontsize=14)
    mp.grid(linestyle=':')
    d=(x-173)**2+(y-65)**2
    mp.scatter(x,y,marker='o',s=12,label='Person',c=d,cmap='jet_r')
    mp.legend()
    mp.show()

def get_random(e,q,n):
    #期望
    #标准差
    #生成数量
    return np.random.normal(e,q,n)



if __name__=='__main__':
    san_dian()