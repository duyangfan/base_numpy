#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/11/30 9:06
#@Author: dudu
#@File  : matplotlib_base.py

import numpy as np
import matplotlib.pyplot as mp
import  matplotlib.gridspec as mg





#窗口参数
def create_wind():
    mp.figure('Figure A', facecolor='lightgray')
    mp.title('test',fontsize=12)
    mp.ylabel('Y',fontsize=14)
    mp.xlabel('X', fontsize=14)
    mp.tick_params(labelsize=10)
    mp.grid(linestyle=':')
    mp.tight_layout()
    mp.show()


#子图
def create_child_win():
    mp.figure('child_win',facecolor='lightgray')
    #row :行数
    #cols:列数
    #num:编号

    mp.subplot(331)
    mp.plot()
    mp.show()

#矩阵式子图布局测试
def ces_win():
    mp.figure('mat_child_win',facecolor='lightgray')
    for i in range(1,10):
        #行号  列号   编号
        mp.subplot(3,3,i)
        mp.text(0.5,0.5,i,size=36,ha='center',va='center')
        mp.xticks([])
        mp.yticks([])
        mp.tight_layout()
    mp.show()


#网格式
def wg_win():
    mp.figure('gide layout',facecolor='lightgray')
    #拆分 GridSpace 方法拆分网格式布局
    gs=mg.GridSpec(3,3)
    #合并单元格
    mp.subplot(gs[0,:2])
    mp.text(0.5, 0.5, 1, size=36, ha='center', va='center')
    mp.subplot(gs[:2, 2])
    mp.text(0.5,0.5,2,size=36,ha='center',va='center')
    mp.subplot(gs[1:3, 0])
    mp.text(0.5, 0.5, 3, size=36, ha='center', va='center')
    mp.subplot(gs[1,1])
    mp.text(0.5, 0.5, 4, size=36, ha='center', va='center')
    mp.subplot(gs[2,1:3])
    mp.text(0.5,0.5,5,size=36,ha='center',va='center')
    mp.xticks([])
    mp.yticks([])
    mp.tight_layout()
    mp.show()



#自由布局
def free_win():
    mp.figure('Flow layout',facecolor='lightgray')
    mp.axes([0.03,0.12,0.94,0.55])
    mp.text(0.5,0.5,'1',ha='center',va='center',size=36)
    mp.show()



if __name__ =='__main__':
    free_win()
    print("finished")