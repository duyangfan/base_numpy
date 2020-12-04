#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/12/2 17:29
#@Author: dudu
#@File  : mat_animation.py

import matplotlib.animation as ma
import matplotlib.pyplot as mp
import numpy as np

n=30
balls=np.zeros(n,dtype=[('position',float,2),('size',float,),('growth',float,),('color',float,4)])
#uniform:均匀分布的随机数
balls['position']=np.random.uniform(0,1,(n,2))
balls['size']=np.random.uniform(40,50,n)
balls['growth'] = np.random.uniform(10, 20, n)
balls['color'] = np.random.uniform(0, 1, (n,4))

mp.figure('Bubble', facecolor='lightgray')
mp.title('Bubble', fontsize=16)
#散点图
sc = mp.scatter(balls['position'][:, 0], balls['position'][:, 1], balls['size'], color=balls['color'])




def update(number):
    balls['size']+=balls['growth']
    index=number % n
    balls['size'][index]=np.random.uniform(40,50,1)
    balls['position'][index]=np.random.uniform(0,1,(1,2))

    #重绘属性
    sc.set_sizes(balls['size'])
    sc.set_offsets(balls['position'])



amin=ma.FuncAnimation(mp.gcf(),update,interval=300)

mp.show()


