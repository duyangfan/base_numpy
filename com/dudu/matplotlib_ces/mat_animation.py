#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/12/2 17:29
#@Author: dudu
#@File  : mat_animation.py

import matplotlib.animation as ma
import matplotlib.pyplot as mp



def exe():
    ma.FuncAnimation(mp.gca(),update,interval=10)
    mp.show()

def update(number):
    pass


if __name__=='__main__':
    exe()