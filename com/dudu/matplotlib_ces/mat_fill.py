#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/12/2 9:09
#@Author: dudu
#@File  : mat_fill.py  填充

import matplotlib.pyplot as mp
import numpy as np
#3d坐标系
from mpl_toolkits.mplot3d import axes3d

'''
mp.fill(
    x,    #定义域
    fx,   #
    gx,   #
    fx<gx,#条件
    color,
    alpha,
)
'''
def fill():
    x=np.linspace(-np.pi,np.pi,1000)
    sinx=np.sin(x)
    cosx=np.cos(2*x)

    mp.figure('fill',facecolor='lightgray')
    mp.title('fill',fontsize=18)
    mp.grid(linestyle='-.')

    mp.plot(x,sinx,color='lightblue',label='sin(x)')
    mp.plot(x,cosx,color='orangered',label='cos(x)')

    mp.xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
              [r'$-\pi$', r'$-\frac{\pi}{2}$', '0', r'$\frac{\pi}{2}$', r'$\pi$'])
    mp.yticks([-1, 1])

    # 设置坐标轴
    ax = mp.gca()
    # 获取某个坐标轴 ax.spines['坐标轴名'] left top right bottom
    # 设置坐标轴值 set_position('data',val)
    ax.spines['left'].set_position(('data', 0))
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position(('data', 0))

    #绘制填充
    mp.fill_between(x,sinx,cosx,sinx < cosx,color='dodgerblue',alpha=0.3)
    mp.fill_between(x, sinx, cosx, sinx > cosx, color='blue', alpha=0.3)
    #添加文本注释

    mp.legend()
    mp.show()





'''
    柱状图
    mp.bar(
        x,      #水平坐标数组
        y,      #柱状图高度数组
        width,  #柱子的宽度
        color,
        label,  
        alpha,
    )
'''
def draw_bar():
    apples=np.array([75,96,84,26,85,74,12,58,96,32,58,64])
    oranges=np.array([75,89,62,48,59,61,23,65,87,45,69,56])
    mp.figure('Bar_Chart',facecolor='lightgray')
    mp.title('Bar_Chart')
    mp.xlabel('Month',fontsize=16)
    mp.ylabel('Volume',fontsize=16)
    mp.tick_params(labelsize=10)
    mp.grid(linestyle=':',axis='y')
    x=np.arange(12)+1


    #draw_bar:绘图
    mp.bar(x-0.2,apples,0.4,color='dodgerblue',label='Apple')
    mp.bar(x+0.2, oranges, 0.4, color='orange', label='Apple')

    mp.xticks(x,['Jan','Feb','Mar','Apr','Mar','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    mp.legend()
    mp.show()


'''
    饼图
    mp.pie(
        values,             #值列表
        spaces,             #扇形之间的间距列表
        label,              #标签列表
        colors,             #颜色列表
        '%d%%',             #标签所占比例格式
        shadow=true,        #是否显示阴影
        startangle=90,      #逆时针绘制饼状图时的起始的角度
        radius=1            #半径
    )
'''
def draw_pie():
    values=[20,17,51,29,11]
    labels=['python','javaScript','c++','java','php']
    spaces=[0.05,0.01,0.01,0.01,0.01]
    colors=['dodgerblue','orangered','limegreen','violet','gold']

    mp.figure('pie', facecolor='lightgray')
    mp.title('pie')
    mp.pie(values,spaces,labels,colors,'%.2f%%',shadow=False,startangle=0,radius=1)

    #等轴比例
    mp.axis('equal')
    mp.legend()
    mp.show()

'''
等高线图
mp.contour(
    x,
    y,
    z,
    8,
    colors='black',
    linewidths=0.5
'''
def draw_contour():
    n=500
    x,y=np.meshgrid(np.linspace(-3,3,n),np.linspace(-3,3,n))
    #计算每个坐标点的高度
    z=(1-x/2+x**5+y**3)* np.exp(-x**2-y**2)*1000


    mp.figure('contour',facecolor='lightgray')
    mp.title('contour',fontsize=18)
    #绘制等高线
    contr=mp.contour(x,y,z,8,colors='black',linewidths=0.5)
    #绘制等高线文本
    mp.clabel(contr,inline_spacing=1,fmt='%.1f',fontsize=10)
    #填充等高线
    mp.contourf(x,y,z,8,cmap='jet')
    mp.show()

'''
热成像图像
    origin:坐标轴方向————>upper---缺省值，原点在左上角lower----原点在左下角
    z: 二维数组
    mp.imshow(z,cmap='jet',origin='low')
'''
def draw_reimg():
    mp.figure('reimg')
    n = 500
    x, y = np.meshgrid(np.linspace(-3, 3, n), np.linspace(-3, 3, n))
    # 计算每个坐标点的高度
    z = (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2) * 1000

    mp.imshow(z,cmap='jet',origin='lower')
    mp.colorbar()

    mp.show()


'''
   3D 散点图
   ax3d.scatter(
   x,      x轴坐标arr
   y,      y轴arr
   z,      z轴arr
   marker, 点型
   s=10    大小
   zorder, 图层序号
   color,  颜色
   edgecolor,  边缘颜色
   facecolor,  填充色
   c=v,         
   cmap        颜色映射
   )
   
'''
def draw_3d_scatter():
    n=100
    x=np.random.normal(0,1,n)
    y = np.random.normal(0, 1, n)
    z = np.random.normal(0, 1, n)

    #绘制3维散点图
    ax3d=mp.gca(projection='3d')
    ax3d.set_xlabel('X',fontsize=16)
    ax3d.set_ylabel('Y', fontsize=16)
    ax3d.set_zlabel('Z', fontsize=16)
    d=x**2+y**2+z**2
    ax3d.scatter(x,y,z,marker='o',s=70,c=d,cmap='jet',alpha=0.5)
    #紧凑布局
    mp.tight_layout()

    mp.show()


'''
  3d 曲面图
  ax3d.plot_surface(
    x,        x（2维arr)
    y,
    z,
    rstride,  行跨距
    cstride   列跨距
    cmap      颜色映射
  )
'''

def draw_3d_surface():

    n = 500
    x, y = np.meshgrid(np.linspace(-3, 3, n), np.linspace(-3, 3, n))
    # 计算每个坐标点的高度
    z = (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2) * 1000

    mp.figure('qu_mian_tu')

    ax3d=mp.gca(projection='3d')
    ax3d.set_xlabel('x',fontsize=14)
    ax3d.set_ylabel('y', fontsize=14)
    ax3d.set_zlabel('z', fontsize=14)
    ax3d.plot_surface(x,y,z,cmap='jet',rstride=10,cstride=10)

    mp.tight_layout()

    mp.show()



'''
    3d线框图
    plot_wireframe(x,yx,z,rstrid,cstride,linewidth,color)

'''
def draw_plot_frame():
    n = 500
    x, y = np.meshgrid(np.linspace(-3, 3, n), np.linspace(-3, 3, n))
    # 计算每个坐标点的高度
    z = (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2) * 1000

    mp.figure('qu_mian_tu')

    ax3d = mp.gca(projection='3d')
    ax3d.set_xlabel('x', fontsize=14)
    ax3d.set_ylabel('y', fontsize=14)
    ax3d.set_zlabel('z', fontsize=14)
    ax3d.plot_wireframe(x, y, z, rstride=10, cstride=10,linewidth=1,color='dodgerblue')

    mp.tight_layout()

    mp.show()

'''
 极坐标系
  极径ρ  极角：θ
'''
def draw_polar():
    mp.figure('polar',facecolor='lightgray')
    mp.gca(projection='polar')
    mp.title('porlar',fontsize=20)
    mp.xlabel(r'$\theta$',fontsize=14)
    mp.ylabel(r'$\rho$', fontsize=14)
    mp.tick_params(labelsize=10)
    mp.grid(linestyle=':')

    t=np.linspace(0,4*np.pi,1000)
    r=0.8*t
    mp.plot(t,r)
    mp.show()



if __name__ =='__main__':

    draw_polar()

