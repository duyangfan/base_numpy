#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/12/3 10:44
#@Author: dudu
#@File  : numpy_fun.py

import numpy as np
import datetime as dt
import matplotlib.pyplot as mp
import matplotlib.dates as md
'''
    numpy  文件读取
    np.loadtxt(
        '', 文件路径
        delimiter=','  分隔符
        usecols=(1,3),  读取列元组 eg 1,3
        unpack=True   True:拆分成单个数组    False:不拆分数组
        dtype=''   返回每一列数组的类型
        converters     转化器函数字典
     )
        
        
    标准差
    np.std(arrs)
'''

def read_file():
    dates,bc,pc,nc,kl=np.loadtxt('../files/ces.csv',delimiter=',',usecols=(1,3,4,5,6),unpack=True,dtype='M8[D],f8,f8,f8,f8',converters={1:conv_date})
    #标准差
    print('标准差：',np.std(bc))

    e=bc.mean()
    d=(bc-e)**2
    v=np.mean(d)
    s=np.sqrt(v)
    print('标准差',s)


    mp.figure('ceshi',facecolor='lightgray')
    mp.title('ces')
    mp.tick_params(labelsize=10)
    mp.grid(linestyle=':')
    ax=mp.gca()
    ax.xaxis.set_major_locator(md.WeekdayLocator(byweekday=md.MO))
    #刻度显示格式化
    ax.xaxis.set_major_formatter(md.DateFormatter('%d%b%Y'))
    ax.xaxis.set_minor_locator(md.DayLocator())

    mp.plot(dates,bc,linestyle="--",color='dodgerblue',linewidth=1,alpha=0.6,label='AAPL BC')
    mp.plot(dates, pc, linestyle="-.", color='red', linewidth=1, alpha=0.6, label='AAPL PC')
    mp.plot(dates, kl, linestyle=":", color='green', linewidth=1, alpha=0.6, label='AAPL kl')

    mp.legend()
    mp.gcf().autofmt_xdate()
    mp.show()

def conv_date(val):
    val=str(val,encoding='utf-8')
    time=dt.datetime.strptime(val,'%d-%m-%Y').date()
    t=time.strftime('%Y-%m-%d')
    return t




'''
 统计每周均值
'''
def get_week_e():
    wdays,bc,pc,nc,kl=np.loadtxt('../files/ces.csv',delimiter=',',usecols=(1,3,4,5,6),unpack=True,dtype='f8,f8,f8,f8,f8',converters={1:dmy2wday})
    avg_price=np.zeros(5)
    for wday in range(avg_price.size):
        #掩码计算 wdays==wday   true:返回数据
        avg_price[wday]=np.mean(bc[wdays==wday])
    print(avg_price)



'''
    时间数据处理
'''
def dmy2wday(dmy):
    dmy=str(dmy,encoding='utf-8')
    date=dt.datetime.strptime(dmy,'%d-%m-%Y').date()
    wday=date.weekday()
    return wday


'''
   数组的轴向汇总：
   func:function
   axis:轴向 0：垂直  1：水平
   array:二维数组
   np.apply_along_axis(func,axis,array)
'''
def reslut():
    array=np.arange(1,21).reshape(4,5)
    res=np.apply_along_axis(get_res,1,array)
    print(res)

#针对每一个轴向的数据进行处理的函数
def get_res(arr):
   return arr.mean(),arr.max(),arr.min()


'''
  移动均线
'''
def draw_move_line():
    dates,bc=np.loadtxt('../files/ces.csv',delimiter=',',usecols=(1,3),unpack=True,dtype='M8[D],f8',converters={1:conv_date})
    mp.figure('ceshi', facecolor='lightgray')
    mp.title('ces')
    mp.tick_params(labelsize=10)
    mp.grid(linestyle=':')
    ax = mp.gca()
    ax.xaxis.set_major_locator(md.WeekdayLocator(byweekday=md.MO))
    # 刻度显示格式化
    ax.xaxis.set_major_formatter(md.DateFormatter('%d%b%Y'))
    ax.xaxis.set_minor_locator(md.DayLocator())

    mp.plot(dates, bc, linestyle="--", color='dodgerblue', linewidth=1, alpha=0.6, label='AAPL BC')
    #绘制5日均线
    sma5=np.zeros(bc.size-4)
    for i in range(sma5.size):
        sma5[i]=bc[i:i+5].mean()
    print(sma5.shape,"   ",dates.shape)
    #卷积运算
    core=np.ones(10)/10
    sma2=np.convolve(bc,core,mode='valid')
    #基于加权卷积实现5日均线
    #从y=e^x
    x=np.linspace(-1,0,5)
    kernel=np.exp(x)
    kernel/=kernel.sum()#让卷积核元素之和为1
    print(kernel)
    sma53=np.convolve(bc,kernel[::-1],mode='valid')


    #计算布林带的上下轨
    stds=np.zeros(bc.size-4)
    for i in range(stds.size):
        stds[i]=bc[i:i+5].std()

    upper=sma53+stds*2

    lower=sma53-stds*2



    #5日均线设计
    mp.plot(dates[4:],sma5,linestyle=":",color='orangered',linewidth=5,label='SMA-5',alpha=0.5)
    #10日均线设计
    mp.plot(dates[9:], sma2, linestyle=":", color='red', linewidth=2, label='SMA-10',)
    # 5日均线设计
    mp.plot(dates[4:], sma53, linestyle='-', color='green', linewidth=1, label='SMA-5', alpha=0.5)
    #上轨
    mp.plot(dates[4:],upper, linestyle="-", color='blue', linewidth=1, label='upper')
    #下轨
    mp.plot(dates[4:], lower, linestyle='-', color='blue', linewidth=1, label='lower')

    mp.fill_between(dates[4:],upper,lower,upper>lower,color='gray')

    mp.legend()
    mp.gcf().autofmt_xdate()
    mp.show()



'''
卷积（convolve):
    a=[1,2,3,4,5]
    b=[8,7,9]
    
    使用b作为卷积核数组对a数组执行卷积运算：
                    47  71  95              有效卷积结果(valid)  
                23  47  71  95   71         同纬卷积结果(same)   中心元素为准
        res:8   23  47  71  95   71 45      完全卷积结果(full)
    0   0   1   2   3   4   5   0   0
1.  9   7   8
2.      9   7   8
3.          9   7   8
4.              9   7   8
5.                  9   7   8
6.                      9   7   8
7.                          9   7   8


一维度：
c=numpy.convolve(a,b,卷积类型）

卷积核运算翻转：因为从左到右，生效，翻转后能正好是左边先进入计算


加权卷积：

加权平均值：
np.average() 
np.max() np.min() np.argmax（）最大元素 np.argmin()最小元素
中位数：np.median()
'''
def convo():
    a=np.array([1,2,3,4,5])
    b=np.array([8,7,9])
    res = np.convolve(a, b, mode='full')
    print(res)
    res=np.convolve(a,b,mode='same')
    print(res)
    res = np.convolve(a, b, mode='valid')
    print(res)




'''
线性预测：
A*X=B  ===>x=A*(A^-1)*X=B*(A^-1)
X=np.linalg.lstsq(A,B)[0]
'''
def get_next_data():
    bc = np.loadtxt('../files/ces.csv', delimiter=',', usecols=(3), unpack=True, dtype='f8',
                           converters={1: conv_date})
    N=4
    A=np.zeros((N,N))
    for j in range(N):
        A[j,]=bc[j:j+N]
    B=bc[N:N*2]
    X=np.linalg.lstsq(A,B,rcond=None)[0]
    print(X)
    pred=B.dot(X) #点乘
    print(pred,"  ",bc[7])


'''
 线性拟合
 各个模块采用插件方式：
  -->用户体验过程------>分析用户画像-------->智能加载模块---->得到不同奖励-->
 |<----------------------------------------------------------------------|
 奖励<---->交易
'''

if __name__ == '__main__':
    get_next_data()