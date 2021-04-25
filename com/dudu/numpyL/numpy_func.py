#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/12/3 10:44
#@Author: dudu
#@File  : numpy_fun.py

import numpy as np
import numpy.fft as nf
import datetime as dt
import cv2 as cv
import matplotlib.pyplot as mp
import matplotlib.dates as md
import scipy.io.wavfile as wf
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

'''
   协方差    相关矩阵      相关系数
   协方差：  可以评估两组统计数据的相似程度，值为正，正相关，值为负，负相关，绝对值越大相关性越强
   算法：
   1.计算平均值
   2.计算离差：元素-平均值
   3.计算协方差，对应元素相乘之后就平均值。
   
   
   相关系数：协方差
   1.标准差：①离差^2  ②之后开方 np.std(arr):计算标准差
   1.协方差除以两组统计样本的标准差的乘积是一个[-1,1]的数。
   
   
   相关矩阵：np.corrcoef(A,B)
   [[a与a的相关系数     a与b的相关系数]
    [b与a的相关系数     b与b的相关系数]]
    
    np.cov(A,B)相关矩阵的分子矩阵
'''

'''
  多项式拟合
  多项式一般表示：y=p0*x^n+p1*x^(n-1)...+pn   表示为：[p0,p1....,pn]
  根据一组样本，给出最高次幂，求出拟合系数：np.ployfit(x,y,最高次幂)-->P_arr
  
  多项式运算：
  np.polyval(p_arr,X)->：根据多项式函数与自变量求出拟合值，由此可得【拟合】曲线坐标样本【数据】。
  
  多项式求导,得到导数函数：np.ployder(p_arr)
  
  已知多项式系数p_arr,求多项式函数的根：xs=np.roots(p_arr)
  
  两个多项式函数的差函数的系数：np.polysub(p1_arr,p2_arr)
  
  数据平滑：包涵降噪，拟合。降噪去除额外的影响因素，拟合的目的是数学模型化，可以通过更多的数学方法识别曲线的特征
  
  np.diff(arr):前一个数 —— 后一个数  返回arr
  np.hanning(8)
'''
def duo_fun():
    x=np.linspace(-20,20,1000)
    y=4*x**3+3*x**2-1000*x+1

    #提取多项式系数
    p=np.array([4,3,-1000,1])
    Q=np.polyder(p)
    R=np.roots(p)
    rY=np.polyval(p,R)



    mp.plot(x,y)
    mp.scatter(R,rY,s=60,marker='D')
    mp.show()



'''
矩阵的逆：E.I

'''

def mat_fun():
   A=np.mat('1 2 6;3 5 7;4 8 9')
   print(A)

   print(A.I)

   print(A*A.I)


'''
  特征值   特征向量   奇异值
  对于n阶  -方阵-  A, 如果存在数a和非零n维列向量x,使得Ax=ax,则称a是矩阵A的一个特诊值，x是矩阵A属于特诊值a的特诊向量
  A  X = a X  a:是A的特诊值    X:是A的特征向量
  a,X=np.linalg.eig(A).
  
  已知特征值和特征向量，求方阵
  S=np.mat(X)*np.mat(np.diag(a)*np.mat(X.I)
  
  
  奇异值分析：--可以针对非方阵
  有一个矩阵M,可以分解为3个矩阵U,S,V,使得U*S*V等于M.U与V都是正交矩阵（乘以自身的转置矩阵结果为单位矩阵），那么S矩阵主对角线上的元素称为矩阵M的奇异值，其它元素为0
  np.linalg.svd(
  
'''
def mat_fun():

    A=np.mat('1 6 3 7;3 8 4 6;1 4 9 5;6 8 3 5')
    print(A)
    a,X=np.linalg.eig(A)
    print('a',a)
    print('--------------')
    print('X',X)
    #逆向推导原矩阵
    A_2=X*np.diag(a)*X.I
    print(A_2)



###提取图片的特征值
def img_arr():
    #读取图片
    img=cv.imread('../files/apple.jpg',flags=0)
    print(img.shape)
    img=np.mat(img)
    eigvals,eigves=np.linalg.eig(img)

    #抹掉一部分特征值
    eigvals[50:]=0
    img2=eigves*np.diag(eigvals)*eigves.I
    print(img2.dtype)
    img2=img2.real
    #奇异值分解
    U,sv,V=np.linalg.svd(img)
    sv[50:]=0
    img3=U*np.diag(sv)*V



    #绘图
    mp.figure('apple')
    mp.subplot(221)
    mp.imshow(img,cmap='gray')
    mp.xticks([])
    mp.yticks([])
    mp.subplot(222)
    mp.imshow(img2,cmap='gray')
    mp.xticks([])
    mp.yticks([])
    mp.subplot(223)

    mp.imshow(img3, cmap='gray')
    mp.xticks([])
    mp.yticks([])
    mp.tight_layout()
    mp.show()


#奇异值
def svd_fun():
    M=np.mat('4 11 14;8 7 -2')
    print(M)
    U,SV,V=np.linalg.svd(M,full_matrices=False)
    print(U*U.I)
    print('--------')
    print(V*V.I)
    print('-----------')
    print(SV)
    S=np.diag(SV)
    print(S)
    print(U*S*V)


'''
 傅里叶定理：任何一条周期曲线，无论多么跳跃或不规则，都能表示成一组光滑正选曲线叠加之和
 傅里叶变换：对于一条不规则的曲线进行拆解，从而得到一组光滑正选曲线函数过程。
 y=Asin(wX,φ)
 时域图---->频域图 
 import numpy.fft as nf
 
 通过采样数与采样周期求得傅里叶变换分解所得曲线的频率序列
 freqs=nf.fftfreq(采样数，采样周期)->频率序列
 
 通过原函数的序列经过快速傅里叶变换得到一个复数数组，复数的摸代表的是振幅，复数的辐角代表初相位
 nf.fft(原函数值序列)->目标函数值序列(复数)
 
 通过一个复数数组（复数的模代表的是振幅，复数的辐角代表初相位)经过逆向傅里叶变换得到合成的函数值数组
 np.fft.ifft(目标函数值序列(复数))->原函数值序列
 
 
 
 
 
'''


def fourier_fun():
    x = np.linspace(-np.pi, np.pi, 1000)
    y = np.zeros(1000)
    n = 1000
    for i in range(1, n + 1):
        y += 4 / ((2 * i - 1) * np.pi) * np.sin((2 * i - 1) * x)


    conplex_arr=_arr=nf.fft(y)
    #复数的模
    i=np.abs(conplex_arr)
    print(i)
    y_=nf.ifft(conplex_arr)

    mp.plot(x, y, label='n=1000')
    mp.subplot(121)
    mp.plot(x, y_, label='ifft',linewidth=7,color='orangered',alpha=0.3)
    ax = mp.gca()
    ax.spines['bottom'].set_position(('data', 0))
    mp.xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
              [r'$-\pi$', r'$-\frac{\pi}{2}$', '0', r'$\frac{\pi}{2}$', r'$\pi$'])
    mp.yticks([-1, 1])


    #绘制频域图像
    freqs=nf.fftfreq(y.size,x[1]-x[0])
    pows=np.abs(conplex_arr)
    mp.subplot(122)
    mp.plot(freqs,pows,color='orangered')



    mp.legend()
    mp.show()

'''
  基于傅里叶变换的频域滤波
'''
def music_fun():
    #采样率      每个采样位移值
    sample_rate,sigs=wf.read('../files/music.wav')
    print(sample_rate)
    #绘制音频时域图：时间、位移图像
    times=np.arange(len(sigs))/sample_rate
    sigs=sigs/(2**15)
    print(times)
    mp.figure('fitter',facecolor='lightgray')
    mp.subplot(221)
    mp.title('Time Domain',fontsize=16)
    mp.ylabel('signal',fontsize=12)
    mp.grid(linestyle=':')
    mp.plot(times[:1000],sigs[:1000],color='dodgerblue')


    #获取频域信息
    freqs=nf.fftfreq(sigs.size,1/sample_rate)
    print('sigs ',len(sigs))
    complex_arr=nf.fft(sigs)
    pows=np.abs(complex_arr)

    mp.subplot(222)
    mp.title('Frequence Domain', fontsize=16)
    mp.ylabel('pow', fontsize=12)
    mp.grid(linestyle=':')
    mp.semilogy(freqs[:1000], pows[0:1000], color='dodgerblue',label='f')




    pows_med=np.median(pows)
    print('pows',pows_med)
    #拿到所有噪声的索引
    noised_indices=np.where(freqs<=pows_med)[0]


    print('noised_indices', noised_indices[:16148])
    freq_x=freqs[noised_indices[:16148]]
    pows=np.abs(freq_x)

    mp.subplot(223)
    mp.title('Domain', fontsize=16)
    mp.ylabel('pow', fontsize=12)
    mp.grid(linestyle=':')
    mp.semilogy(freqs[:1000], pows[0:1000], color='dodgerblue', label='f')


    mp.legend()

    mp.show()



'''
 随机数模块(random)
 二项分布：np.random.binomial(n,p,size) 每次试验都相互独立,n:尝试中的成功次数   p:成功的概率,size:重复次数
 正态分布：np.random.normal(loc=期望,scale=标准差,size)
 超几何分布：np.random.hypergeometrc(ngood,nbad,nsample,size)


'''

def random_fun():
    arr=np.random.binomial(10,0.7,10)
    print(arr)



if __name__ == '__main__':
    random_fun()