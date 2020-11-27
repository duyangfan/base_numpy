from PIL import Image as Image
from PIL import ImageFilter as IFilter
from PIL import ImageEnhance as IEnhance
import numpy as np


path="D:\imageInfo"
imageName="\\apple.jpg"
img=Image.open(path+imageName)
print("图片格式："+img.format)
print(""+img.mode)
print(img.size)

'保存图片信息'
img.save(path+"\save.jpg")
'重置大小'
reSizeImage=img.resize((256,256))
reSizeImage.save(path+"\\reSize.jpg")
'旋转'
img_rote=img.rotate(30)
img_rote.save(path+"\img_rote.png")
'灰度处理   1，L，P，RGB，RGBA，CMYK，YCbCr,I，F '
vert_img=img.convert("L")

img_arr=np.array(vert_img)
print("shape: ",img_arr.shape)
print(img_arr)
vert_img.save(path+"\\vert_img.jpg")


'''
输入图像为 (H,W)
卷积核为 （N,N）
输出图像为：A , B
对图像卷积：
采用valid模式：不添加padding 那么输出的结果图像为：A=H-N+1,B=W-N+1
采用same模式：需要添加padding，一般添加0作为填充，上下左右都需要填充，根据A=H ,B=W  
计算出添加的padding层数为：H+2p-N+1=H =>p=(N-1)/2 
'''

def num_conv(path,name,padding,core_arr):
    img=Image.open(path+name)
    r,g,b=img.split()
    img_arr=np.array(r)
    o_h,o_w=img_arr.shape
    img_h,img_w=0,0
    if padding=='valid':
        img_h=o_h-len(core_arr)
        img_w=o_w-len(core_arr[0])
        print("核卷积的shape为：",(img_h,img_w))

    else:
        print("else")
print("finish")



'把图像分解为rgb  然后 合并的'

def deal_img(path,name):
    img=Image.open(path+name)
    r,g,b=img.split()
    r.save(path+"\\r.jpg")
    g.save(path + "\\g.jpg")
    b.save(path + "\\b.jpg")
    res_img=Image.merge('RGB',[r,g,b])
    res_img.save(path+"\\result_img.jpg")



'''


'''






if __name__ =='__main__':
    # core_arr=[[1,0,1],
    #           [1,0,1],
    #           [1,0,1]]
    path = "D:\imageInfo"
    imageName = "\\apple.jpg"
    # num_conv(path,imageName,'valid',core_arr)
    deal_img(path,imageName)










