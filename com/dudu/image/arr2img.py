
from PIL import Image
import numpy as np


def MatrixToImage():
    r = np.zeros((768,1366))
    g = np.zeros((768, 1366))
    b = np.zeros((768, 1366))
    i=0
    j=0
    count=0
    '''
       17x^2-16xy+17y^2=225
       17（x-16）-16(x-16)(y+16)+17(y+16)^2=255
       
    '''
    for i in range(0,768,1):
        for j in range(0,1366,1):
            if (i-384)*(i-384)+(j-384)*(j-384)==400:
                r[i][j]=220
                g[i][j] =20
                b[i][j] =60
                count = count + 1
            else:
                r[i][j] = 2 * i + 2 * j
                g[i][j] = 2 * j + i
                b[i][j] = -i - j

    print("计算次数：",count)
    r_matrix=np.mat(r)
    g_matrix = np.mat(g)
    b_matrix = np.mat(b)
    r = Image.fromarray(r_matrix.astype(np.uint8))
    g = Image.fromarray(g_matrix.astype(np.uint8))
    b = Image.fromarray(b_matrix.astype(np.uint8))
    b.save(path+"matrix2img_b.jpg")
    new_im=Image.merge('RGB',[r,g,b])
    return new_im


path="D:\\imageInfo\\"
filename = 'apple.jpg'
#data = ImageToMatrix(path+filename)
new_im = MatrixToImage()
new_im.save(path+"matrix2img.jpg")
print("r=",768/2)