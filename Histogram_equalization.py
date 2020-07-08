import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt;
'''
t[i][j]=c[s[i][j]]*(max-min)+min;
'''
def getHx(src: object) -> object:#HX是直方图
    (b, g, r) = cv.split(src)
    h =src.shape[0]
    w =src.shape[1]
    hist_b = cv.calcHist([b], [0], None, [256], [0, 256])
    hist_g = cv.calcHist([g], [0], None, [256], [0, 256])
    hist_r = cv.calcHist([r], [0], None, [256], [0, 256])
    plt.subplot(321)#121=1行2列第1位置
    plt.imshow(src, 'gray')
    plt.xticks([])
    plt.yticks([])
    plt.title("Original")
    plt.subplot(322)#122=1行2列第2位置
    plt.hist(b.ravel(), 256, [0, 256])
    plt.title("b")
    plt.subplot(323)  # 122=1行2列第2位置
    plt.hist(g.ravel(), 256, [0, 256])
    plt.title("g")
    plt.subplot(324)  # 122=1行2列第2位置
    plt.hist(r.ravel(), 256, [0, 256])
    plt.title("r")
    plt.show()
    return  b, g, r,hist_b,hist_g,hist_r

def toEqual(src,b, g, r,hist_b,hist_g,hist_r):
    dst1 = src
    h = b.shape[0]
    w = b.shape[1]
    max = np.max(src)
    min = np.min(src)
    b = channelToEqual(b,hist_b)
    g = channelToEqual(g, hist_g)
    r = channelToEqual(r, hist_r)
    for i in range(h):
        for j in range(w):
            dst1[i][j][0]=b[i][j]
    for i in range(h):
        for j in range(w):
            dst1[i][j][1]=g[i][j]
    for i in range(h):
        for j in range(w):
            dst1[i][j][2]=r[i][j]
    return dst1

def channelToEqual(b,hist_b):
    max = np.max(b)
    min = np.min(b)
    h = b.shape[0]
    w = b.shape[1]
    N = h*w
    hist =hist_b
    hist = hist / N
    for k in range(1,hist.shape[0]):
        hist[k] = hist[k]+hist[k-1]
    hist = np.round(hist*255)
    '''for jkl in range(256):
        for uzi in range(256):
            if hist[uzi]== jkl:
                hist_b[jkl] += hist_b[uzi]'''
    for i in range(h):
        for j in range(w):
            b[i][j] = hist[b[i][j]]

    plt.xticks([])
    plt.yticks([])
    plt.hist(b.ravel(), 256, [0, 256])
    plt.show()
    return b



if __name__ == '__main__':
    src = cv.imread('C:\\Users\\86139\\Pictures\\Uplay\\timm.png')
    cv.imshow("input.jpg", src)
    b, g, r,hist_b,hist_g,hist_r = getHx(src)
    dst1 = toEqual( src,b, g, r,hist_b,hist_g,hist_r)
    cv.imshow("output.jpg", dst1)
    cv.waitKey(0)