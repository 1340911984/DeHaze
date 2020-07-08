import cv2 as cv
import numpy as np
import sys
def minImg(src,r):
    return cv.erode(src, np.ones((2 * r + 1, 2 * r + 1)))

def DarkChannel(src,dst):#dst = min(scr,7)
    dst = np.min(dst,2)
    cv.imshow('dark,jpg',dst)
    return dst

def getA(src):
    A = 0
    (b, g, r) = cv.split(src)
    for k in range(b.shape[0]-1):
        for i in range(b.shape[1]-1):
            w0 = max(b[k][i] ,r[k][i],g[k][i])
            w1 = min(b[k][i] ,r[k][i],g[k][i])
            w = w0/2 + w1/2
            if A<w:
                A = w
    if A>250:
        A = 250
    return A

def getTx(dst,A):
    tx = 1-0.95*(dst/A)
    for k in range(tx.shape[0]-1):
        for i in range(tx.shape[1]-1):
            if tx[k][i]<0.1:
                tx[k][i]=0.1
    return tx

def deHaze(src):
    dst = minImg(src, 7)
    dst1 = DarkChannel(src, dst)
    A = getA(src)
    tx = getTx(dst1,A)
    for k in range(dst1.shape[0]-1):
        for i in range(dst1.shape[1]-1):
            for j in range(3):
                dst[k][i][j] = (src[k][i][j]-A)/tx[k][i] + A
    output = dst
    return output

if __name__ == '__main__':
    src = cv.imread('C:\\Users\\86139\\Pictures\\Uplay\\timm.png')
    cv.imshow('input.jpg',src)
    output = deHaze(src)
    cv.imshow('output.jpg',output)
    cv.waitKey(0)