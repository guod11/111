import cv2
import random
import numpy as np
from matplotlib import pyplot as plt
#读取灰度图，并且显示image的数组元素类型和尺寸。
img_grey = cv2.imread('C:/zihao.jpg', 0)
#Q:a = cv2.imread('C:/用户/温婉的火莲/Desktop/cv/zihao.jpg') 图片读取路径出现错误？
cv2.imshow('zihao', img_grey)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()
print(img_grey)
print(img_grey.dtype)
#type是数据类型比如list,ndarray,dictionary,tuple...dtype是数组元素的类型，比如uint8，或者float32，float64
print(img_grey.shape)
print(type(img_grey))

#彩色图，读取出来的shape有三通道
#图片的类型是numpy.ndarray。多维矩阵（多维数组）
img = cv2.imread('C:/zihao.jpg')
cv2.imshow('zihao', img)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()
print(img.shape)
print(type(img))

#image crop
img_crop = img[100:200, 0:300]
cv2.imshow('img_crop', img_crop)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()

#RGB三通道单独显示
#Q：三通道的图片都是黑白的，但是只是表征参数大小，实际颜色还是BGR
B, G, R = cv2.split(img)
cv2.imshow('B', B)
cv2.imshow('G', G)
cv2.imshow('R', R)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()

#change color
def random_light_color(img):
    B, G, R = cv2.split(img)
    b_rand = random.randint(-50, 50)
    if b_rand == 0:
        pass
    elif b_rand > 0:
        lim = 255 - b_rand
        B[B > lim] = 255
        #改变图片的数值，直接类似于matlab的矩阵操作。
        #Q：为什么要加astype而且，astype不应该加int，list这种数据类型吗？为什么是数组元素类型？
        B[B <= lim] = (B[B <= lim] + b_rand).astype(img.dtype)
    elif b_rand < 0:
        lim = 0 - b_rand
        B[B < b_rand] = 0
        B[B >= b_rand] = (B[B >= b_rand] + b_rand).astype(img.dtype)

    g_rand = random.randint(-50, 50)
    if g_rand == 0:
        pass
    elif g_rand > 0:
        lim = 255 - g_rand
        G[G > lim] = 255
        G[G <= lim] = (G[G <= lim] + g_rand).astype(img.dtype)
    elif g_rand < 0:
        lim = 0 - g_rand
        G[G < g_rand] = 0
        G[G >= g_rand] = (G[G >= g_rand] + g_rand).astype(img.dtype)

    r_rand = random.randint(-50, 50)
    if r_rand == 0:
        pass
    elif r_rand > 0:
        lim = 255 - r_rand
        R[R > lim] = 255
        R[R <= lim] = (R[R <= lim] + r_rand).astype(img.dtype)
    elif r_rand < 0:
        lim = 0 - r_rand
        R[R < r_rand] = 0
        R[R >= r_rand] = (R[R >= r_rand] + r_rand).astype(img.dtype)

    img_merge = cv2.merge((B, G, R))
    return img_merge

img_random_color = random_light_color(img)
cv2.imshow('img_random_color', img_random_color)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()

#gamma correction让图像变亮
def adjust_gamma(img, gamma = 1.0):
    invGamma = gamma / 1.0
    table = []
    for i in range(256):
        table.append(((i/255)**invGamma)*255)
    table = np.array(table).astype('uint8')
    return cv2.LUT(img, table)
img_brighter = adjust_gamma(img, 2)
cv2.imshow('img_brighter', img_brighter)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()

#scale + rotation + translation = similarity transform
M = cv2.getRotationMatrix2D((img.shape[0]/2, img.shape[1]/2), 30, 0.5)
#center, angle, scale放大缩小的倍数
img_rotate = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
cv2.imshow('img_rotate', img_rotate)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()
print(M)

#affine transform 把图片挤扁，不改变边的平行
rows, cols, ch = img.shape
pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.9, rows * 0.2], [cols * 0.1, rows * 0.9]])
M = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(img, M, (cols, rows))
cv2.imshow('affine_img', dst)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()
print(M)
#perspective transform 不平行同时挤扁
def random_warp(img, row, col):
    height, width, channels = img.shape
    # warp:
    random_margin = 60
    x1 = random.randint(-random_margin, random_margin)
    y1 = random.randint(-random_margin, random_margin)
    x2 = random.randint(width - random_margin - 1, width - 1)
    y2 = random.randint(-random_margin, random_margin)
    x3 = random.randint(width - random_margin - 1, width - 1)
    y3 = random.randint(height - random_margin - 1, height - 1)
    x4 = random.randint(-random_margin, random_margin)
    y4 = random.randint(height - random_margin - 1, height - 1)

    dx1 = random.randint(-random_margin, random_margin)
    dy1 = random.randint(-random_margin, random_margin)
    dx2 = random.randint(width - random_margin - 1, width - 1)
    dy2 = random.randint(-random_margin, random_margin)
    dx3 = random.randint(width - random_margin - 1, width - 1)
    dy3 = random.randint(height - random_margin - 1, height - 1)
    dx4 = random.randint(-random_margin, random_margin)
    dy4 = random.randint(height - random_margin - 1, height - 1)

    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
    M_warp = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, M_warp, (width, height))
    return M_warp, img_warp

M_warp, img_warp = random_warp(img, img.shape[0], img.shape[1])

cv2.imshow('img_warp', img_warp)
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()
