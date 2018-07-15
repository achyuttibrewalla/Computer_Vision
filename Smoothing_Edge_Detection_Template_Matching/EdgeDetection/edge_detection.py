#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: achyut
"""
import numpy as np
import cv2 as cv
import copy
import math
import matplotlib.pyplot as plt

#read the image from the disk
# 1 is for color, 0 for grayscale, -1 for the image as it is
img = cv.imread("capitol.jpg", 0)
y = img.shape[0]
x = img.shape[1]


#input filter size
fs = int (input("Enter filter size"))
v = int((fs-1)/2)

#initialize filter matrix with 1's
filter = np.ones([fs,fs], dtype=float)

#copy the input image to the new image so that borders are pereserved
filteredImage = copy.deepcopy(img)
filteredImage.astype(float)

#loop to filter the input image and save it as a new image
for k in range(1, y-v):
    for l in range(1, x-v):
        sum = 0.0
        m = k-v
        n = l-v
        for i in range(0, fs):
            for j in range(0, fs):
                sum+= filter[i][j]  * float(img[m+i][n+j])
        filteredImage[k][l] = (sum / (fs*fs))

#Intialise the first derivative matrix along x-direction
dx= np.zeros([y,x], dtype=float)
dx.astype(float)

#loop to convolve the image with the [-1/2,0,1/2] mask
for i in range(0, y-1):
    for j in range(1, x-2):
        dx[i][j] = float(-((0.5)* filteredImage[i][j-1]) + ((0.5) * filteredImage[i][j+1]))

#save and show the image
cv.imwrite('dx_capitol.png', dx)
plt.imshow(dx, cmap='gray')
plt.show()

#Intialise the first derivative matrix along y-direction
dy= np.zeros([y,x], dtype=float)
for i in range(0, x-1):
    for j in range(1, y-2):
        dy[j][i] = float(-((0.5)* filteredImage[j-1][i]) + ((0.5) * filteredImage[j+1][i]))

plt.imshow(dy, cmap='gray')
plt.show()
cv.imwrite('dy_capitol.png', dy)


#Intialise the edge map matrix
dxdy = np.zeros([y,x], dtype=float)

#loop to calculate the vector norm i.e., edge map
for i in range(0, y-1):
    for j in range(0, x-1):
        dxdy[i][j] = math.sqrt ((dx[i][j] * dx[i][j]) + (dy[i][j] * dy[i][j]))

cv.imwrite('dxdy_capitol.png', dxdy)
plt.imshow(dxdy, cmap='gray')
plt.show()


#Intialise the orientation map
ormap = np.zeros([y,x], dtype=float)

for i in range(0, y):
    for j in range(0, x):
        # print (i, j)
        if dx[i][j] != 0:
            value = dy[i][j] / dx[i][j]
            ormap[i][j] = np.arctan(value)
        else:
            if dy[i][j] >= 0:
                value = float(math.pi/2)
                ormap[i][j] = (value)
            else:
                value = -float(math.pi/2)
                ormap[i][j] = (value)

#scale the orientation map in the range [0...255]
max_ormap = ormap.max()
min_ormap = ormap.min()
d = max_ormap - min_ormap

#Intialise the scaled orientation map matrix
scaled_ormap = np.zeros([y,x], dtype=float)

for i in range(1, y-1):
    for j in range(1, x-1):
        scaled_ormap[i][j]= int((np.arctan(ormap[i][j])- min_ormap) / d * 255)

#save and display the image
cv.imwrite('tan1_capitol.png', scaled_ormap)
plt.imshow(scaled_ormap, cmap='gray')
plt.show()

#Laplacian operator to detect edges
filteredImage = filteredImage.T
        
template = ([[0,-1,0], [-1,4,-1], [0,-1,0]])
template = np.asarray(template)
template = template.astype(float)

fxSize = template.shape[0]
fySize = template.shape[1]

vx = int((fxSize-1)/2)

vy = int((fySize-1)/2)

filteredImage1 = copy.deepcopy(filteredImage)
filteredImage1 = filteredImage1.astype(float)


for k in range(vx, x - vx):
    print (k)
    for l in range(vy, y - vy):
        sum = 0.0
        m = k - vx
        n = l - vy
        for i in range(0, fxSize):
            for j in range(0, fySize):
                sum += template[i][j] *  (filteredImage[m + i][n + j])
        filteredImage1[k][l] = sum
filteredImage1 = filteredImage1.T
cv.imwrite('log.jpg', filteredImage1)
plt.imshow(filteredImage1, cmap='gray')
plt.show()


