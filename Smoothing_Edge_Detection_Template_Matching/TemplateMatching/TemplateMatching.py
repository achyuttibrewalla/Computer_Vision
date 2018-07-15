#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# print(template.shape) #gives (h, w)
@author: achyut
"""
import numpy as np
import cv2 as cv
import copy
import matplotlib.pyplot as plt

#read the image
image1 = cv.imread("shapes-bw.jpg", 0)
h1 = image1.shape[0]
w1 = image1.shape[1]

#read the template
template = cv.imread("ch.jpg", 0)
template = template.astype(float)

#reshaping the templae to odd size, as it will be used for filtering
template = np.delete(template, 67, axis=1)
template = np.delete(template, 67, axis=0)

#store the height and width of the template
template_height = template.shape[0]
template_width = template.shape[1]

#calculating the position in the image till which filter is to be applie
vx = int((template_height-1)/2)
vy = int((template_width-1)/2)

#padding the original image so that boundary can be filtered
image = np.pad(image1, ((vx,),(vy,)), 'constant')

x = image.shape[0]
y = image.shape[1]

#initialisng the matrix to store the correlated image
correlatedImage = np.zeros([x, y], dtype=float)

#list to store the mean of the sub regions of the image which is filtered using template
subRegionMean = []

#to store the magnitude(norm) of the sub regions of the image
norm_subRegion = []

#array of size of filter to calculate the respective sub region mean and the corresponding norm
subRegion = np.ones([template_height,template_width], dtype=float)


#loop to calculate the mean and norm of each of the sub regions in the image, the size of the 
#sub region is same as the size of the filter, since we have to make the subregion also zero mean before
#convolving it with the filter with zero mean.
for i in range(vx, x - vx):
    for j in range(vy, y - vy):
        for u in range(-vx, vx + 1):
            for v in range(-vy, vy + 1):
                subRegion[vx + u, vy + v] = image[i + u, j + v]
        
        norm_subRegion.append((np.linalg.norm(subRegion)))
        subRegionMean.append(np.mean(subRegion))
        
        
#calculating the mean of the template
filterMean = np.mean(template)

#subtracting each value of the filter with the mean of the filter, to obtain a zero-mean filter
for x1 in np.nditer(template, op_flags=['readwrite']):
   x1[...]= float(x1 - filterMean)

#calculate the norm of the filter
norm_filter = np.linalg.norm(template)

#loop to perform normalised cross correaltion of the image with the filter
#the mean of the respective sub region is deducted with each value before performing the cross-correlation
#the value after correlation calculation is divided by the norm of the filter and the sub region so as to
#obtain normalised cross correlation values between -1 and 1

for i in range(vx, x - vx):
    print(i)
    for j in range(vy, y - vy):
        sum = 0
        val = subRegionMean.pop(0)
        nI = norm_subRegion.pop(0)
        
        for u in range(-vx, vx + 1):
            for v in range(-vy, vy + 1):
                sum += (image[i + u, j + v] - val) * template[vx + u, vy + v]

        correlatedImage[i, j] = sum/ (norm_filter * nI)

#plot the correlated image
plt.imshow(correlatedImage, cmap='gray')
plt.show()

#apply thresholding to the correlated image so as to detect peaks.
pos = []
thresholdedImage = copy.deepcopy(correlatedImage)
for i in range(0, x):
    for j in range(0, y):
        if correlatedImage[i][j] < 0.6 :
            thresholdedImage[i][j] = 0
        else:
            pos.append([i, j])
                    
#plot the thresholded image
plt.imshow(thresholdedImage, cmap='gray')
plt.show()

#show the template on the image
templatePositionImage = copy.deepcopy(thresholdedImage)       
template = cv.imread("ch.jpg", 0)
x_offset=145
y_offset=144
templatePositionImage[y_offset:y_offset+template.shape[0], x_offset:x_offset+template.shape[1]] = template   
x_offset=146
y_offset=217
templatePositionImage[y_offset:y_offset+template.shape[0], x_offset:x_offset+template.shape[1]] = template
#plot the thresholded image
plt.imshow(templatePositionImage, cmap='gray')
plt.show() 



#Laplacian to the correlation image
template = ([[0,-1,0], [-1,4,-1], [0,-1,0]])
template = np.asarray(template)
template = template.astype(float)

fxSize = template.shape[0]
fySize = template.shape[1]

vx = int((fxSize-1)/2)

vy = int((fySize-1)/2)

LaplacedImage = copy.deepcopy(correlatedImage)
LaplacedImage = LaplacedImage.astype(float)


for k in range(vx, x - vx):
    print (k)
    for l in range(vy, y - vy):
        sum = 0.0
        m = k - vx
        n = l - vy
        for i in range(0, fxSize):
            for j in range(0, fySize):
                sum += template[i][j] *  (correlatedImage[m + i][n + j])
        LaplacedImage[k][l] = sum

plt.imshow(LaplacedImage, cmap='gray')
plt.show()

#scale the Laplaced image in the range [0...255]
max_LaplacedImage = LaplacedImage.max()
min_LaplacedImage = LaplacedImage.min()
d = max_LaplacedImage - min_LaplacedImage

#Intialise the scaled image matrix
scaled_LaplacedImage = np.zeros([x,y], dtype=float)

for i in range(0, x):
    for j in range(0, y):
        scaled_LaplacedImage[i][j]= ((LaplacedImage[i][j]- min_LaplacedImage) / d ) 

#applying thresholding to the scaled Image, the value of the scaled_laplaceImage lies between 0 and 1.0
thresholdedImage = copy.deepcopy(LaplacedImage)
for i in range(0, x):
    for j in range(0, y):
        if scaled_LaplacedImage[i][j] < 0.4 : #to be checked
            thresholdedImage[i][j] = 1
        else:
            thresholdedImage[i][j] = 0
plt.imshow(thresholdedImage, cmap='gray')
plt.show()



########################printing template########################################
#a = []
#thresholdedImage1 = copy.deepcopy(correlatedImage)
#for i in range(0, x):
#    for j in range(0, y):
#        if correlatedImage[i][j] < 0.6 :
#            thresholdedImage1[i][j] = 0
#        else:
#            a.append([i,j])
#                    
#print (a) 
#templatePositionImage = copy.deepcopy(thresholdedImage1)       
#plt.imshow(thresholdedImage1, cmap ='gray')
#plt.show()
#template = cv.imread("ch.jpg", 0)
#x_offset=145
#y_offset=144
#templatePositionImage[y_offset:y_offset+template.shape[0], x_offset:x_offset+template.shape[1]] = template   
#x_offset=146
#y_offset=217
#templatePositionImage[y_offset:y_offset+template.shape[0], x_offset:x_offset+template.shape[1]] = template
##plot the thresholded image
#plt.imshow(templatePositionImage, cmap='gray')
#plt.show()  
#################################################################################
