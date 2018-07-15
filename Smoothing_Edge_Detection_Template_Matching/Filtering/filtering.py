#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: achyut
"""
import numpy as np
import cv2 as cv
import copy
import matplotlib.pyplot as plt
debug = 1


def filtering(image, filterSize):
    y = image.shape[0]
    x = image.shape[1]

    v = int((filterSize-1)/2) #value so as to not to filter the border pixels

    #initialize filter matrix with 1's
    filter = np.ones([filterSize,filterSize], dtype=float)

    #copy the input image to the new image so that borders are pereserved
    filteredImage = copy.deepcopy(img)
    filteredImage.astype(float)

    #loop to filter the input image and save it as a new image
    for k in range(1, y-v):
        for l in range(1, x-v):
            sum = 0.0
            m = k-v
            n = l-v
            for i in range(0, filterSize):
                for j in range(0, filterSize):
                    sum+= filter[i][j]  * float(img[m+i][n+j])
            filteredImage[k][l] = (sum / (filterSize*filterSize))
    
    #save the filtered image
    cv.imwrite('c1.png',filteredImage)
    
    
    


#read the image from the disk
# 1 is for color, 0 for grayscale, -1 for the image as it is
#driver function
if __name__ == "__main__":
    img = cv.imread("capitol.jpg", 0)
    #input filter size
    fs = int (input("Enter filter size"))
    filtering(img, fs)