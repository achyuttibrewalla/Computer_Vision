import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt

def plot(intensity, freq, name = "PLOT"):
    """
        function to plot the intensity vs frequency graph
        :param intensity: x-axis coordinates
        :param freq: y-axis coordinates
        :param name: graph name
        :return: None
        """
    # plot the histogram using bar function
    plt.bar(intensity, freq)
    plt.title(name)
    plt.show()

def probabilityDensityFunction(freq, imageSize):
    """
        function to calculate PDF
        freq: list of frequencies of each intensity value
        imageSize: size of the inout image
        """
    pdf = []
    for i in range(0, 256):
        pdf.append(freq[i]/imageSize)
    # plot(intensity, pdf, "PDF")
    return pdf

def cumulativeDensityFunction(pdfi):
    """
        function to calculate CDF
        pdfi: probability density function of the input image
    """
    pdf = pdfi
    cdf = []
    cdf.insert(0, pdf[0])
    for i in range(1, 256):
        cdf.append(cdf[i - 1] + pdf[i])
    # plot(intensity, cdf, "CDF")
    return cdf

def histogramEqualization(image):
    """
    function to calculate the equalized histogram
    :param image: input image
    :return: None
    """
    img1 = image
    imgSize = img1.size
    h = np.zeros(256)
    for pixelValue in np.nditer(img1):
        h[pixelValue] += 1

    #finding the minimum intensity value in the image
    gMinimum = 0
    for i in range(len(h)):
        if h[i] > 0:
            gMinimum = i
            break

    # calculating cumulative histogram
    cumulativeHist = []
    cumulativeHist.insert(0, h[0])
    for i in range(1, 256):
        cumulativeHist.append(cumulativeHist[i - 1] + h[i])
    hMinimum = cumulativeHist[gMinimum]

    #loop to iterate and overwrite the new intensity values in the input image
    for x in np.nditer(img1, op_flags=['readwrite']):
        x[...] = round(255 * (cumulativeHist[x] - hMinimum) / (imgSize - hMinimum))

    # calculate and generate the  equalised histogram
    intensity = np.arange(0, 256)
    frequency = np.zeros(256)
    for pixelValue in np.nditer(img1):
        if pixelValue in intensity:
            frequency[pixelValue] += 1
    plot(intensity, frequency, "EQUALIZED HISTOGRAM")

    return (intensity, frequency)


# READ A POOR CONTRAST IMAGE
img = cv2.imread('crowd.png',0)

#shape of the 2d image
shape = img.shape

#total image size
imageSize = float(img.size)

#equalize the histogram
intensity, image_PC_freq = histogramEqualization(img)


#calculate and plot PDF
pdf1 = probabilityDensityFunction(image_PC_freq, imageSize)
plot(intensity, pdf1, "PDF_")

#calculate and plot CDF
cdf1 = cumulativeDensityFunction(pdf1)
plot(intensity, cdf1, "CDF_")


#IMAGE WITH GOOD CONTRAST
img2 = cv2.imread('checker.png',0)

imageSize = float(img2.size)

#equalize the histogram
intensity, image_GC_freq = histogramEqualization(img2)


#calculate and plot PDF
pdf2 = probabilityDensityFunction(image_GC_freq, imageSize)
plot(intensity, pdf2, "PDF")

#calculate and plot CDF
cdf2 = cumulativeDensityFunction(pdf2)
plot(intensity, cdf2, "CDF")

#intialise the lookup table for histogram matching
lookuptable = np.zeros(256)

pos = -1
for i in range(256):
    minValue = sys.maxint
    for j in range(256):
        diff = abs(cdf1[i] - cdf2[j])
        # print diff
        if diff < minValue:
            minValue = diff
            pos = j
    lookuptable[i] = pos

#a new 2d array to store the new intensity value of the input image, i.e., image with a bad contrast
equ = np.zeros((shape[0], shape[1]))
for i in range(shape[0]):
    for j in range(shape[1]):
        v = img[i][j]
        equ[i][j] = lookuptable[v]


#save the image to disk
cv2.imwrite('output_crowd_2312.png',equ)


frequency245 = np.zeros(256)
#calculate and plot the histogram of the adjusted input image
for pixelValue in np.nditer(equ):
    if pixelValue in intensity:
        frequency245[int(pixelValue)] += 1
intensity = np.arange(0,256)
plot(intensity, frequency245, "HISTOGRAM_FINAL")

imS = shape[0] * shape[1]

#calculate and plot PDF of the adjusted input image
pdf3 = probabilityDensityFunction(frequency245, imS)
plot(intensity, pdf3, "PDF_FINAL")

#calculate and plot the CDF of the adjusted input image
cdf3 = cumulativeDensityFunction(pdf3)
plot(intensity, cdf3, "CDF_FINAL")

