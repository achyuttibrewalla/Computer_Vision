import numpy as np
import cv2
from matplotlib import pyplot as plt

def plot(intensity, freq, name = "PLOT"):
    """
    function to plot the intensity vs frequency graph
    :param intensity: x-axis coordinates
    :param freq: y-axis coordinates
    :param name: graph name
    :return: None
    """
    #custom x-axis values
    # my_xticks = intensity[:]
    # plt.xticks(intensity, my_xticks)

    #plot the histogram using bar function
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
    for i in range(0, len(freq)):
        pdf.append(freq[i]/imageSize)

    return pdf

def cumulativeDensityFunction(pdfi):
    """
    function to calculate CDF
    pdfi: probability density function of the input image
    """
    pdf = pdfi
    cdf = []
    cdf.insert(0, pdf[0])
    for i in range(1, len(pdfi)):
        cdf.append(cdf[i - 1] + pdf[i])

    return cdf

def calcHistogram(image, bins):
    """
    function to calcualte and plot the histogram
    :param image: input image as nd-array
    :param bins: bin size
    :return: frequency of each intensity value
    """
    img = image
    imageSize = float(img.size)
    frequency = np.zeros(256)
    for pixelValue in np.nditer(img):
            frequency[pixelValue] += 1

    frequencyBin = []
    for i in range(0, 256, bins):
        sum = 0
        for j in range(i, i + bins):
            if j<256:
                sum += frequency[j]
        frequencyBin.append(sum)

    return frequencyBin

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
    for x in np.nditer(img, op_flags=['readwrite']):
        x[...] = round(255 * (cumulativeHist[x] - hMinimum) / (imgSize - hMinimum))

    # calculate and generate the  equalised histogram
    intensity = np.arange(0, 256)
    frequency = np.zeros(256)
    for pixelValue in np.nditer(img):
        if pixelValue in intensity:
            frequency[pixelValue] += 1
    plot(intensity, frequency, "EQUALIZED HISTOGRAM")

    #calculating and plotting PDF
    pdf = probabilityDensityFunction(frequency, imgSize)
    plot(intensity, pdf, "PDF OF EQUALISED HISTOGRAM")

    # calculating and plotting CDF
    cdf = cumulativeDensityFunction(pdf)
    plot(intensity, cdf, "CDF OF EQUALISED HISTOGRAM")

    #save the image after equalisation
    cv2.imwrite('OUTPUT.png', img)


#driver function
if __name__ == "__main__":

    #read the input image
    img = cv2.imread('checker.png',0)

    #calculate the imaege size
    imgSize = float(img.size)

    #input from the user
    n = input("Enter the number of bins")

    grayLevel = 256
    binSize = int((round(grayLevel / n)))

    intensity = np.arange(binSize, 256+binSize, binSize)
    # print intensity
    frequency = calcHistogram(img, binSize)
    frequency = map(int, frequency)
    plot(intensity, frequency, "HISTOGRAM")
    # print frequency


    #calculate and plot PDF
    pdf = probabilityDensityFunction(frequency, imgSize)
    plot(intensity, pdf, "PDF")

    #calculate and plot CDF
    cdf = cumulativeDensityFunction(pdf)
    plot(intensity, cdf, "CDF")

    #call to histogram equalisation function
    histogramEqualization(img)