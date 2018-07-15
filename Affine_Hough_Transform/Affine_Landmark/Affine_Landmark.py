import cv2
import numpy as np
import copy
from matplotlib import pyplot as plt
points_x=[]
points_y=[]

ix,iy = -1,-1
# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy
    if event == cv2.EVENT_LBUTTONDBLCLK:
        # cv2.circle(img,(x,y),100,(255,0,0),-1)
        points_x.append(x)
        points_y.append(y)
        ix,iy = x,y


def Interpolation(image, transformation_matrix):
    #Apply the interpolation to calculate the intensity in the target image
    img = copy.deepcopy(image)
    count = 0
    shape = img.shape
    transImg = np.zeros(shape)
    trans_matrix_inverse = np.linalg.inv(transformation_matrix)
    for x in range(0, shape[0]):
            for y in range(0, shape[1]):
                count+=1
                point_mat = np.array([ [x], [y], [1]])
                temp_mat =  np.matmul(trans_matrix_inverse , point_mat)
                a = int(temp_mat[0][0])
                b = int(temp_mat[1][0])
                if (a>0) and (a<shape[0]) and (b>0) and (b < shape[1]):
                    transImg[x][y] = img[a][b]

    print("count:", count)
    return transImg

if __name__ == "__main__":
    # Create a black image, a window and bind the function to window
    img = cv2.imread("CTscan.jpg", 0)
    cv2.namedWindow('SourceImage')
    cv2.setMouseCallback('SourceImage',draw_circle)

    while(1):
        cv2.imshow('SourceImage',img)
        k = cv2.waitKey(20) & 0xFF
        # print(ix,iy)
        if k == 27:
            break
    cv2.destroyAllWindows()

    spoints_x = points_x[:]
    spoints_y = points_y[:]
    print(spoints_x)
    print(spoints_y)

    points_x = []
    points_y = []

    targetimg = cv2.imread("CTscani.jpg", 0)
    cv2.namedWindow('TargetImage')
    cv2.setMouseCallback('TargetImage', draw_circle)

    while (1):
        cv2.imshow('TargetImage', targetimg)
        k = cv2.waitKey(20) & 0xFF
        # print(ix,iy)
        if k == 27:
            break
    cv2.destroyAllWindows()

    tpoints_x = points_x[:]
    tpoints_y = points_y[:]

    source_matrix = np.zeros([0,6], dtype='float')
    target_matrix = np.zeros([0,1], dtype='float')

    #Initialisng the matrix from the pixel locations from the source image
    for i in range(0, len(spoints_x)):
        source_matrix = np.append(source_matrix, [[spoints_x[i], spoints_y[i], 1, 0, 0, 0]], axis=0)
        source_matrix = np.append(source_matrix, [[0, 0, 0, spoints_x[i], spoints_y[i], 1]], axis=0)
    print(source_matrix)

    # Initialisng the matrix from the pixel locations from the target image
    for j in range(0, len(tpoints_x)):
        target_matrix = np.append(target_matrix, [[tpoints_x[j]]], axis=0)
        target_matrix = np.append(target_matrix, [[tpoints_y[j]]], axis=0)


    #solve the linear equation
    trans_matrix = np.linalg.solve(source_matrix, target_matrix)


    a11 = trans_matrix[0][0]
    a12 = trans_matrix[1][0]
    a13 = trans_matrix[2][0]
    a21 = trans_matrix[3][0]
    a22 = trans_matrix[4][0]
    a23 = trans_matrix[5][0]

    #Initialising the transformation matrix
    trans_matrix1 = np.array([[a11, a12, a13], [a21, a22, a23], [0, 0, 1]])
    transImg = Interpolation(img, trans_matrix1)


    #Display the image
    plt.imshow(transImg, cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

















