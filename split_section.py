import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

plt.ion()

def compute_slope(hor, ver):
    slope = (ver[1:] - ver[:-1])/(hor[1:] - hor[:-1]+1)
    return slope

def compute(image):
    '''
    Returns coordinates of points in 2D cross section of a slice
    Input: PxQ array of representing the map or 2D profile
    '''
    row, column = np.arange(image.shape[0]), np.arange(image.shape[1])
    xx, yy = np.meshgrid(row, column)
    xx = xx.flatten('F')
    yy = yy.flatten('F')
    flat_image = image.flatten()
    xx = xx[flat_image > 0]
    yy = yy[flat_image > 0]

    dict_ = average_height(xx, yy)
    hor = []; ver = []
    for key in dict_.keys():
        val = dict_[key]
        hor.append(key)
        ver.append(69 - round(sum(val)/len(val), 3))
    hor, ver = np.array(hor), np.array(ver)
    hor , ver = quantize(hor, ver)
    return hor, ver

def quantize(hor, ver):
    hor_split = np.array_split(hor, 20)
    ver_split = np.array_split(ver, 20)
    hor_avg = []; ver_avg = []
    for i in range(len(hor_split)):
        hor_avg.append(np.mean(hor_split[i]))
        ver_avg.append(np.mean(ver_split[i]))
    return np.array(hor_avg), np.array(ver_avg)

def average_height(xx, yy):
    dict_ = {}
    xx = xx.astype('float')
    for i in range(len(yy)):
        key = yy[i]
        if key in dict_:
            dict_[key].append(xx[i])
        else:
            dict_[key] = [xx[i]]
    return dict_

# def get_transition(slope):
#     '''
#     Compute transitions points for portions of the 2D section
#     Input: Array containing slope at each point on the 2D section
#     Output: Points where the section changes
#     '''
#     transition = []
#     flag = True
#     val = float('inf')
#     print(slope)
#     for i in range(len(slope)-1):
#         if slope[i]*slope[i+1] < 0:
#             transition.append(i)
#             if slope[i] < 0:
#                 val = slope[i+1]
#         elif slope[i] == 0:
#             val = 0
#             transition.append(i)
#         elif slope[i] - val > 0.3:
#             transition.append(i)
#             val = float('inf')
#     return transition

def get_transition(slope):
    '''
    Compute transitions points for portions of the 2D section
    Input: Array containing slope at each point on the 2D section
    Output: Points where the section changes
    '''
    transition = []
    flag = True
    mark = False
    val = float('inf')
    # print(slope)
    for i in range(len(slope)-1):
        if abs(slope[i]) >= 0.3 and flag:
            transition.append(i)
            flag = False
            mark = True
        elif abs(slope[i]) < 0.3 and mark:
            mark = False
            flag = True
            transition.append(i)
        # elif slope[i] - val > 0.3:
        #     transition.append(i)
        #     val = float('inf')
    return transition


def determine_characteristics(path):
    plt.figure()
    for file_ in os.listdir(path):
        img = cv2.imread(path+file_,  cv2.IMREAD_GRAYSCALE)
        img[img > 200] = 255
        img[img < 50] = 0
        hor, ver = compute(img)
        slope = compute_slope(hor, ver)
        # slope = compute_slope(np.arange(len(slope)), np.array(slope))
        transition = get_transition(slope)
        points = zip(hor[transition], ver[transition])
        # plt.clf()
        # plt.plot(slope)
        # plt.scatter(transition, slope[transition])
        # plt.savefig('../slopes/' + file_)
        # plt.show(); plt.pause(0.2)
        if len(points) == 4:
            img = np.dstack((img,img,img))
            for pt in points:
                img = cv2.circle(img, (int(pt[0]), 69 - int(pt[1])), 1, (255,0,0), 2) 
                print(int(pt[0]), int(pt[1]))
            img = cv2.resize(img, (600, 210), interpolation = cv2.INTER_AREA)
            cv2.imwrite('../transition/bag_3/' + file_, img)
            # cv2.imshow('Window 1', img)
            # cv2.waitKey(0)
    return points

if __name__ == '__main__':
    file_path = '../slices/bag_3/'
    determine_characteristics(file_path)
