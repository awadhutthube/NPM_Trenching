import numpy as np
import cv2
import matplotlib.pyplot as plt
import utils

def segment_wheel(points, threshold):
    min_array = np.amin(points, axis = 0)
    points -= min_array
    bbox, heightmap = get_wheel_bbox(points, threshold - min_array[2] + 0.04)
    return bbox, heightmap


def get_wheel_bbox(points, shifted_threshold, dim = 500):
    heightmap = np.zeros((dim,dim)).astype('uint8')
    x_co, y_co, z_co = utils.discretize(points)
    x_co = x_co[z_co > shifted_threshold]
    y_co = y_co[z_co > shifted_threshold]
    heightmap[x_co, y_co] = 255
    kernel = np.ones((5,5),np.uint8)
    heightmap = cv2.dilate(heightmap,kernel,iterations = 2)
    _, cnt, _ = cv2.findContours(heightmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnt, key = cv2.contourArea)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    bbox = np.int0(box)
    return bbox, heightmap


def visualize_wheel_segment(bbox, heightmap):
    cv2.drawContours(heightmap,[bbox],0,(255,0,0),2)
    return heightmap


def fit_line(trans, slope, idx):
    shift1 = 80
    shift2 = 120
    m = np.tan(np.pi*-slope/180)
    x, y, z = trans
    if m != 0:
        l1 = -np.array([m, -1, y - m*x - shift1])
        l2 = -np.array([m, -1, y - m*x + shift1])
        l3 = -np.array([-1/m, -1, y - (-1/m)*x - shift2])
    else:   
        l1 = np.array([0,1,-y + shift1])
        l2 = np.array([0,1,-y - shift1])
        l3 = np.array([1,0,-x + shift2])

    
    return l1, l2, l3


def get_points(bbox):
    a = bbox[0]
    b = bbox[1]
    return a, b


def check_side(points, l1, l2, l3):
    points[:,0] -= np.amin(points[:,0])
    points[:,1] -= np.amin(points[:,1])  

    points[:,0] = (points[:,0]*1000).round().astype('int')
    points[:,1] = (points[:,1]*1000).round().astype('int')
    points[:,2] = 1

    bool_array1 = points*l1
    bool_array1 = np.sum(bool_array1, axis = 1)

    bool_array2 = points*l2
    bool_array2 = np.sum(bool_array2, axis = 1)

    bool_array3 = points*l3
    bool_array3 = np.sum(bool_array3, axis = 1)

    bool_array1[bool_array1 > 0] = 1
    bool_array1[bool_array1 < 0] = 0
    bool_array2[bool_array2 > 0] = 0
    bool_array2[bool_array2 < 0] = 1

    bool_array3[bool_array3 > 0] = 0
    bool_array3[bool_array3 < 0] = 1

    bool_array = np.multiply(bool_array1, bool_array2)
    bool_array = np.multiply(bool_array, bool_array3)
    # print(bool_array)    
    return bool_array