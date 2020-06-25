import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

plt.ion()
def slice_points(cloud_array, n_slices = 5):
    # slices = np.split(cloud_array, n_slices, axis = 0)
    max_val = np.amax(cloud_array[:,0])
    min_val = np.amin(cloud_array[:,0])
    jump = (max_val - min_val)/n_slices
    slice_points = [min_val+i*jump for i in range(n_slices+1)]
    return slice_points

def get_section(cloud_array, low_bound, high_bound):
    section = cloud_array[cloud_array[:,0] >= low_bound]
    section = section[section[:,0] < high_bound]
    return section

def project_section(section, idx, i):
    section = sort_increasing(section)
    y_co = (section[:,1]*1000).round().astype('int')
    z_co = (section[:,2]*1000).round().astype('int')
    y_co -= np.amin(y_co)
    z_co -= np.amin(z_co)
    unique, counts = np.unique(y_co,return_counts = True)
    dict_ = {element: round(np.mean(z_co[y_co == element])) for element in unique}
    row = np.array(dict_.values()).astype('int')
    column = np.array(dict_.keys()).astype('int')
    s_map = np.zeros((70,200)).astype('int')
    s_map[69-row, 199-column] = 255
    return s_map

def compute(image):
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
    plot = visualize_profile(hor.astype('int'), ver.astype('int'))
    # hor , ver = quantize(hor, ver)
    return hor, ver, plot

def visualize_profile(hor, ver):
    s_map = np.zeros((70,200))
    s_map[69-ver, hor] = 255
    return s_map

def quantize(hor, ver):
    hor_split = np.array_split(hor, 30)
    ver_split = np.array_split(ver, 30)
    hor_avg = []
    ver_avg = []
    for i in range(30):
        hor_avg.append(np.mean(hor_split[i]))
        ver_avg.append(np.mean(ver_split[i]))
        continue
    # print(ver_avg)
    return np.array(hor_avg), np.array(ver_avg)

def average_height(xx, yy):
    dict_ = {}
    xx = xx.astype('float')
    # yy = yy.astype('float')
    for i in range(len(yy)):
        key = yy[i]
        if key in dict_:
            dict_[key].append(xx[i])
        else:
            dict_[key] = [xx[i]]
    return dict_

def compute_slope(hor, ver):
    slope = (ver[1:] - ver[:-1])/(hor[1:] - hor[:-1]+1)
    return slope

def sort_increasing(array):
    return array[array[:,1].argsort()]

def determine_characteristics(path):
    plt.figure()
    for file_ in os.listdir(path):
        img = cv2.imread(path+file_,  cv2.IMREAD_GRAYSCALE)
        img[img > 200] = 255
        img[img < 50] = 0
        hor, ver, plot = compute(img)
        slope = compute_slope(hor, ver)
        print(slope)
        plt.clf()
        # # plt.plot(hor, ver)
        plt.plot(slope)
        plt.show()
        plt.pause(0.2)
        img = cv2.resize(img, (600, 210), interpolation = cv2.INTER_AREA)
        cv2.imshow('Window 1', img)
        # plot = cv2.resize(plot, (1000, 350), interpolation = cv2.INTER_AREA) 
        # cv2.imshow('Window 2', plot)
        cv2.waitKey(0)

if __name__ == '__main__':
    file_path = '../slices/bag_1/'
    determine_characteristics(file_path)
