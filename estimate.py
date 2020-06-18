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

def project_section(section):
    section = sort_increasing(section)
    y_co = (section[:,1]*3000).round().astype('int')
    z_co = (section[:,2]*3000).round().astype('int')
    y_co -= np.amin(y_co)
    z_co -= np.amin(z_co)
    # slope = compute_slope(y_co, z_co)
    # slope = np.append(slope,1)
    s_map = np.zeros((200,500))
    s_map[199-z_co, 499-y_co] = 1
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
        ver.append(200 - round(sum(val)/len(val), 4))
    hor , ver = quantize(hor, ver)
    hor, ver = np.array(hor), np.array(ver)
    return hor, ver

def quantize(hor, ver):
    hor_split = np.array_split(hor, 30)
    ver_split = np.array_split(ver, 30)
    hor_avg = []
    ver_avg = []
    print(len(hor_split), len(ver_split))
    for i in range(30):
        hor_avg.append(np.mean(hor_split[i]))
        ver_avg.append(np.mean(ver_split[i]))
        continue
    # print(ver_avg)
    return hor_avg, ver_avg

def average_height(xx, yy):
    dict_ = {}
    xx = xx.astype('float')
    yy = yy.astype('float')
    for i in range(len(yy)):
        key = yy[i]
        if key in dict_:
            dict_[key].append(xx[i])
        else:
            dict_[key] = [xx[i]]
    return dict_

def compute_slope(hor, ver):
    slope = (ver[1:].astype('float') - ver[:-1].astype('float'))/(hor[1:] - hor[:-1] + 1)
    return slope

def sort_increasing(array):
    return array[array[:,1].argsort()]

def determine_characteristics(path):
    plt.figure()
    for file_ in os.listdir(path):
        img = cv2.imread(path+file_,  cv2.IMREAD_GRAYSCALE)
        compute(img)
        hor, ver = compute(img)
        slope = compute_slope(hor, ver)
        slope[slope > 0.1] = 0.3
        slope[slope < -0.1] = -0.3
        plt.clf()
        plt.plot(slope)
        plt.show()
        plt.savefig('../slopes/' + file_)
        plt.pause(0.2)
        cv2.imshow('Window 1', img)
        cv2.waitKey(0)

if __name__ == '__main__':
    file_path = '../slices/bag_1/'
    determine_characteristics(file_path)
