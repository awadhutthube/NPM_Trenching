import os
import cv2

if __name__ == '__main__':
    file_path = '../slices/bag_1/'
    for file_ in os.listdir(file_path):
        img1 = cv2.imread(file_path + file_, cv2.IMREAD_GRAYSCALE)
        img1 = cv2.resize(img1, (600, 210), interpolation = cv2.INTER_AREA)
        cv2.imwrite('../resize/bag_1/' + file_, img1)