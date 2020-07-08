import numpy as np
import cv2
import os
import split_section as ss


def get_transition_points(img):
    img[img > 200] = 255
    img[img < 50] = 0
    x_co, y_co = ss.compute(img)
    slope = ss.compute_slope(x_co, y_co)
    transitions = ss.get_transition(slope)
    points = zip(x_co[transitions].astype('int'), 69-y_co[transitions].astype('int'))
    points.insert(0, (int(x_co[0]), int(69-y_co[0])))
    points.append(((int(x_co[-1]), int(69-y_co[-1]))))
    return points, x_co, y_co


def fit_line(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    if x1 == x2:
        m = float('inf')
        l = np.array([1,0,y1])
    else:
        m = (float(y2)-y1)/(float(x2)-x1)
        l = np.array([m, -1, y1-m*x1])
    # print('Slope is {}'.format(m))
    # print('Line is {}'.format(l))
    return l


def draw_line(img, pt1, pt2, i):
    color = ((255,0,0), (0,255,0), (0,0,255))
    img = cv2.line(img, pt1, pt2, color[i%3], 1)
    return img

def get_intercepts(line):
    m = line[0]
    c = line[2]
    if m == 0:
        y_int = int(c)
        return (0, y_int), (199, y_int)
    else:
        pts = []
        for y in [0,69]:
            x_int = int(round((y-c)/m))
            if x_int >=0 and x_int <= 199:
                pts.append((int(x_int), y))
        for x in [0, 199]:
            y_int = m*x + c
            if y_int >=0 and y_int <= 69:
                pts.append((x, int(y_int)))
        return pts[0], pts[1]

def compute_features(points):
    points = np.array(points)
    p1, p2, p3, p4, p5, p6 = points
    if p3[0] != p2[0]:
        slope1 = (p3[1]-p2[1])/(p3[0]-p2[0])
    else:
        slope1 = float('inf')
    
    if p4[0] != p5[0]:
        slope2 = (p4[1] - p5[1])/(p4[0]-p5[0])
    else:
        slope2 = float('inf')

    bottom_width = np.linalg.norm(p3-p4)
    # print('Bottom Width = {}'.format(bottom_width))

    top_width = np.linalg.norm(p2-p5)
    # print('Top Width = {}'.format(top_width))

    mp1 = (p3+p4)/2
    mp2 = (p2+p5)/2

    depth = np.linalg.norm(mp1-mp2)
    # print("Trench Depth = {}".format(depth))
    # print('--------------------------------------')
    return np.array([slope1, slope2, top_width, bottom_width, depth])


if __name__ == '__main__':
    file_path = '../slices/best/'
    for file_ in os.listdir(file_path):
        img1 = cv2.imread(file_path + file_, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(file_path + file_, cv2.IMREAD_GRAYSCALE)
        points, x_co, y_co = get_transition_points(img1)
        points.insert(0, (int(x_co[0]), int(69-y_co[0])))
        points.append(((int(x_co[-1]), int(69-y_co[-1]))))
        img1 = np.dstack((img1, img1, img1))
        img2 = np.dstack((img2, img2, img2))
        for i in range(len(points)-1):
            print("Points are {} {}".format(points[i], points[i+1]))
            # img = cv2.circle(img, (int(points[i][0]), int(points[i][1])), 1, (255,0,0), 1)
            # img = cv2.circle(img, (int(points[i+1][0]), int(points[i+1][1])), 1, (255,0,0), 1)
            line = fit_line(points[i], points[i+1])
            pt1, pt2 = get_intercepts(line)
            img1 = draw_line(img1, pt1, pt2, i)
            img2 = draw_line(img2, points[i], points[i+1], i)
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
            # break

        img1 = cv2.resize(img1, (600, 210), interpolation = cv2.INTER_AREA)
        img2 = cv2.resize(img2, (600, 210), interpolation = cv2.INTER_AREA)
        img = np.concatenate((img1, img2))
        cv2.imshow('Window 1', img)
        # cv2.imshow('Window 2', img2)
        cv2.waitKey(0)
        print(" ")
        print("------------------------------------------------------------------------")
        # break
    