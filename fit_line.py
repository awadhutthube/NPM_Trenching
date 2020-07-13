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
    points = zip(x_co[transitions].astype('int'), y_co[transitions].astype('int'))
    points.insert(0, (int(x_co[0]), int(y_co[0])))
    points.append(((int(x_co[-1]), int(y_co[-1]))))
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
    img = cv2.line(img, (int(pt1[0]), int(69-pt1[1])), (int(pt2[0]), int(69-pt2[1])), color[i%3], 1)
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

def get_intersection(line, pt):
    y = pt[1]
    # print(line)
    x = (y - line[2])/line[0]
    # print(x,y)
    return (x, y)

def compute_features(img1, points, x_co, y_co, flip):
    x_co = np.array(x_co)
    y_co - np.array(y_co)
    points = np.array(points).astype('float')
    p1, p2, p3, p4, p5, p6 = points

    l = fit_line(p4, p5)
    pt1, pt2 = get_intercepts(l)
    point = get_intersection(l, p2)
    # print(points)
    # print(point)
    # img_1 = cv2.circle(img1, (int(point[0]), int(69 - point[1])), 1, (255,0,0), 2)
    # n_img = draw_line(img1, pt1, pt2, 1)
    # n_img = draw_line(n_img, p2, point, 1)
    # cv2.imshow("Line Fit", n_img)
    # cv2.waitKey(0)

    if not flip:
        slope1 = two_point_slope(p2, p3)
        slope2 = two_point_slope(p4, p5)
        slope3 = two_point_slope(p5, p6)

        flat_ref = y_co[x_co < p2[0]]
        trench_ref = y_co[x_co > p3[0]]
        x_co = x_co[x_co > p3[0]]
        trench_ref = trench_ref[x_co < p4[0]]

        bottom_width = np.linalg.norm(p3-p4)
        top_width = np.linalg.norm(p2-point)
        depth = np.mean(flat_ref) - np.mean(trench_ref)
        pile_height = (p5[1]) - np.mean(flat_ref)
    else:
        slope3 = two_point_slope(p1, p2)
        slope2 = two_point_slope(p2, p3)
        slope1 = two_point_slope(p3, p4)

        flat_ref = y_co[x_co > p5[0]]
        trench_ref = y_co[x_co > p3[0]]
        x_co = x_co[x_co > p3[0]]
        trench_ref = trench_ref[x_co < p4[0]]

        bottom_width = np.linalg.norm(p3-p4)
        top_width = np.linalg.norm(p2-point)
        depth = np.mean(flat_ref) - np.mean(trench_ref)
        pile_height = (p2[1]) - np.mean(flat_ref)

    # print(flat_ref, p5[1], pile_height)
    # print(np.array([slope1, slope2, top_width, bottom_width, depth, pile_height, slope3]))
    return np.array([slope1, slope2, top_width, bottom_width, depth, pile_height, slope3])

def two_point_slope(pt1, pt2):
    if pt1[0] != pt2[0]:
        slope = (pt1[1]-pt2[1])/(pt1[0]-pt2[0])
    else:
        slope = float('inf')
    return slope

if __name__ == '__main__':
    file_path = '../output/slices/bag_1/'
    for file_ in os.listdir(file_path):
        img1 = cv2.imread(file_path + file_, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(file_path + file_, cv2.IMREAD_GRAYSCALE)
        points, x_co, y_co = get_transition_points(img1)
        # points.insert(0, (int(x_co[0]), int(69-y_co[0])))
        # points.append(((int(x_co[-1]), int(69-y_co[-1]))))

        if len(points) == 6:
            # print(points)
            img1 = np.dstack((img1, img1, img1))
            compute_features(img1, points, x_co, y_co, False)
        
        # img2 = np.dstack((img2, img2, img2))
        # if len(points) == 6:
        #     for i in range(len(points)-1):
        #         print("Points are {} {}".format(points[i], points[i+1]))
        #         # img = cv2.circle(img, (int(points[i][0]), int(points[i][1])), 1, (255,0,0), 1)
        #         # img = cv2.circle(img, (int(points[i+1][0]), int(points[i+1][1])), 1, (255,0,0), 1)
        #         line = fit_line(points[i], points[i+1])
        #         pt1, pt2 = get_intercepts(line)
        #         img1 = draw_line(img1, pt1, pt2, i)
        #         img2 = draw_line(img2, points[i], points[i+1], i)
        #         print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
        #         # break

        #     img1 = cv2.resize(img1, (600, 210), interpolation = cv2.INTER_AREA)
        #     img2 = cv2.resize(img2, (600, 210), interpolation = cv2.INTER_AREA)
        #     img = np.concatenate((img1, img2))
        #     print(file_)
        #     cv2.imwrite('../output/lines/bag_1/' + file_, img2)
        #     # cv2.imshow('Window 1', img)
        #     # # cv2.imshow('Window 2', img2)
        #     # cv2.waitKey(0)
        #     # print(" ")
        #     # print("------------------------------------------------------------------------")
        #     # break
    