from scipy.spatial.transform import Rotation as R
import tf
import matplotlib.pyplot as plt
import numpy as np
import rospy
import cv2


def get_transformation_matrix(tvec, quat):
    transformation_matrix = np.eye(4)
    r = R.from_quat(quat)
    transformation_matrix[:3,:3] = r.as_dcm()
    transformation_matrix[:3,-1] = tvec
    return transformation_matrix


def euler_from_quaternion(quat):
    r = R.from_quat(quat)
    return r.as_euler('zyx', degrees=True)


def publish_threshold_frame(threshold):
    br = tf.TransformBroadcaster()
    br.sendTransform((0,0,threshold),
                     tf.transformations.quaternion_from_euler(0, 0, 0),
                     rospy.Time.now(),
                     "threshold_frame",
                     "map")
    return


def visualize_heightmap(heightmap, number = None):
    cv2.imwrite('../heightmaps/bag_24/frame_' + str(number) + '.jpg', heightmap*255)
    # plt.figure(2)
    # plt.clf()
    # if number%5 == 0:
    #     plt.imshow(heightmap, cmap = 'gray')
    # # # plt.savefig('../heightmaps/bag_3/frame_' + str(number))
    #     plt.pause(0.01)
    return


def visualize_histogram(datapoints, bin_sequence, f_idx):
    plt.figure(1); plt.clf()
    axes = plt.gca()
    axes.set_ylim([0,7000])
    axes.set_xlim([-0.03,0.08])
    data, bins, _ = plt.hist(datapoints, bin_sequence)
    plt.savefig('../histograms/bag_4/frame_' + str(f_idx))
    plt.pause(0.002)
    return data, bins


def log_data(f_idx, trench_threshold):
    print("Frame index is {}".format(f_idx))
    print("Trench threshold = {}".format(trench_threshold))
    print("----------------------------------------------")
    return


def discretize(points):
    x_co = (points[:,0]*1000).round().astype('int')
    y_co = (points[:,1]*1000).round().astype('int')
    z_co = points[:,2]
    return x_co, y_co, z_co


def mask_heightmap(points_array, bool_array, heightmap):
    points_array[:,0] -= np.amin(points_array[:,0])
    points_array[:,1] -= np.amin(points_array[:,1])
    points_array = points_array[bool_array == 1]
    x_co, y_co, z_co = discretize(points_array)
    heightmap[x_co, y_co,1] = 1
    return heightmap


def draw_circles(bbox, heightmap):
    plt.figure(2)
    plt.clf()
    plt.imshow(heightmap, cmap = 'gray')
    plt.scatter(bbox[0][0], bbox[0][1], color = 'g')
    plt.scatter(bbox[1][0], bbox[1][1], color = 'g')
    plt.plot((bbox[0][0], bbox[1][0]), (bbox[0][1], bbox[1][1]), '-')
    plt.pause(0.2)


def generate_histogram(datapoints, frame_idx, start = -0.03, end = 0.08, interval_size = 0.0005, draw = False):
    num = (end-start)//interval_size
    bin_sequence = np.linspace(start, end, num)
    if draw:    
        data, bins = utils.visualize_histogram(datapoints, bin_sequence, frame_idx)
    else:
        data, bins = np.histogram(datapoints, bin_sequence) 
    return data, bins


def get_trench_threshold(hist_data, hist_bins):
    max_idx = np.argmax(hist_data)
    threshold = hist_bins[max_idx]
    return threshold
