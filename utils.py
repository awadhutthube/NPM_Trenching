from scipy.spatial.transform import Rotation as R
import tf
import matplotlib.pyplot as plt
import numpy as np
import rospy

def get_transformation_matrix(tvec, quat):
    transformation_matrix = np.eye(4)
    r = R.from_quat(quat)
    transformation_matrix[:3,:3] = r.as_dcm()
    transformation_matrix[:3,-1] = tvec
    return transformation_matrix

def publish_threshold_frame(threshold):
    br = tf.TransformBroadcaster()
    br.sendTransform((0,0,threshold),
                     tf.transformations.quaternion_from_euler(0, 0, 0),
                     rospy.Time.now(),
                     "threshold_frame",
                     "map")
    return

def visualize_heightmap(heightmap, number = None):
    plt.figure(2)
    plt.clf()
    plt.imshow(heightmap, cmap = 'gray')
    # plt.savefig('../heightmaps/bag_4/frame_' + str(number))
    plt.pause(0.01)
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
    # heightmap = np.zeros((dim,dim,3)).astype('float')
    # bool_array = bool_array[bool_array > 0]
    # print(bool_array.shape)
    points_array[:,0] -= np.amin(points_array[:,0])
    points_array[:,1] -= np.amin(points_array[:,1])
    points_array = points_array[bool_array > 0]
    x_co, y_co, z_co = discretize(points_array)
    heightmap[x_co, y_co] = 1

    return heightmap