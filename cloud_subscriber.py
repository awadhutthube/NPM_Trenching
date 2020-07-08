#! /usr/bin/python
import numpy as np
from sensor_msgs.msg import PointCloud2
import ros_numpy
import rospy
import sensor_msgs.point_cloud2 as pcl2
import std_msgs
import cv2
import matplotlib.pyplot as plt
import utils
import wheel_boundary as wb
import copy
import tf
import sys
import estimate
import time
import fit_line as fl
from rover_msgs.msg import FeatureArray as fa
from rover_msgs.msg import FeatureMsg as fm

# Non-blocking visualization of images
plt.ion()

# Creating publisher objects
pub1 = rospy.Publisher('/cloud_transformed', PointCloud2, queue_size=5)
pub2 = rospy.Publisher('/cloud_original', PointCloud2, queue_size=5)
feature_pub = rospy.Publisher('/features', fa, queue_size=5)

# Initialization of constants and utility variables
idx = 0
tvec = [-0.21973428009999108, -0.3433707857423185, -0.031694521194154585]
quat = [0.9403625348787676, 4.915599287674821e-05, -0.00033386367422225116, -0.3401737631198417]
# tvec = [-0.16927510039335014, -0.3717676310584948, -0.0332700755223688]
# quat = [0.9378418721923328, 0.1422734925170896, -0.04777733124311015, -0.31293482182246196]
mean_array = []
num_slices = 5

def cloud_sub_callback(msg):
    global idx, mean_array, num_slices
    xyz = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg, remove_nans=True)
        
    # Transforming cloud into map frame and normalizing for 0 mean
    H = utils.get_transformation_matrix(tvec, quat)
    transformed_xyz = transform_cloud(xyz, H)
    mean_xyz = np.mean(transformed_xyz, axis = 0)
    transformed_xyz -= mean_xyz
    
    # publish_transformed_cloud(transformed_xyz,1)
    # Finding the wheel's pose w.r.t the robot's base_link
    try:
        (trans,rot) = tf_tree.lookupTransform('base_link', 'RR', rospy.Time(0))
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        print("No TF data")
        return
    
    # Computing translation and rotation of the wheel frame along each axis
    euler_angles = utils.euler_from_quaternion(rot)
    trans -= np.amin(transformed_xyz, axis = 0)
    trans -= mean_xyz
    trans = (trans*1000).round()

    # Generating heightmap using all points in the cloud
    heightmap = generate_heightmap(transformed_xyz.copy())

    # Segmenting the trench using equations of lines passing through the wheel's frame
    line1, line2, line3, line4 = wb.fit_line(trans, 0, idx)
    bool_array = wb.check_side(transformed_xyz.copy(), line1, line2, line3, line4)
    heightmap = utils.mask_heightmap(transformed_xyz, bool_array, heightmap)
    # utils.visualize_heightmap(heightmap)
    # Publishing the segmented cloud and logging essential data
    
    transformed_xyz = transformed_xyz[bool_array == 1]
    transformed_xyz -= np.mean(transformed_xyz, axis = 0)
    # z_quat = utils.quaternion_from_euler([0,0, np.pi*euler_angles[1]/180])
    # H = utils.get_transformation_matrix(np.zeros(3), z_quat)
    # transformed_xyz = transform_cloud(transformed_xyz, H)
    # transformed_xyz[:,0] -= np.amin(transformed_xyz[:,0])
    publish_transformed_cloud(transformed_xyz,0)
    
    intervals = estimate.slice_points(transformed_xyz, num_slices)

    boolean_list = [False]*num_slices
    feature_list = [None]*num_slices

    for i in range(num_slices):
        section = estimate.get_section(transformed_xyz, intervals[i], intervals[i+1])
        img1 = estimate.project_section(section, idx, i)
        cv2.imwrite('../slices/bag_2/slice{}img{}.jpg'.format(i,idx), img1)
        points, x_co, y_co = fl.get_transition_points(img1)        
        if len(points) != 6:
            continue

        feature_list[i] = fl.compute_features(points, x_co, y_co)
        boolean_list[i] = True

        # img1 = utils.visualize_section_lines(points, img1)
    
    publish_feature_msg(feature_list, boolean_list)
    print("Frame index is {}".format(idx))
    print('Bool Array = {}'.format(boolean_list))
    print('Feature Array = {}'.format(feature_list))
    print('-------------------------------------------')
    idx += 1
    return

def publish_feature_msg(f_list, b_list):
    h = std_msgs.msg.Header()
    h.stamp = rospy.Time.now()
    feature_array = fa()
    feature_array.header = h
    for idx, element in enumerate(f_list):
        feature_msg = fm()
        if element == None:
            continue
        else:
            feature_msg.section_no = idx
            feature_msg.slope1, feature_msg.slope2, feature_msg.top_width, feature_msg.bottom_width, feature_msg.trench_depth, feature_msg.pile_height, feature_msg.slope3 = element
            feature_array.boolean_list = b_list
            feature_array.feature_list.append(feature_msg)
    feature_pub.publish(feature_array)
    return 

def plot_graph(array):
    plt.plot(array)
    plt.show()
    plt.pause(0.5)
    plt.savefig('../plots/bag_4')
    return

def generate_heightmap(points, dim = 500):
    '''
    Create a 2D representation of the point cloud
    Input: Nx3 array of points in cloud
           Value of threshold (debugging purposes)
           Dimension of 2D image
           Mask Variable (debugging purposes)
    Output: 2D image of size (dim x dim) representing the point cloud data
    '''
    heightmap = np.zeros((dim,dim)).astype('float')
    x_co, y_co, z_co = utils.discretize(points)
    x_co -= np.amin(x_co)
    y_co -= np.amin(y_co)
    min_val = np.amin(z_co)
    z_co -= min_val
    heightmap[x_co,y_co] = z_co*10
    return heightmap


def publish_transformed_cloud(cloud_array, c, flag = True):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    '''
    Publishes pointcloud w.r.t base_link
    '''
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = 'base_link'
    if not flag:
        header.frame_id = 'sensor/side_camera_depth_optical_frame'
        cloud = pcl2.create_cloud_xyz32(header,cloud_array)
        pub2.publish(cloud)
    else:
        cloud = pcl2.create_cloud_xyz32(header,cloud_array)
        if c == 0:
            pub1.publish(cloud)
        else:
            pub2.publish(cloud)
    return

def transform_cloud(cloud_array, transformation_matrix = np.eye(4)):
    '''
    Transforms all points in a cloud given a transformation matrix
    Input: Nx3 array of points in cloud
    Output: Nx3 array of transformed points
    '''
    if cloud_array.shape[0] > cloud_array.shape[1]:
        homogenious_coordinates = np.vstack((cloud_array.T, np.ones(cloud_array.shape[0])))
    else:
        homogenious_coordinates = np.vstack((cloud_array, np.ones(cloud_array.shape[1])))

    transformed_cloud = np.matmul(transformation_matrix, homogenious_coordinates)
    return transformed_cloud[:3,:].T

if __name__ == '__main__':
    rospy.init_node('cloud_processing_node')
    tf_tree = tf.TransformListener()
    rospy.Subscriber('/sensor/side_camera/depth/color/points', PointCloud2, cloud_sub_callback)
    rospy.spin()
