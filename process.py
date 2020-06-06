import rosbag
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

plt.ion()
pub1 = rospy.Publisher('/cloud_transformed', PointCloud2, queue_size=5)
pub2 = rospy.Publisher('/cloud_original', PointCloud2, queue_size=5)

tvec = [-0.16927510039335014, -0.3717676310584948, -0.0332700755223688+1]
quat = [0.9378418721923328, 0.1422734925170896, -0.04777733124311015, -0.31293482182246196]


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

def read_rosbag(filepath):
    bag = rosbag.Bag(bagfile_path, 'r')
    
    for idx, (topic, msg, t) in enumerate(bag.read_messages(topics=['/sensor/side_camera/depth/color/points'])):
        xyz = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg, remove_nans=True)
        
        # Transforming cloud into map frame and normalizing for 0 mean
        H = utils.get_transformation_matrix(tvec, quat)
        transformed_xyz = transform_cloud(xyz, H)
        mean_xyz = np.mean(transformed_xyz, axis = 0)
        transformed_xyz -= mean_xyz

        # Generating a histogram for the depth of points
        data, bins = generate_histogram(transformed_xyz[:,2], idx, draw = False)
        
        # Estimating a rough threshold based on the histogram
        trench_thresh = get_trench_threshold(data, bins)

        # Segmenting the wheel from raw cloud
        bbox, heightmap = wb.segment_wheel(transformed_xyz.copy(), trench_thresh)
        heightmap = wb.visualize_wheel_segment(bbox, heightmap)
        # utils.visualize_heightmap(heightmap, idx)    

        # cv2.circle(heightmap,(bbox[0][0], bbox[0][1]), 10, (255), -1)
        # cv2.circle(heightmap,(bbox[1][0], bbox[1][1]), 10, (255), -1)
        # cv2.imshow('test', heightmap)
        # cv2.waitKey(1)

        # draw_circles(bbox, heightmap)



        bool_array = wb.check_side(transformed_xyz.copy(), bbox)


        # Generating a heightmap and masking based on estimated threshold
        heightmap, trench_thresh = generate_heightmap(transformed_xyz, trench_thresh, mask = False)
        heightmap = utils.mask_heightmap(transformed_xyz, bool_array, heightmap)
        utils.visualize_heightmap(heightmap, idx)        

        transformed_xyz = transformed_xyz[bool_array < 0]
        # Publishing the transformed cloud and logging essential data
        publish_transformed_cloud(transformed_xyz)
        utils.log_data(idx, trench_thresh)

    return

def draw_circles(bbox, heightmap):
    plt.figure(2)
    plt.clf()
    plt.imshow(heightmap, cmap = 'gray')
    plt.scatter(bbox[0][0], bbox[0][1], color = 'g')
    plt.scatter(bbox[1][0], bbox[1][1], color = 'g')
    plt.plot((bbox[0][0], bbox[1][0]), (bbox[0][1], bbox[1][1]), '-')
    plt.pause(0.2)

def generate_heightmap(points, threshold, dim = 500, mask = False):
    heightmap = np.zeros((dim,dim,3)).astype('float')
    x_co, y_co, z_co = utils.discretize(points)
    x_co -= np.amin(x_co)
    y_co -= np.amin(y_co)
    min_val = np.amin(z_co)
    z_co -= min_val
    threshold -= min_val
    utils.publish_threshold_frame(threshold)
    heightmap[x_co,y_co,0] = z_co*20
    heightmap[x_co,y_co,1] = z_co*20
    heightmap[x_co,y_co,2] = z_co*20
    if mask:
        masked_x_co = x_co[z_co < threshold]
        masked_y_co = y_co[z_co < threshold]
        heightmap[masked_x_co, masked_y_co, 1] = 1    
    return heightmap, threshold

def publish_transformed_cloud(cloud_array, flag = True):
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = 'map'
    if not flag:
        header.frame_id = 'sensor/side_camera_depth_optical_frame'
        cloud = pcl2.create_cloud_xyz32(header,cloud_array)
        pub2.publish(cloud)
    else:
        cloud = pcl2.create_cloud_xyz32(header,cloud_array)
        pub1.publish(cloud)
    return

def transform_cloud(cloud_array, transformation_matrix = np.eye(4)):
    if cloud_array.shape[0] > cloud_array.shape[1]:
        homogenious_coordinates = np.vstack((cloud_array.T, np.ones(cloud_array.shape[0])))
    else:
        homogenious_coordinates = np.vstack((cloud_array, np.ones(cloud_array.shape[1])))

    transformed_cloud = np.matmul(transformation_matrix, homogenious_coordinates)
    return transformed_cloud[:3,:].T

if __name__ == '__main__':
    rospy.init_node('cloud_processing_node')
    # bagfile_path = '../test_data/01_K10MINI_2020-05-14-16-23-00.bag'
    # bagfile_path = '../test_data/02_K10MINI_2020-05-14-16-31-57.bag'
    # bagfile_path = '../test_data/03_K10MINI_2020-05-14-16-44-14.bag'
    # bagfile_path = '../test_data/04_K10MINI_2020-05-14-16-49-59.bag'
    read_rosbag(bagfile_path)
    plt.ioff()
    print("Processing Complete")