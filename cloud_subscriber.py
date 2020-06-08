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
import tf

# Non-blocking visualization of images
plt.ion()

# Creating publisher objects
pub1 = rospy.Publisher('/cloud_transformed', PointCloud2, queue_size=5)
pub2 = rospy.Publisher('/cloud_original', PointCloud2, queue_size=5)

# Initialization of constants and utility variables
idx = 0
tvec = [-0.16927510039335014, -0.3717676310584948, -0.0332700755223688]
quat = [0.9378418721923328, 0.1422734925170896, -0.04777733124311015, -0.31293482182246196]
    
def cloud_sub_callback(msg):
    global idx
    xyz = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg, remove_nans=True)
        
    # Transforming cloud into map frame and normalizing for 0 mean
    H = utils.get_transformation_matrix(tvec, quat)
    transformed_xyz = transform_cloud(xyz, H)
    mean_xyz = np.mean(transformed_xyz, axis = 0)
    transformed_xyz -= mean_xyz

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
    heightmap, trench_thresh = generate_heightmap(transformed_xyz.copy(), 0, mask = False)

    # Segmenting the trench using equations of lines passing through the wheel's frame
    line1, line2 = wb.fit_line(trans, euler_angles[1], idx)
    bool_array = wb.check_side(transformed_xyz.copy(), line1, line2)
    heightmap = utils.mask_heightmap(transformed_xyz, bool_array, heightmap)
    
    # Visualizing and saving the heightmaps
    utils.visualize_heightmap(heightmap, idx)        

    # Publishing the segmented cloud and logging essential data
    transformed_xyz = transformed_xyz[bool_array == 1]
    publish_transformed_cloud(transformed_xyz)
    idx += 1
    return

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
    header.frame_id = 'base_link'
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
    tf_tree = tf.TransformListener()
    rospy.Subscriber('/sensor/side_camera/depth/color/points', PointCloud2, cloud_sub_callback)
    rospy.spin()
