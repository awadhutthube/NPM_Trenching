
import rosbag
import numpy as np
from sensor_msgs.msg import PointCloud2
import ros_numpy
# import open3d as o3d
import rospy
import sensor_msgs.point_cloud2 as pcl2
import std_msgs
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
plt.ion()

pub1 = rospy.Publisher('/cloud_transformed', PointCloud2, queue_size=5)
pub2 = rospy.Publisher('/cloud_original', PointCloud2, queue_size=5)

tvec = [-0.16927510039335014, -0.3717676310584948, -0.0332700755223688+1]
quat = [0.9378418721923328, 0.1422734925170896, -0.04777733124311015, -0.31293482182246196]

def get_transformation_matrix(tvec, quat):
    transformation_matrix = np.eye(4)
    r = R.from_quat(quat)
    transformation_matrix[:3,:3] = r.as_dcm()
    transformation_matrix[:3,-1] = tvec
    return transformation_matrix

def plot_histogram(datapoints, number, start = -0.03, end = 0.08, interval_size = 0.0005):
    num = (end-start)//interval_size
    bin_sequence = np.linspace(start, end, num)
    axes = plt.gca()
    axes.set_ylim([0,7000])
    axes.set_xlim([-0.03,0.08])
    plt.hist(datapoints, bin_sequence)
    plt.savefig('../histograms/bag_1/frame_' + str(number))
    plt.show()
    plt.pause(0.005)
    plt.clf()

def read_rosbag(filepath):
    bag = rosbag.Bag(bagfile_path, 'r')
    for idx, (topic, msg, t) in enumerate(bag.read_messages(topics=['/sensor/side_camera/depth/color/points'])):
        xyz = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg, remove_nans=True)
        H = get_transformation_matrix(tvec, quat)
        transformed_xyz = transform_cloud(xyz, H)
        mean_xyz = np.mean(transformed_xyz, axis = 0)
        transformed_xyz -= mean_xyz
        # plot_histogram(transformed_xyz[:,2], idx)
        heightmap = generate_heightmap(transformed_xyz)
        # visualize_heightmap(heightmap)
        # publish_transformed_cloud(transformed_xyz)
        # publish_transformed_cloud(xyz, False)
        print(idx)
    return

def generate_heightmap(points, dim = 500):
    heightmap = np.zeros((dim,dim)).astype('float')
    x_co = (points[:,0]*1000).round().astype('int')
    y_co = (points[:,1]*1000).round().astype('int')
    z_co = points[:,2]
    x_co -= np.amin(x_co)
    y_co -= np.amin(y_co)
    z_co -= np.amin(z_co)
    print(z_co)
    heightmap[x_co,y_co] = z_co
    return heightmap

def visualize_heightmap(heightmap):
    plt.figure(1)
    plt.clf()
    plt.imshow(heightmap, cmap = 'gray')
    plt.pause(0.0002)
    return

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
    bagfile_path = '../test_data/02_K10MINI_2020-05-14-16-31-57.bag'
    # bagfile_path = '../test_data/03_K10MINI_2020-05-14-16-44-14.bag'
    # bagfile_path = '../test_data/04_K10MINI_2020-05-14-16-49-59.bag'
    read_rosbag(bagfile_path)
    print("Processing Complete")