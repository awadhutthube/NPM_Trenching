import rospy
import rosbag
import numpy as np
from tf import TransformListener
import ros_numpy
import tf

rospy.init_node('tf_listener')

tf_tree = TransformListener()

while not rospy.is_shutdown():
    try:
        (trans,rot) = tf_tree.lookupTransform('base_link', 'sensor/side_camera_depth_optical_frame', rospy.Time(0))
        print(trans, rot)
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        print("No TF data")
        break


translation = [-0.16927510039335014, -0.3717676310584948, -0.0332700755223688]
rotation = [0.9378418721923328, 0.1422734925170896, -0.04777733124311015, -0.31293482182246196]