import rospy
import matplotlib.pyplot as plt
import matplotlib
import message_filters
from sensor_msgs.msg import PointCloud2, Image
from nav_msgs.msg import Odometry
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray
from RobotPose import RobotPose
from hmcl_msgs.msg import OdometryList
from Map import LidarMap, CameraMap, TraversabilityMap 
import time
import os
import torch
from functools import partial

matplotlib.use('Qt5Agg')  # Set the backend to Qt5Agg

def callback(velodyne_msg, camera_msg, traversability_msg, state_list_msg, action_list_msg):
    # Your processing code here
    lidar_map.BEV_Projection(velodyne_msg)
    camera_map.UpdateImage(camera_msg)
    traversability_map.CalculateMap(traversability_msg)
    robot_pose.update_pose(state_list_msg)
    robot_pose.update_action(action_list_msg)

def logging_callback(event,folder_name,data_dict):
    current_time = time.strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(folder_name, f'{current_time}.pt')
    torch.save(data_dict, file_path)
    print("dataset saved! :",file_path)

if __name__ == '__main__':
    rospy.init_node('lidar_map_projection_node')
    plt.switch_backend('Qt5Agg')

    current_time = time.strftime("%Y%m%d_%H%M%S")
    folder_name = '../input/data/' + current_time
    os.makedirs(folder_name, exist_ok=True)
    
    width, height, cell_size = 10, 10, 0.5

    data_dict = {}
    traversability_map = TraversabilityMap(width, height, cell_size, data_dict)
    camera_map = CameraMap(width, height, cell_size, data_dict)
    lidar_map = LidarMap(width, height, cell_size, camera_map, data_dict)
    robot_pose = RobotPose(data_dict, folder_name) 
    
    # robot_x, robot_y, robot_yaw = robot_pose.GetPose()

    velodyne_sub = message_filters.Subscriber('/velodyne_points', PointCloud2)
    state_sub = message_filters.Subscriber('/state_sh', OdometryList)
    action_sub = message_filters.Subscriber('/action_sh', Float32MultiArray)
    camera_sub = message_filters.Subscriber('/left_camera/image_raw', Image)
    traversability_sub = message_filters.Subscriber('/traversability_estimation/traversability_map', GridMap)
    # rospy.Subscriber('/ground_truth/state', Odometry, robot_pose.update_pose)
    
    ts = message_filters.ApproximateTimeSynchronizer(
        [velodyne_sub, camera_sub, traversability_sub, state_sub, action_sub],
        queue_size=10, slop=0.1, allow_headerless=True
    )
    ts.registerCallback(callback)

    callback_with_args = partial(logging_callback, folder_name=folder_name, data_dict=data_dict)
    logging_timer = rospy.Timer(rospy.Duration(0.1), callback_with_args)
    
    rate = rospy.Rate(20)  # 10 Hz

    while not rospy.is_shutdown():
        rate.sleep()
