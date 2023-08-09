import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from grid_map_msgs.msg import GridMap

import matplotlib.pyplot as plt
import matplotlib

from RobotPose import RobotPose
from Map import LidarMap, CameraMap, TraversabilityMap 

matplotlib.use('Qt5Agg')  # Set the backend to Qt5Agg

if __name__ == '__main__':
    rospy.init_node('lidar_map_projection_node')
    plt.switch_backend('Qt5Agg')
    width, height, cell_size = 10, 10, 0.5 

    data_dict = {}
    robot_pose = RobotPose() 
    camera_map = CameraMap(width, height, cell_size)
    lidar_map = LidarMap(width, height, cell_size, camera_map)
    traversability_map = TraversabilityMap(width, height, cell_size)

    robot_x, robot_y, robot_yaw = robot_pose.GetPose()
    
    rospy.Subscriber('/velodyne_points', PointCloud2, lidar_map.BEV_Projection) 
    rospy.Subscriber('/ground_truth/state', Odometry, robot_pose.update_pose)
    rospy.Subscriber('/left_camera/image_raw', Image, camera_map.UpdateImage)
    rospy.Subscriber('/traversability_estimation/traversability_map', GridMap, traversability_map.CalculateMap)
    

    # lidar_map.show()
    # camera_map.show()
    traversability_map.show( )

    rospy.spin()  # This line won't be reached because rospy.spin() blocks the code execution
