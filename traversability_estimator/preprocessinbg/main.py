import rospy
import matplotlib.pyplot as plt
import matplotlib
import message_filters
from sensor_msgs.msg import PointCloud2, Image
from nav_msgs.msg import Odometry
from grid_map_msgs.msg import GridMap
from RobotPose import RobotPose
from Map import LidarMap, CameraMap, TraversabilityMap 

matplotlib.use('Qt5Agg')  # Set the backend to Qt5Agg

def callback(velodyne_msg, camera_msg, traversability_msg):
    # Your processing code here
    lidar_map.BEV_Projection(velodyne_msg)
    camera_map.UpdateImage(camera_msg)
    traversability_map.CalculateMap(traversability_msg)

if __name__ == '__main__':
    rospy.init_node('lidar_map_projection_node')
    plt.switch_backend('Qt5Agg')
    width, height, cell_size = 10, 10, 0.1

    data_dict = {}
    robot_pose = RobotPose() 
    camera_map = CameraMap(width, height, cell_size, data_dict)
    lidar_map = LidarMap(width, height, cell_size, camera_map, data_dict)
    traversability_map = TraversabilityMap(width, height, cell_size, data_dict)

    robot_x, robot_y, robot_yaw = robot_pose.GetPose()

    velodyne_sub = message_filters.Subscriber('/velodyne_points', PointCloud2)
    odometry_sub = message_filters.Subscriber('/ground_truth/state', Odometry)
    camera_sub = message_filters.Subscriber('/left_camera/image_raw', Image)
    traversability_sub = message_filters.Subscriber('/traversability_estimation/traversability_map', GridMap)
    rospy.Subscriber('/ground_truth/state', Odometry, robot_pose.update_pose)
    ts = message_filters.ApproximateTimeSynchronizer([velodyne_sub, camera_sub, traversability_sub], queue_size=10, slop=0.1, allow_headerless=True)
    ts.registerCallback(callback)
    
    rate = rospy.Rate(10)  # 10 Hz

    while not rospy.is_shutdown():
        # Your processing code here
        rate.sleep()
