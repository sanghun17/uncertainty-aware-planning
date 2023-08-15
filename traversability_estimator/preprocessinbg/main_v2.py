import rospy
from sensor_msgs.msg import PointCloud2, Image
from std_msgs.msg import Float32MultiArray
from RobotPose import RobotPose
from hmcl_msgs.msg import OdometryList, ActionList
import time
import os
import torch
from functools import partial
from ImageSaver import RGBSaver, DepthSaver
import cv2
import message_filters

def logging_callback(event, folder_name, data_dict):
    # Check if data_dict contains the expected keys
    expected_keys = ['state', 'action', 'RGB_image','depth_image']  # Replace with the actual keys
    missing_keys = [key for key in expected_keys if key not in data_dict]
    if missing_keys:
        print(f"Error: Missing keys in data_dict: {missing_keys}")
        return
    current_time = time.strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(folder_name, f'{current_time}.pt')
    torch.save(data_dict, file_path)
    print("dataset saved! :", file_path)

def callback(velodyne_msg, camera_msg, state_list_msg, action_list_msg):
    lidar_saver.DepthGenerator(velodyne_msg)
    camera_saver.UpdateImage(camera_msg)
    robot_pose.update_pose(state_list_msg)
    robot_pose.update_action(action_list_msg)

if __name__ == '__main__':
    rospy.init_node('camera_lidar_saver_node')

    current_time = time.strftime("%Y%m%d_%H%M%S")
    folder_name = '../input/data/' + current_time
    os.makedirs(folder_name, exist_ok=True)
    
    data_dict = {}
    camera_saver = RGBSaver(data_dict)
    lidar_saver = DepthSaver(camera_saver, data_dict)
    robot_pose = RobotPose(data_dict, folder_name)

    velodyne_sub = message_filters.Subscriber('/velodyne_points', PointCloud2)
    state_sub = message_filters.Subscriber('/state_sh', OdometryList)
    action_sub = message_filters.Subscriber('/action_sh', ActionList)
    camera_sub = message_filters.Subscriber('/left_camera/image_raw', Image)
    
    ts = message_filters.ApproximateTimeSynchronizer(
        [velodyne_sub, camera_sub, state_sub, action_sub],
        queue_size=10, slop=0.1, allow_headerless=True
    )
    ts.registerCallback(callback)

    callback_with_args = partial(logging_callback, folder_name=folder_name, data_dict=data_dict)
    logging_timer = rospy.Timer(rospy.Duration(0.5), callback_with_args)

    rate = rospy.Rate(20)  # 10 Hz

    while not rospy.is_shutdown():
        # lidar_saver.show()
        # camera_saver.show()
        # key = cv2.waitKey(1)  # Wait for a key event (1 millisecond)
        # if key == 27:  # Press 'Esc' key to exit the loop
        #     break
        rate.sleep()
    # cv2.destroyAllWindows()
