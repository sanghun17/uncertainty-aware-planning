import tf.transformations
import tf
import os
import torch
import numpy as np

class RobotPose:    
    def __init__(self,data_dict,folder_name):
        self.state_list = None # 0:vx, 1:vy, 2:wz, 3:roll, 4: pitch
        self.action_list = None # 0: vx, 1: steering
        self.true_position_list= None # x,y,z,roll,pitch,yaw
        os.makedirs(folder_name, exist_ok=True)
        self.data_dict = data_dict
    
    def update_pose(self, state_list_msg):
        n = len(state_list_msg.odometry_list)
        self.state_list = np.zeros((n, 5))  # Initialize an array to store robot states
        self.true_position_list = np.zeros((n, 6))  # Initialize an array to store true positions
        
        for i in range(n):
            odometry_msg = state_list_msg.odometry_list[i]
            robot_state = (
                odometry_msg.twist.twist.linear.x,
                odometry_msg.twist.twist.linear.y,
                odometry_msg.twist.twist.angular.z, 
                odometry_msg.pose.pose.orientation.x,
                odometry_msg.pose.pose.orientation.y
            )
            robot_true_position = (
                odometry_msg.pose.pose.position.x,
                odometry_msg.pose.pose.position.y,
                odometry_msg.pose.pose.position.z,
                odometry_msg.pose.pose.orientation.x,
                odometry_msg.pose.pose.orientation.y,
                odometry_msg.pose.pose.orientation.z
            )
            self.state_list[i, :] = robot_state
            self.true_position_list[i, :] = robot_true_position
        
        self.data_dict['state'] = torch.tensor(self.state_list, dtype=torch.float32)
        self.data_dict['true_position'] = torch.tensor(self.true_position_list, dtype=torch.float32)
        
    def update_action(self,action_list_msg):
        n = len(action_list_msg.action_list.data) // 2
        self.action_list = np.array(action_list_msg.action_list.data).reshape((n, 2))
        self.data_dict['action']=torch.tensor(self.action_list,dtype=torch.float32)

    def GetStateAction(self):
        # print("GetPose: ",self.robot_yaw)
        return  self.robot_state , self.action