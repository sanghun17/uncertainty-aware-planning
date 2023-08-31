#!/usr/bin/env python3
"""   
 Software License Agreement (BSD License)
 Copyright (c) 2023 Ulsan National Institute of Science and Technology (UNIST)
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice, this
 list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************** 
  @author: Hojin Lee <hojinlee@unist.ac.kr>, Sanghun Lee <sanghun17@unist.ac.kr>
  @date: September 1, 2023
  @copyright 2023 Ulsan National Institute of Science and Technology (UNIST)
  @brief: ROS node for Adaptive uncertainty aware naviation algorithm for offroad environment. 
  @details: (auto) data logging node
"""
from re import L
import rospy
import time
import threading
import os
import numpy as np
import math 
import torch


from std_msgs.msg import Bool, Float64, Header
from visualization_msgs.msg import MarkerArray

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from autorally_msgs.msg import chassisState, chassisCommand
from sensor_msgs.msg import Image, CameraInfo
from auc.common.pytypes import AUCModelData, VehicleState, SimData, CameraIntExt
from auc.common.file_utils import *
from auc.simulation.vehicle_model import VehicleModel
import rospkg
from collections import deque
from dynamic_reconfigure.server import Server
from auc.cfg import predictorDynConfig
import message_filters
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge
import tf 
rospack = rospkg.RosPack()
pkg_dir = rospack.get_path('auc')

class DataLogger:
    def __init__(self):     

        self.n_nodes = rospy.get_param('~n_nodes', default=10)
        self.t_horizon = rospy.get_param('~t_horizon', default=1.0)                           
        self.dt = self.t_horizon / (self.n_nodes-1)*1.0 # between node is num node -1    
        
        '''
        ############### Vehicle specific ############### 
        '''
        self.vehicle_state = VehicleState()
        self.vehicle_model = VehicleModel(dt = self.dt, N_node = self.n_nodes)
        self.vx_cmd = 1.0
        self.vx_max = 2.0
        self.vx_multiplier = 1.0
        self.delta_cmd = 0.0
        self.delta_max = 25*np.pi/180.0
        self.delta_multiplier = 1.0
        ####################### Vehicle Specific End ############### 

        '''
        ###############  Dataset related variables ###############
        '''
        self.sampled_cmd_list = None
        self.auc_dataset = []
        self.data_save = False
        self.save_buffer_length = 1000
        self.cum_data_save_length = 0
        ############### Dataset related variables End ########################
        

        '''
        ###############  Camera related variables ############### 
        '''        
        found_camera_to_baselink = False
        listener = tf.TransformListener()
        listener.setUsingDedicatedThread(True)  # Use dedicated thread for the listener

        robot_to_camera_matrix = None
        while not found_camera_to_baselink:                   
            try:
                listener.waitForTransform('/camera_link', '/base_link', rospy.Time(0), rospy.Duration(5.0))
                rospy.sleep(1.0)                 
                (trans, rot) = listener.lookupTransform('/camera_link', '/base_link', rospy.Time(0))
                robot_to_camera_matrix = tf.transformations.compose_matrix(translate=trans, angles=tf.transformations.euler_from_quaternion(rot))
                found_camera_to_baselink = True
                rospy.logwarn("camera_link to base_link Transformation lookup found.")
                
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):                
                rospy.logwarn("camera_link to base_link Transformation lookup failed.")
                
        camera_info_topic = "/camera_name/color/camera_info"  # Replace with your actual camera_info topic
        self.camera_info_msg = rospy.wait_for_message(camera_info_topic, CameraInfo)
        self.cam_int_ext = CameraIntExt()
        self.cam_int_ext.update_cam_int_ext_info(self.camera_info_msg, robot_to_camera_matrix)
        self.bridge = CvBridge()
        print("Camera Init Done")
        ############### Camera related variables End ########################

        
        self.ini_pub_and_sub()
        self.init_timer_callback()     

        rate = rospy.Rate(1)     
        while not rospy.is_shutdown():            
            msg = Bool()
            msg.data = True
            # self.status_pub.publish(msg)          
            rate.sleep()
        
    ''' 
        ###############  ROS pub and sub ###############  
    '''
    def ini_pub_and_sub(self):       
        
        self.dyn_srv = Server(predictorDynConfig, self.dyn_callback)
        delta_cmd_topic = "/delta_cmd"
        speed_cmd_topic = "/setpoint"
        vehicle_status_topic = "/chassisState"
        vehicle_gt_odom_topic = "/ground_truth/state"        
        # pub
        self.delta_pub = rospy.Publisher(delta_cmd_topic, Float64, queue_size=2)            
        self.speed_pub = rospy.Publisher(speed_cmd_topic, Float64, queue_size=2)            
        self.node_hz_pub = rospy.Publisher('data_logging_hz', Header, queue_size=1)
        self.node_hz = Header()        
        self.node_hz.stamp= rospy.Time.now()
        self.node_hz.seq = 0        
        # Subscribers     
        steer_cmd_sub = Subscriber('/delta_cmd', Float64)        
        vel_cmd_sub = Subscriber('/setpoint', Float64)                
        odom_sub = Subscriber('/ground_truth/state', Odometry)        
        depth_camera_sub = Subscriber('/camera_name/depth/image_raw', Image)
        color_camera_sub = Subscriber('/camera_name/color/image_raw', Image)        
        self.ts = ApproximateTimeSynchronizer(
        [steer_cmd_sub, vel_cmd_sub, odom_sub, depth_camera_sub, color_camera_sub],
        queue_size=10, slop=0.1, allow_headerless=True
        )
        self.ts.registerCallback(self.msg_filter_callback)        
        ############### ROS pub and sub  End ########################

    def init_timer_callback(self):
        ## controller callback
        mode = "vehicle model"
        if mode == "constant":
            self.cmd_hz = 5
            self.cmd_timer = rospy.Timer(rospy.Duration(1/self.cmd_hz), self.cmd_callback)         
            self.delta_hz = 10
            self.delta_timer = rospy.Timer(rospy.Duration(1/self.delta_hz), self.delta_callback)         
            self.speed_hz = 10
            self.speed_timer = rospy.Timer(rospy.Duration(1/self.speed_hz), self.speed_callback)

        if mode == "vehicle model":
            self.cmd_hz = 1/self.t_horizon
            self.cmd_timer_model = rospy.Timer(rospy.Duration(1/self.cmd_hz), self.cmd_sampling_model)
            self.delta_hz = 1/self.dt
            self.delta_timer = rospy.Timer(rospy.Duration(1/self.delta_hz), self.cmd_callback_model)         
            
            
        

    def dyn_callback(self,config,level):        
        self.data_save = config.logging_vehicle_states
        if config.clear_buffer:
            self.clear_buffer()        
        print("dyn reconfigured")
        
        return config
    
    def update_node_hz(self):
        self.node_hz.stamp= rospy.Time.now()
        self.node_hz.seq +=np.random.randint(5)+1
        if self.node_hz.seq > 1000:
            self.node_hz.seq = 0
        self.node_hz_pub.publish(self.node_hz)
                         
    def datalogging(self,cur_data):           
        if isinstance(self.vehicle_state.u.steer,float):
            self.auc_dataset.append(cur_data.copy())     
            self.cum_data_save_length+=1              
        if len(self.auc_dataset) > self.save_buffer_length:
            self.save_buffer_in_thread()
      
    def save_buffer_in_thread(self):
        # Create a new thread to run the save_buffer function
        t = threading.Thread(target=self.save_buffer)
        t.start()
    
    def clear_buffer(self):
        if len(self.auc_dataset) > 0:
            self.auc_dataset.clear()            
        rospy.loginfo("states buffer has been cleaned")

    def save_buffer(self):        
        real_data = SimData(len(self.auc_dataset), self.auc_dataset, self.cam_int_ext)        
        create_dir(path=train_dir)        
        pickle_write(real_data, os.path.join(train_dir, str(self.vehicle_state.header.stamp.to_sec()) + '_'+ str(len(self.auc_dataset))+'.pkl'))
        rospy.loginfo("states data saved")        
        self.clear_buffer()

    def preprocess_image_depth(self,depth_msg, color_msg):

        depth_image = np.copy(self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough"))
        replacement_distance = 100.0  # if nan -> maximum distance as 100
        # Find NaN values in the image
        depth_nan_indices = np.isnan(depth_image)
        # Replace NaN values with the replacement distance
        if len(depth_image[depth_nan_indices]) > 0:
            depth_image[depth_nan_indices] = replacement_distance

        color_image = np.copy(self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="passthrough"))
        replacement_color = [0, 0, 0]  # if non -> White
        # Find NaN values in the image
        nan_indices = np.isnan(color_image)
        # Replace NaN values with the replacement color
        if len(color_image[nan_indices]) > 0:
            color_image[nan_indices] = replacement_color

        return depth_image, color_image

    def msg_filter_callback(self,steer_cmd_msg,vel_cmd_msg,odom_msg,depth_msg,color_msg):
        print("self.cum_data_save_length = " + str(self.cum_data_save_length))
        self.vehicle_state.update_from_auc(steer_cmd_msg,vel_cmd_msg,odom_msg)
        cur_data = AUCModelData()
        depth_img, color_img = self.preprocess_image_depth(depth_msg, color_msg)        
        cur_data.update_from_auc(steer_cmd_msg,vel_cmd_msg,odom_msg,depth_img,color_img)
        if self.data_save:
            self.datalogging(cur_data)
          
    def cmd_callback(self,event):
        delta_cmd = Float64()
        delta_cmd.data = self.delta_cmd
        vx_cmd = Float64()
        vx_cmd.data =  self.vx_cmd     
        self.delta_pub.publish(delta_cmd)
        self.speed_pub.publish(vx_cmd)
    
    def cmd_sampling_model(self,event):
        nominal_preds, pred_u = self.vehicle_model.torch_kinematic_updates(self.vehicle_state)
        sample_idxs = torch.rand(pred_u.shape[0])*pred_u.shape[1] # sample idx for each time step 
        self.sampled_cmd_list = torch.zeros((self.n_nodes-1,2))
        for i in range(self.n_nodes-1):
            self.sampled_cmd_list[i,:]=pred_u[i,int(sample_idxs[i]),:]
        # print("new samples!")

    def cmd_callback_model(self,event):
        if self.sampled_cmd_list is not None:
            delta_cmd = self.sampled_cmd_list[0,1].item()
            vx_cmd = self.sampled_cmd_list[0,0].item()
            if self.sampled_cmd_list.shape[0] > 1:
                self.sampled_cmd_list=self.sampled_cmd_list[1:]
                # print("in of sample", vx_cmd, delta_cmd)
            else:
                # print("last of sample",vx_cmd, delta_cmd)
                pass
            self.delta_pub.publish(delta_cmd)
            self.speed_pub.publish(vx_cmd)

    def delta_callback(self,event):
        delta_in_degree = np.random.rand(1)*5
        self.delta_cmd += self.delta_multiplier*(delta_in_degree*np.pi/180.0)
        if abs(self.delta_cmd) > self.delta_max:
            self.delta_multiplier = -1*self.delta_multiplier

    def speed_callback(self,event):
        vx_sample = 0.1 + np.random.rand(1)*0.6
        self.vx_cmd +=self.vx_multiplier*vx_sample
        if abs(self.vx_cmd) > self.vx_max:
            self.vx_multiplier = -1*self.vx_multiplier
        if self.vx_cmd < 1.0+vx_sample:
            self.vx_multiplier = 1.0
        
###################################################################################

def main():
    rospy.init_node("data_logger")    
    DataLogger()

if __name__ == "__main__":
    main()




 
    


