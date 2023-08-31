#!/usr/bin/env python
"""   
 Software License Agreement (BSD License)
 Copyright (c) 2022 Ulsan National Institute of Science and Technology (UNIST)
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
  @author: Hojin Lee <hojinlee@unist.ac.kr>
  @date: September 10, 2022
  @copyright 2022 Ulsan National Institute of Science and Technology (UNIST)
  @brief: ROS node for learning based Uncertainty-aware path planning module.
  @details: uncertainty-aware path planning module  
"""

from re import L
import rospy
import time
import threading
import numpy as np
import math 
import torch
from std_msgs.msg import Bool, Float64, Header
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from hmcl_msgs.msg import Waypoints, vehicleCmd
from visualization_msgs.msg import MarkerArray
from autorally_msgs.msg import chassisState
from grid_map_msgs.msg import GridMap
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from auc.simulation.vehicle_model import VehicleModel
# from gptrajpredict.vehicle_model_batch import VehicleModel
from auc.common.utils_batch import find_closest_idx_from_path, predicted_trajs_visualize, multi_predicted_distribution_traj_visualize,  get_odom_euler, get_local_vel, predicted_trj_visualize, wrap_to_pi, path_to_waypoints, elips_predicted_path_distribution_traj_visualize
# from gptrajpredict.gp_model import GPModel
from auc.map.gpgridmap_batch import GPGridMap

from dynamic_reconfigure.server import Server

from auc.cfg import predictorDynConfig
from auc.common.pytypes import AUCModelData, VehicleState, SimData, CameraIntExt
from auc.common.file_utils import *
from sensor_msgs.msg import Image, CameraInfo
from auc.AUC_estimate import AUCEStimator
import rospkg


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

class AUCPlanner:
    def __init__(self):        
        
        self.prediction_hz = rospy.get_param('~prediction_hz', default=3.0)
        self.n_nodes = rospy.get_param('~n_nodes', default=10)
        self.t_horizon = rospy.get_param('~t_horizon', default=2.0)           
        self.n_sample = rospy.get_param('~n_sample', default= 20)                           
        self.input_random = rospy.get_param('~input_random', default=False)                              
        self.dt = self.t_horizon / self.n_nodes*1.0        
         # x, y, psi, vx, vy, wz, z ,  roll, pitch 
         # 0  1  2     3  4   5   6    7, 8   
        self.bridge = CvBridge()
        self.cur_state = VehicleState()
        self.local_map = GPGridMap(dt = self.dt)
        self.auc_model = AUCEStimator(dt = self.dt, N_node = self.n_nodes, model_path='singl_aucgp_snapshot.pth')

        self.vehicle_model = VehicleModel(dt = self.dt, N_node = self.n_nodes)

        self.goal_pose = None
        self.prev_path = None
        self.local_traj_msg = None
        self.prev_local_traj_msg = None
        self.local_traj_max_pathlength = 15
               # Subscribers
        self.dyn_srv = Server(predictorDynConfig, self.dyn_callback)

        self.goal_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)        
        # self.vehicle_status_sub = rospy.Subscriber('/chassisState', chassisState, self.vehicle_status_callback)                        

        steer_cmd_sub = Subscriber('/delta_cmd', Float64)        
        vel_cmd_sub = Subscriber('/setpoint', Float64)          
        odom_sub = Subscriber('/ground_truth/state', Odometry)        
        depth_camera_sub = Subscriber('/camera_name/depth/image_raw', Image)
        color_camera_sub = Subscriber('/camera_name/color/image_raw', Image)                
        self.ts = ApproximateTimeSynchronizer(
        [steer_cmd_sub, vel_cmd_sub, odom_sub, depth_camera_sub, color_camera_sub],
        queue_size=10, slop=0.2, allow_headerless=True
        )
        self.ts.registerCallback(self.msg_filter_callback)        
        
                
        self.odom_available   = False 
        self.vehicle_status_available = False 
        self.waypoint_available = False 
        self.map_available = False
        
        # Thread for optimization
        self.vehicleCmd = vehicleCmd()
        self._thread = threading.Thread()        
        self.chassisState = chassisState()
        self.odom = Odometry()
        self.waypoint = PoseStamped()
        
        
        # Real world setup
        
        status_topic = "/is_data_busy"        
        sample_pred_traj_topic_name = "/sample_pred_trajectory"
        nominal_pred_traj_topic_name = "/nominal_pred_trajectory" 
        mean_pred_traj_topic_name = "/gpmean_pred_trajectory"         
        best_pred_traj_topic_name = "/best_gplogger_pred_trajectory" 
        local_traj_topic_name = "/local_traj"        
        # Publishers        
        self.local_traj_pub = rospy.Publisher(local_traj_topic_name, Waypoints, queue_size=2)        
        self.sample_predicted_trj_publisher = rospy.Publisher(sample_pred_traj_topic_name, MarkerArray, queue_size=2)        
        self.mean_predicted_trj_publisher    = rospy.Publisher(mean_pred_traj_topic_name, MarkerArray, queue_size=2)    
        self.nominal_predicted_trj_publisher = rospy.Publisher(nominal_pred_traj_topic_name, MarkerArray, queue_size=2)    
        self.best_predicted_trj_publisher = rospy.Publisher(best_pred_traj_topic_name, MarkerArray, queue_size=2)    
        self.status_pub = rospy.Publisher(status_topic, Bool, queue_size=2)    
        self.nominal_predicted_trj_publisher = rospy.Publisher(nominal_pred_traj_topic_name, MarkerArray, queue_size=2)    
        self.mean_predicted_trj_publisher    = rospy.Publisher(mean_pred_traj_topic_name, MarkerArray, queue_size=2)    
        
 

        self.local_map_sub = rospy.Subscriber('/traversability_estimation/global_map', GridMap, self.gridmap_callback)
               
        
        self.path_planner_timer = rospy.Timer(rospy.Duration(1/self.prediction_hz), self.pathplanning_callback)         
        rate = rospy.Rate(1)     
        while not rospy.is_shutdown():            
            msg = Bool()
            msg.data = True
            self.status_pub.publish(msg)
            rate.sleep()
        
    
    def msg_filter_callback(self,steer_cmd_msg: Float64,vel_cmd_msg: Float64,odom_msg: Odometry, depth_msg: Image,color_msg: Image):
        start_time = time.time()

        self.cur_state.update_from_auc(steer_cmd_msg,vel_cmd_msg,odom_msg)
        nominal_preds, pred_u = self.vehicle_model.torch_kinematic_updates(self.cur_state)
        print(nominal_preds)

        cur_data = AUCModelData()
        cur_data.update_from_rosmsgs( self.bridge, steer_cmd_msg,vel_cmd_msg,odom_msg,depth_msg,color_msg)
        ## get unstandardized predicted mean and std from the learning model
        pred_residual_mean, pred_residual_std = self.auc_model.pred(cur_data, pred_u)
        
        ## stack time horizon dim
        batch_size = pred_u.shape[1]
        pred_residual_mean = pred_residual_mean.view(batch_size,-1,2)
        pred_residual_std = pred_residual_std.view(batch_size,-1,2)
        
        ## compute the model error and local map error 
        best_path, total_trav_costs = self.local_map.compute_best_path(nominal_preds, pred_residual_mean, pred_residual_std, self.goal_pose)
        
        # elip_pred_traj_marker = elips_predicted_path_distribution_traj_visualize(x_pose_means_set,x_pose_vars_set,y_pose_means_set,y_pose_vars_set,nominal_predictedStates[:,0,:,:],sample_path_color)        
        nominal_path_color = [0,1,0,1]        
        nominal_pred_traj_marker = predicted_trajs_visualize(nominal_preds[:,:,:2],nominal_path_color)
        
        gpmean_path_color = [0,0,1,1]
        pred_pose_mean_seq = (nominal_preds[:,:,:2]+ pred_residual_mean[:,:,:2])
        mean_pred_traj_marker = predicted_trajs_visualize(pred_pose_mean_seq[:,:,:2],gpmean_path_color)
        self.mean_predicted_trj_publisher.publish(mean_pred_traj_marker)    
        self.nominal_predicted_trj_publisher.publish(nominal_pred_traj_marker) 

        # Your code or function calls here

        # Record the end time
        end_time = time.time()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time

        print(f"Elapsed time: {elapsed_time:.6f} seconds")

    def dyn_callback(self,config,level):                
        # if config.set_goal:
        #     cur_goal = PoseStamped()            
        #     cur_goal.pose.position.x = config.goal_x
        #     cur_goal.pose.position.y = config.goal_y            
        #     self.goal_pub.publish(cur_goal)

        scale_data = {"dist_heuristic_cost_scale":  config.dist_heuristic_cost_scale,                         
                        "model_error_weight": config.model_error_weight, 
                        "local_map_cost_weight": config.local_map_cost_weight}        
        self.local_map.set_scales(scale_data)
        
        return config

    def goal_callback(self,msg):        
        self.goal_pose = [msg.pose.position.x, msg.pose.position.y]

    def gridmap_callback(self,msg):                     
        if self.map_available is False:
            self.map_available = True
        self.local_map.set_map(msg)
                
    def ctrl_callback(self,msg):
        self.vehicleCmd = msg

    def vehicle_status_callback(self,data):
        if self.vehicle_status_available is False:
            self.vehicle_status_available = True
        self.chassisState = data
        self.steering = -data.steering*25*math.pi/180
        
    def odom_callback(self, msg):
        """                
        :type msg: PoseStamped
        """              
        if self.odom_available is False:
            self.odom_available = True 
        self.odom = msg        
        
    def waypoint_callback(self, msg):
        if self.waypoint_available is False:
            self.waypoint_available = True
        self.waypoint = msg

 
    def pathplanning_callback(self,timer):        
        return
        
        


###################################################################################
def main():
    rospy.init_node("auc_planner")    
    AUCPlanner()
if __name__ == "__main__":
    main()
