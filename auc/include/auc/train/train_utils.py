#!/usr/bin/env python3
import copy
import os
import pickle
import multiprocessing as mp
import time
import string
from dataclasses import dataclass, field
import random

import pynput
from typing import List, Tuple
import numpy as np

from auc.common.file_utils import * 
from auc.common.pytypes import VehicleState, AUCModelData, SimData, CameraIntExt
from auc.simulation.vehicle_model import VehicleModel
import torch 
import matplotlib.pyplot as plt
from auc.common.utils import b_to_g_rot, wrap_to_pi, get_attented_img_given_aucData
from torch.utils.data import Dataset, DataLoader

def normalize_depth(depth):
    depth = np.clip(depth,0.0, 20.0) 
    norm_depth = depth/20.0
    return  norm_depth

def normalize_color(img):
    norm_color = img/255
    return norm_color

def get_action_set(auc_data_list:List[AUCModelData]):
    action_set = np.array([[data.vehicle.u.vx, data.vehicle.u.steer] for data in auc_data_list])
    return action_set

def get_cur_vehicle_state_input(auc_data:AUCModelData):
    
    '''
    # we use local velocity and pose as vehicle current state         
    TODO: see if vz, wx, wy can be ignored and independent of estimation process
    '''
    return np.array([auc_data.vehicle.local_twist.linear.x,
                    auc_data.vehicle.local_twist.linear.y,
                    auc_data.vehicle.local_twist.linear.z,
                    auc_data.vehicle.local_twist.angular.x,
                    auc_data.vehicle.local_twist.angular.y,
                    auc_data.vehicle.local_twist.angular.z,
                    auc_data.vehicle.euler.pitch,
                    auc_data.vehicle.euler.roll])

    # [vx,vy,vz, wx,wy,wz,roll,pitch]

def get_cur_vehicle_state4_input(auc_data:AUCModelData):
    
    '''
    # we use local velocity and pose as vehicle current state         
    TODO: see if vz, wx, wy can be ignored and independent of estimation process
    '''
    return np.array([auc_data.vehicle.odom.pose.pose.position.x,
                    auc_data.vehicle.odom.pose.pose.position.y,
                    auc_data.vehicle.odom.pose.pose.position.z,
                    auc_data.vehicle.euler.yaw])
    

class AUCDataset(Dataset):    
    def __init__(self, input_d, output_d):        
        (states,pred_actions, colors, depths,states4)  = input_d 
        (pred_pose_residuals, pred_att_color_imgs, pred_att_depth_imgs) = output_d 

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.img_input = torch.tensor(img_input).to(self.device).float()
        # self.state_input = torch.tensor(state_input).to(self.device).float()
        # self.action_input = torch.tensor(action_input).to(self.device).float()
        # self.img_output = torch.tensor(img_output).to(self.device).float()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.states = states
        self.states4 = states4
        self.pred_actions = pred_actions
        self.colors = colors
        self.depths = depths
        self.pred_pose_residuals = pred_pose_residuals
        self.pred_att_color_imgs = pred_att_color_imgs
        self.pred_att_depth_imgs = pred_att_depth_imgs
        self.states_mean = None
        self.states_std = None
        self.pred_actions_mean = None
        self.pred_actions_std = None
        self.pred_pose_residuals_mean = None
        self.pred_pose_residuals_std = None
        self.pred_pose_residuals_max = None
        self.pred_pose_residuals_min = None
        self.data_noramlize(states, pred_actions, pred_pose_residuals)


        
        assert len(self.states) == len(self.pred_actions) , "All input and output must have the same length"
    
    def get_norm_stats(self):
        stat_dict = {
                'states_mean':self.states_mean,
                'states_std':self.states_std,
                'pred_actions_mean':self.pred_actions_mean,
                'pred_actions_std':self.pred_actions_std,
                'pred_pose_residuals_mean':self.pred_pose_residuals_mean,
                'pred_pose_residuals_std':self.pred_pose_residuals_std,
                'pred_pose_residuals_max':self.pred_pose_residuals_max,
                'pred_pose_residuals_min':self.pred_pose_residuals_min
        }

        return stat_dict

        

    def min_max_scaling(self,data,max,min):
        range_val = max-min+1e-10
        return (data-min)/range_val
        
    def normalize(self,data, mean, std):        
        normalized_data = (data - mean) / std
        return normalized_data
    
    def standardize(self, normalized_data, mean, std):        
        data = normalized_data*std+mean        
        return data
    
    def data_noramlize(self, states, pred_actions, pred_pose_residuals):
        self.states_mean, self.states_std = self.normalize_each(states)
        self.pred_actions_mean, self.pred_actions_std = self.normalize_each(pred_actions)
        self.pred_pose_residuals_mean, self.pred_pose_residuals_std = self.normalize_each(pred_pose_residuals)

        self.pred_pose_residuals_max, self.pred_pose_residuals_min = self.get_min_max(pred_pose_residuals)
        
    def get_min_max(self,x):
        stacked_tensor = torch.stack(x, dim=0)
        max_tensor = torch.max(stacked_tensor, dim=0)
        min_tensor = torch.min(stacked_tensor, dim=0)
        return max_tensor, min_tensor
        
    def normalize_each(self, x):
        stacked_tensor = torch.stack(x, dim=0)
        # Calculate mean and standard deviation along dimension 1
        mean_tensor = torch.mean(stacked_tensor, dim=0)
        std_tensor = torch.std(stacked_tensor, dim=0)
        return mean_tensor, std_tensor

        

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        assert self.states_mean is not None, 'input, output should be normlized'
        # states = self.normalize(self.states[idx], self.states_mean, self.states_std)        
        # pred_actions = self.normalize(self.pred_actions[idx], self.pred_actions_mean, self.pred_actions_std)                
        states = self.states[idx]
        pred_actions = self.pred_actions[idx]
        colors = self.colors[idx]
        depths = self.depths[idx]        
        
        # pred_pose_residuals = self.min_max_scaling(self.pred_pose_residuals[idx], self.pred_pose_residuals_max, self.pred_pose_residuals_min)
        pred_pose_residuals = self.normalize(self.pred_pose_residuals[idx], self.pred_pose_residuals_mean, self.pred_pose_residuals_std)        

        pred_att_color_imgs = self.pred_att_color_imgs[idx]
        pred_att_depth_imgs = self.pred_att_depth_imgs[idx]

        input_d = (states,pred_actions, colors, depths)  
        output_d = (pred_pose_residuals, pred_att_color_imgs, pred_att_depth_imgs) 

        return input_d, output_d
    
    def getitem_sh(self, idx):
        assert self.states_mean is not None, 'input, output should be normlized'
        # states = self.normalize(self.states[idx], self.states_mean, self.states_std)        
        # pred_actions = self.normalize(self.pred_actions[idx], self.pred_actions_mean, self.pred_actions_std)                
        states4 = self.states4[idx]
        states = self.states[idx]
        pred_actions = self.pred_actions[idx]
        colors = self.colors[idx]
        depths = self.depths[idx]        
        
        # pred_pose_residuals = self.min_max_scaling(self.pred_pose_residuals[idx], self.pred_pose_residuals_max, self.pred_pose_residuals_min)
        pred_pose_residuals = self.normalize(self.pred_pose_residuals[idx], self.pred_pose_residuals_mean, self.pred_pose_residuals_std)        

        pred_att_color_imgs = self.pred_att_color_imgs[idx]
        pred_att_depth_imgs = self.pred_att_depth_imgs[idx]

        data_dict = {}
        data_dict['state4']=states4
        data_dict['state'] = states
        data_dict['pred_actions'] = pred_actions
        data_dict['colors'] = colors
        data_dict['depths'] = depths
        data_dict['pred_pose_residuals'] = pred_pose_residuals
        data_dict['pred_att_color_imgs'] = pred_att_color_imgs
        data_dict['pred_att_depth_imgs'] = pred_att_depth_imgs

        return data_dict

class SampleGenerator():
    '''
    Class that reads simulation results from a folder and generates samples off that for model training. Can choose a
    function to determine whether a sample is useful or not.
    '''

    def __init__(self, abs_path, data_path= None, elect_function=None):
        '''
        abs path: List of absolute paths of directories containing files to be used for training
        randomize: boolean deciding whether samples should be returned in a random order or by time and file
        elect_function: decision function to choose samples
        init_all: boolean deciding whether all samples should be preloaded, can be set to False to reduce memory usage if
                        needed TODO not implemented yet!
        '''
        if elect_function is None:
            elect_function = self.useAll
        self.counter = 0
        self.abs_path = abs_path
        self.samples = []
        self.ego_check_time = []
        self.check_time_diff = []
        self.valid_check_time_diff = []        
        self.sequence_length = 10
        self.N = self.sequence_length
        self.dt = 0.2
        self.vehicle_simulator = VehicleModel(dt = 0.2)
        
        
        self.X_data = []
        self.Y_data = []


        self.plot_validation_result = True
        data_loaded = False

        if data_path is not None :            
           data_loaded = self.preprocessed_data_load(data_path)        
            
        if data_loaded is False:
            for ab_p in self.abs_path:
                for filename in os.listdir(ab_p):
                    if filename.endswith(".pkl"):
                        dbfile = open(os.path.join(ab_p, filename), 'rb')
                        # scenario_data: SimData = pickle.load(dbfile)
                        scenario_data: SimData = pickle.load(dbfile)
                        N = scenario_data.N
                        camera_info = scenario_data.camera_info
                        ## pass if the sequence is not enough 
                        if N < 12:
                            continue
                    
                        for i in range(N-self.sequence_length):         
                            
                            time_diff =scenario_data.auc_data[i+1].header.stamp.to_sec()-scenario_data.auc_data[i].header.stamp.to_sec()                        
                            if not self.validate_time_diff(time_diff):
                                continue

                            pred_actions = get_action_set(scenario_data.auc_data[i:i+self.N-1])
                            pred_pose, pred_pose_residual = self.get_pred_residual_pose(pred_actions, scenario_data.auc_data[i:i+self.N])
                            if not self.residual_validation(pred_pose, pred_pose_residual):
                                continue
                            
                            pred_att_color_imgs, pred_att_depth_imgs = get_attented_img_given_aucData(pred_pose, scenario_data.auc_data[i],camera_info)
                            cur_state = get_cur_vehicle_state_input(scenario_data.auc_data[i])
                            cur_state4=get_cur_vehicle_state4_input(scenario_data.auc_data[i])
                            
                            input_data = {}
                            input_data['state4'] = torch.tensor(cur_state4)
                            input_data['state'] = torch.tensor(cur_state)
                            input_data['pred_actions'] = torch.tensor(pred_actions)
                            color_img = np.transpose(scenario_data.auc_data[i].color,(2,0,1))
                            depth = scenario_data.auc_data[i].depth
                            depth_img = depth[np.newaxis,:]
                            
                            input_data['color'] = torch.tensor(normalize_color(color_img)).float()  
                            input_data['depth'] = torch.tensor(normalize_depth(depth_img)).float()                          

                            output_data= {}
                            output_data['pred_pose_residual'] = torch.tensor(pred_pose_residual).float()                        
                            output_data['pred_att_color_imgs'] = torch.tensor(normalize_color(pred_att_color_imgs)).float()                        
                          
                            output_data['pred_att_depth_imgs'] = torch.tensor(normalize_depth(pred_att_depth_imgs)).float()                        
                            # is_valid = self.data_validation(pred_actions,pred_pose_residual,pred_att_images)                        
                            # if is_valid:
                            if torch.max(output_data['pred_att_color_imgs']) > 1:
                                print(str(i) + " data pred_att_color_imgs is not normalized....")
                                continue

                            if self.detect_nan_from_data(output_data) or self.detect_nan_from_data(input_data):
                                print(str(i) + " NAN is included ..")
                                continue

                            self.X_data.append(input_data)
                            self.Y_data.append(output_data)

                        dbfile.close()
        
            if self.plot_validation_result:        
                self.plot_residual_list()
                self.plotTimeDiff()
            print('Generated Dataset with', len(self.X_data), 'samples!')
            self.preprocessed_data_save()
    
    def resample_image(self,image):

        return image
    def detect_nan_from_data(self, dict_obj):
        def has_nan(tensor):
            return torch.isnan(tensor).any().item()
        has_nan_values = any(has_nan(tensor) for tensor in dict_obj.values())
        return has_nan_values


    def get_train_data_at_once(self):
        state = [d['state'] for d in self.X_data]
        pred_action = [d['pred_actions'] for d in self.X_data]
        color = [d['color'] for d in self.X_data]        # height 480 width 640         
        depth = [d['depth'] for d in self.X_data]
        pred_pose_residual = [d['pred_pose_residual'] for d in self.Y_data]
        state = torch.stack(state)
        pred_action = torch.stack(pred_action)
        color = torch.stack(color)
        depth = torch.stack(depth)
        pred_pose_residual = torch.stack(pred_pose_residual)
        input_d = (state,pred_action, color, depth)
        output_d = pred_pose_residual
        return input_d, output_d

    def get_dataset(self,ratio = 0.8):
        
        state = [d['state'] for d in self.X_data]
        state4 = [d['state4'] for d in self.X_data]
        pred_action = [d['pred_actions'] for d in self.X_data]
        color = [d['color'] for d in self.X_data]        # height 480 width 640         
        depth = [d['depth'] for d in self.X_data]
        pred_pose_residual = [d['pred_pose_residual'] for d in self.Y_data]
        pred_att_color_img = [d['pred_att_color_imgs'] for d in self.Y_data]
        pred_att_depth_img = [d['pred_att_depth_imgs'] for d in self.Y_data]

        
        # ## [batch, sequence(optional), inputfeatures]
        # states = torch.vstack(state)
        # pred_actions = torch.stack(pred_action, axis=0)
        # ### [batch, channel, height, width]
        # colors = torch.stack(color, axis=0)        
        # colors = torch.permute(colors,[0,3,1,2])
        # depths = torch.stack(depth, axis=0).unsqueeze(dim=1)
        # pred_pose_residuals = torch.stack(pred_pose_residual,axis=0)
        # pred_att_color_imgs = torch.stack(pred_att_color_img,axis=0)
        # ## [batch, sequence(optional), channel(3), height, width]
        # pred_att_color_imgs = torch.permute(pred_att_color_imgs,[0,1,4,2,3])
        # ## [batch, sequence(optional), channel(1), height, width]
        # pred_att_depth_imgs = torch.stack(pred_att_depth_img,axis=0).unsqueeze(dim=2)
        # input_d = (states,pred_actions, colors, depths)
        # output_d = (pred_pose_residuals, pred_att_color_imgs, pred_att_depth_imgs)

        input_d = (state,pred_action, color, depth,state4)
        output_d = (pred_pose_residual, pred_att_color_img, pred_att_depth_img)
        auc_dataset = AUCDataset(input_d, output_d)
        return auc_dataset
        


    def preprocessed_data_load(self,path = None):
        if path is None:
            return False        
        loaded = torch.load(path)     
        self.X_data = loaded['input']
        self.Y_data = loaded['ouput']
        print('Loaded Dataset with', len(self.X_data), 'samples!')
        return True
    
    def preprocessed_data_save(self,data_dir = None):
        if data_dir is None:
            data_dir = preprocessed_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        file_path = os.path.join(data_dir, f'preprocessed_data_{len(self.X_data)}.pth')                
        # Create a checkpoint dictionary including both model state and args
        checkpoint = {
            'input': self.X_data,
            'ouput': self.Y_data
        }
        
        torch.save(checkpoint, file_path)
        print(f"preprocessed_data_ saved at epoch {str(file_path)}")





    def residual_validation(self,pred_pose,pred_psoe_residual):
        if not hasattr(self,'residual_list'):
            self.residual_list = []            
            self.valid_residual_list = []
            return        
        max_tmp = np.max(pred_psoe_residual[:,:], axis=0)
        self.residual_list.append(max_tmp)        
        is_valid = True        
        time_horizon = len(pred_psoe_residual[:,0])
        if np.max(abs(pred_psoe_residual[:,0])) > 0.1*time_horizon :
            is_valid = False
        if np.max(abs(pred_psoe_residual[:,1])) > 0.1*time_horizon:
            is_valid = False
        if np.max(abs(pred_psoe_residual[:,2])) > 0.02*time_horizon:
            is_valid = False
        if is_valid:
            self.valid_residual_list.append(max_tmp)        
        else:
            self.valid_residual_list.append(np.zeros(max_tmp.shape))
        return is_valid
        
    def plot_residual_list(self):
        if not hasattr(self,'residual_list'):            
            return       
        residual_list = np.array(self.residual_list)
        valid_residual_list = np.array(self.valid_residual_list)
      

        fig, axs = plt.subplots(4, 1, figsize=(10, 5 * 4))
        labels= ['deltaX', 'deltaY', 'deltaZ', 'deltaYaw']
        for i in range(4):
            axs[i].plot(residual_list[:, i], label=labels[i])
            axs[i].plot(valid_residual_list[:, i], label=labels[i])
            axs[i].set_title(f"Residuals for Array {i}")            
            axs[i].set_ylabel("Residual Value")
            axs[i].legend()

        plt.tight_layout()
        plt.show()
        
    def get_pred_residual_pose(self,pred_actions, auc_data_list:List[AUCModelData]):
        assert len(pred_actions[:,0]) + 1 <= len(auc_data_list), "Number of predicted actions should be one less than the number of AUC data points."
        cur_pose = auc_data_list[0].get_pose()
        roll_pose = cur_pose.copy()
        pred_poses = []
        residual_poses = []        
        gt_next_poses = []
        
        for i in range(len(pred_actions)):            
            roll_pose = self.vehicle_simulator.np_kinematic_updates(pred_actions[i,:],roll_pose)
            gt_next_pose = auc_data_list[i+1].get_pose()            
            residual_pose = gt_next_pose - roll_pose
            residual_pose[-1] = wrap_to_pi(residual_pose[-1])
            gt_next_poses.append(gt_next_pose.copy())
            pred_poses.append(roll_pose.copy())
            residual_poses.append(residual_pose.copy())
        pred_poses = np.array(pred_poses)
        residual_poses = np.array(residual_poses)
        gt_next_poses = np.array(gt_next_poses)
        

        return pred_poses, residual_poses
        
    


        

    def validate_time_diff(self,time_diff):
        is_valid = False
        self.check_time_diff.append(time_diff)
        if time_diff < self.dt*1.2 and time_diff > self.dt*0.8:
            self.valid_check_time_diff.append(time_diff)                            
            is_valid = True
        else:
            self.valid_check_time_diff.append(0.0)                            
            is_valid = False

        return is_valid
    
    def plotTimeDiff(self):
        all_time_diff = np.array(self.check_time_diff)
        valid_time_diff = np.array(self.valid_check_time_diff)
        plt.plot(all_time_diff)
        plt.plot(valid_time_diff,'*')
        plt.show()
    # def interpolation_with_time(self,scenario_data):

    def reset(self, randomize=False):
        if randomize:
            random.shuffle(self.samples)
        self.counter = 0

    def getNumSamples(self):
        return len(self.X_data)

    def nextSample(self):
        self.counter += 1
        if self.counter >= len(self.samples):
            print('All samples returned. To reset, call SampleGenerator.reset(randomize)')
            return None
        else:
            return self.samples[self.counter - 1]

    def plotStatistics(self, param):
        data_list = []
        if param == 'c':
            for i in self.samples:
                data_list.append(i.input[1].lookahead.curvature[0])
        fig, axs = plt.subplots(1, 1)
        axs.hist(data_list, bins=50)
        plt.show()

  
    def useAll(self, ego_state, tar_state):
        return True

