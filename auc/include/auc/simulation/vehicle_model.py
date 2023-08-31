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
  @brief: 3D Vehicle model using torch
"""

import numpy as np
import math
from auc.common.utils import b_to_g_rot, wrap_to_pi, wrap_to_pi_torch
from auc.common.pytypes import VehicleState,VehicleCommand
import torch
from filterpy.kalman import UnscentedKalmanFilter,MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
from auc.gp_encoder.gp_encoderModel import GPAUC

class VehicleModel:
    def __init__(self, dt = 0.2, N_node = 10, n_sample = 100):        
        
        self.N = N_node
        self.m = 25
        self.width = 0.45        
        self.L = 0.9
        self.Lr = 0.45
        self.Lf = 0.45
        self.Caf = self.m * self.Lf/self.L * 0.5 * 0.35 * 180/torch.pi
        self.Car = self.m * self.Lr/self.L * 0.5 * 0.35 * 180/torch.pi
        self.Izz = self.Lf*self.Lr*self.m
        self.g= 9.814195
        self.h = 0.15
        self.dt = dt 
        self.delta_rate_max = 50*math.pi/180.0 # rad/s
        self.rollover_cost_scale = 10.0        
        self.Q = torch.tensor([0.5, 0.5, 0.5])           
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ## kin_state = [x,y,z, yaw, vx]
        self.state_dim = 4
        ## inputs = [vx, delta]
        self.input_dim = 2        
        self.horizon = self.dt* N_node        
        # delta and ax
        self.steering_abs_max_in_radian = 25*torch.pi/180.0
        self.vx_abs_max = 3.0
        self.vx_sampling_num = 5
        self.steer_sampling_num = 5
        self.u_min = torch.tensor([ -self.vx_abs_max, -self.steering_abs_max_in_radian]).to(device= self.device) ## vx steering
        self.u_max = self.u_min*-1
        

                ## sampled accel and delta for motion primitivies         
        vx_set = torch.linspace(0.0,self.vx_abs_max,self.vx_sampling_num).to(device=self.device)        
        delta_set = torch.linspace(-self.steering_abs_max_in_radian,self.steering_abs_max_in_radian,self.steer_sampling_num).to(device=self.device)        
        self.usets = torch.hstack([torch.meshgrid(vx_set,delta_set)[0].reshape(-1,1),torch.meshgrid(vx_set, delta_set)[1].reshape(-1,1)])
        # self.usets = self.uset.repeat(self.gp_n_sample,1)
        ###### Add randomness to input sets #####
        # if self.input_random:
        #     random_add = torch.zeros(self.usets.shape).to(device=self.device)
        #     # random_add = self.usets +torch.rand(self.usets.shape).to(device=self.device)
        #     random_add[:,0] = self.usets[:,0] + (torch.rand(self.usets[:,0].shape).to(device=self.device)-0.5)*self.max_delta
        #     random_add[:,1] = self.usets[:,1] + (torch.rand(self.usets[:,1].shape).to(device=self.device)-0.5)*self.max_accel            
        #     random_add[:,0] = torch.clamp(random_add[:,0],-1*self.max_delta,self.max_delta)
        #     random_add[:,1] = torch.clamp(random_add[:,1],self.max_dccel,self.max_accel)
        #     self.usets = random_add
        #########################################
        self.usets_backup = torch.tensor(self.usets).clone()


        self.cur_state = VehicleState()

    def set_state(self,state : VehicleState):
        self.cur_state = state
    
    def set_state4(self,state):
        self.cur_state4 = state

    def propogate_z(self,z_in):        
        ## just direct pass for 2d case
        return z_in
        
    def torch_kinematic_updates(self, state =None):
        
        x = state.odom.pose.pose.position.x
        y = state.odom.pose.pose.position.y
        z = state.odom.pose.pose.position.z
        yaw = state.euler.yaw
        yaw = wrap_to_pi(yaw)            
       
        
        # uset = [batch , input_dim(delta, vx)]
        cur_stat = torch.tensor([x,y,z,yaw]).to(self.device)        
        uset = self.usets.clone()
        cur_stat = cur_stat.repeat(uset.shape[0],1)
        uset = self.usets.repeat(self.N-1,1,1)
        pred_states = []
        pred_states.append(cur_stat)
        roll_state = cur_stat.clone()
        for i in range(self.N-2):            
            yaw_tmp = roll_state[:,3].clone()
            vx = uset[i,:,0]
            delta = uset[i,:,1]            

            beta = torch.atan((self.Lr*torch.tan(delta)/self.L))
            roll_state[:,0] += vx * torch.cos(yaw_tmp+beta) * self.dt  # Assuming constant velocity
            roll_state[:,1] += vx * torch.sin(yaw_tmp+beta) * self.dt        
            roll_state[:,2] +=  0.0 ## z same for 2d case 
            roll_state[:,3] += (vx / self.L) * torch.tan(delta) *torch.cos(beta)* self.dt  # Assuming constant delta            
            roll_state[:,3] = wrap_to_pi_torch(roll_state[:,3])
            pred_states.append(roll_state.clone())

        pred_states = torch.stack(pred_states,dim=1)
        return pred_states, uset
            
    def np_kinematic_updates(self,u = None,state = None):     
        if isinstance(state, VehicleState):
            x = state.odom.pose.pose.position.x
            y = state.odom.pose.pose.position.y
            z = state.odom.pose.pose.position.z
            yaw = state.euler.yaw
            yaw = wrap_to_pi(yaw)            
        else:
            x = state[0]
            y = state[1]
            z = state[2]
            yaw = state[3]
            yaw = wrap_to_pi(yaw)            
        
        if isinstance(u,VehicleCommand):
            delta = u.steer
            vx = u.vx
        else:
            vx = u[0]
            delta = u[1]
        
        # Kinematic Bicycle Model Equations
        
        beta = math.atan((self.Lr*math.tan(delta)/self.L))
        x += vx * math.cos(yaw+beta) * self.dt  # Assuming constant velocity
        y += vx * math.sin(yaw+beta) * self.dt        
        yaw += (vx / self.L) * math.tan(delta) *math.cos(beta)* self.dt  # Assuming constant delta
        yaw = wrap_to_pi(yaw)
        z  = self.propogate_z(z)
        # Append predicted state to the list
        return np.array([x, y, z, yaw])
    
    def np_kinematic_updates_ukf(self,x,dt,u):
        return self.np_kinematic_updates(u=u,state=x)


    def ukf(self,u_sequence):
        assert len(u_sequence)!=self.horizon, "E: action sequence length does not match with horizon!"
        points = MerweScaledSigmaPoints(n=self.state_dim, alpha=1e-3, beta=2, kappa=0)
        u_sequence = u_sequence.detach().cpu().numpy()
        ukf = UnscentedKalmanFilter(dim_x=self.state_dim, dim_z=0, dt=self.dt, fx=self.np_kinematic_updates_ukf, hx=None, points=points,x_mean_fn=self.state_mean)
        ukf.x = self.cur_state4.detach().cpu().numpy() #init_state
        ukf.P = np.eye(self.state_dim)*0.005 #init_state_covariance # TODO: get from state estimator?
        # ukf.Q = Q_discrete_white_noise(dim=4, dt=self.dt, var=1.0) #TODO: may be estimate process noise from image?
        ukf.Q = np.eye(self.state_dim)*0.0001

        state_mean = []
        state_cov = []
        for i in range(int(self.N)-1):
            cur_u = u_sequence[i] # get first
            ukf.predict(u=cur_u)
            state_mean.append(ukf.x)
            state_cov.append(ukf.P)
        
        return state_mean, state_cov
    
    def state_mean(self,sigmas, Wm): # for mean yaw (1,361 -> 0 not 181)
        x = np.zeros(4)
        sum_sin = np.sum(np.dot(np.sin(sigmas[:, 3]), Wm))
        sum_cos = np.sum(np.dot(np.cos(sigmas[:, 3]), Wm))
        x[0] = np.sum(np.dot(sigmas[:, 0], Wm))
        x[1] = np.sum(np.dot(sigmas[:, 1], Wm))
        x[2] = np.sum(np.dot(sigmas[:, 2], Wm))
        x[3] = math.atan2(sum_sin, sum_cos)
        return x
    

if __name__ == "__main__":
    VehicleModel()