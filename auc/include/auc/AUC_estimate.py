
#!/usr/bin/env python3
from auc.common.file_utils import *
import torch
import os
from auc.gp_encoder.gp_encoderModel import GPAUC
import time
import datetime
from auc.common.pytypes import VehicleState, AUCModelData, SimData, CameraIntExt
from auc.train.train_utils import get_action_set, normalize_depth, normalize_color
import numpy as np
import gpytorch  

class AUCEStimator:
    def __init__(self, dt = 0.2, N_node = 10, model_path = 'singl_aucgp_snapshot.pth') -> None:
        if hasattr(os.environ,"LOCAL_RANK"):        
            self.gpu_id = int(os.environ["LOCAL_RANK"])
        else:
            self.gpu_id = 0
        assert model_path is not None, 'need to load proper model first'
        self.model = None
        self.likelihood = None
        file_path = os.path.join(snapshot_dir, model_path)                            
        assert os.path.exists(file_path), f"Cannot find GPAUC model at {file_path}"
        print("Loading snapshot")
        self.load_model(file_path)
        

        
    def load_model(self,file_path):     
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(file_path, map_location=loc)
        # TODO: need to save args while training   
        self.args = snapshot["Args"]        
        self.model = GPAUC(args=self.args)        
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.model.to(self.gpu_id).float()
        self.model.eval()
        self.likelihood = snapshot["Liklihood"]
        self.likelihood.eval()
        self.model.train_norm_stat = snapshot["Norm_stat"]        
        print(f"Model has been loaded {file_path}")
        


        
    def convert_AUCModelData_to_input(self,input_auc : AUCModelData, pred_actions: torch.tensor):        
        '''
         for each pred_action batch, state and img are repeated 
        '''
        pred_actions = torch.transpose(pred_actions,dim0=0,dim1=1)    
        pred_actions = pred_actions.to(self.gpu_id)                
        batch_size = pred_actions.shape[0]        
        horizon_size = pred_actions.shape[1]
        states = torch.tensor(input_auc.get_cur_vehicle_state_input()).to(self.gpu_id)        
        states = states.repeat(batch_size,1)

        color_img = np.transpose(input_auc.color,(2,0,1))
        depth = input_auc.depth
        depth_img = depth[np.newaxis,:]
        color = torch.tensor(normalize_color(color_img)).to(self.gpu_id)
        depth= torch.tensor(normalize_depth(depth_img)).to(self.gpu_id)
        imgs = torch.cat((color,depth), dim=0).to(self.gpu_id)            
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ## TODO: need to implement model based attentioned image module            
        # attentioned_imgs = attentioned_imgs.repeat(batch_size,horizon_size,1,1,1)
        ###############################################################################
        ###############################################################################
        ###############################################################################
        
        
        intput_source = (states.float(),pred_actions.float(), imgs.float())


        return intput_source


        
    def pred(self, input_auc : AUCModelData, pred_actions : torch.tensor):
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            input_to_model = self.convert_AUCModelData_to_input(input_auc, pred_actions)
            batch_size = input_to_model[0].shape[0]
            model_output = self.model(input_to_model)
            normalized_pred_residual = self.likelihood(model_output)            
            pred_residual_mean,pred_residual_std = self.model.outputToReal(batch_size,normalized_pred_residual, model_output)            
            return pred_residual_mean, pred_residual_std



if __name__ == "__main__":    
    auc_estimator = AUCEStimator(model_path = '/home/racepc/offroad_sim_data/models/island_grass/dist_process_1770.pth', dt = 0.2, N_node = 10)
    