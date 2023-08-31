
#!/usr/bin/env python3
from auc.common.file_utils import *
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from auc.train.train_utils import SampleGenerator
from torch.optim import Adam, SGD
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import torch.nn as nn
import gpytorch
from tensorboardX import SummaryWriter
import time
import datetime
from auc.gp_encoder.gp_encoderModel import GPAUC

import numpy as np
    
def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        liklihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
        save_every: int,
        snapshot_path: str,
    ) -> None:
        if hasattr(os.environ,"LOCAL_RANK"):        
            self.gpu_id = int(os.environ["LOCAL_RANK"])
        else:
            self.gpu_id = 0
        
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = f"{train_log_dir}/single_process_{current_time}"

        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.model = model.to(self.gpu_id).float()
        self.likelihood = liklihood.to(self.gpu_id).float()               
        self.train_data = train_data
        self.mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model.gp_layer, num_data=len(self.train_data)*10).to(self.gpu_id) 
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = os.path.join(snapshot_dir, snapshot_path)                            
        if os.path.exists(self.snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(self.snapshot_path)
        if hasattr(os.environ,"LOCAL_RANK"):        
            self.model = DDP(self.model, device_ids=[self.gpu_id])

    


    def check_model_graph(self):
        dataiter = iter(self.train_data)
        images, labels = dataiter.next()
        target_set, context_set = self._batched_data_to_model_input(images, labels)
        a = (target_set, context_set)
        self.writer.add_graph(self.model, [a], verbose= True)


    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.likelihood = snapshot["Liklihood"]
        self.model.train_norm_stat = snapshot["Norm_stat"]        
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def get_target_context_set(self,states, pred_actions,input_imgs,pred_pose_residuals):
        # TODO: currently target and context set has equal length for computing attention.....
        # how to make it work with varying context and target set length 
        
        batch_size = states.shape[0]
        assert batch_size > 1, 'Batch size should be greater than one'
        indices = torch.randperm(batch_size)[:batch_size // 2]

        c_states = states[indices]
        c_pred_actions = pred_actions[indices]
        c_input_imgs = input_imgs[indices]
        c_pred_pose_residuals = pred_pose_residuals[indices]

        t_states = states[indices]
        t_pred_actions = pred_actions[indices]
        t_input_imgs = input_imgs[indices]
        t_pred_pose_residuals = pred_pose_residuals[indices]
        
        context_set = (c_states, c_pred_actions, c_input_imgs, c_pred_pose_residuals)
        target_set = (t_states, t_pred_actions, t_input_imgs, t_pred_pose_residuals)
        
        return target_set, context_set 

    def _batched_data_to_model_input(self,source,targets):
        (states,pred_actions, colors, depths) = source        
        (pred_pose_residuals, pred_att_color_imgs, pred_att_depth_imgs) = targets

        states = states.to(self.gpu_id)
        pred_actions = pred_actions.to(self.gpu_id)
        colors = colors.to(self.gpu_id)
        depths = depths.to(self.gpu_id)
        pred_pose_residuals = pred_pose_residuals.to(self.gpu_id)
        pred_att_color_imgs = pred_att_color_imgs.to(self.gpu_id)
        pred_att_depth_imgs = pred_att_depth_imgs.to(self.gpu_id)
        input_imgs = torch.cat((colors,depths), dim=1).to(self.gpu_id)                        
        attentioned_img = torch.cat((pred_att_color_imgs,pred_att_depth_imgs), dim=2).to(self.gpu_id)    
        ## TODO: do we need to add epsilon for output image        
        attentioned_img +=1e-6
        target_set, context_set  = self.get_target_context_set(states, pred_actions,input_imgs,pred_pose_residuals)

        return target_set, context_set

    def _plot_model_performance_to_tensorboard(self,epoch, mu, sigma, ground_truth_pred_pose_residuals):        
        # writer.add_histogram('Normal Distribution', data, global_step=0)
        mean_error = (mu- ground_truth_pred_pose_residuals)
        unpack_temporal_mean_error = mean_error.view(-1,9,mean_error.shape[1])
        # writer.add_histogram('Normal Distribution', data, global_step=0)                    
        for sequence_idx in range(unpack_temporal_mean_error.shape[1]):
            for feature_idx in range(unpack_temporal_mean_error.shape[2]):
                sequence_feature_data = unpack_temporal_mean_error[:, sequence_idx, feature_idx]
                self.writer.add_histogram(f'Feature_{feature_idx}/Sequence_{sequence_idx}', sequence_feature_data, global_step=epoch)

        unpack_temporal_sigma = mean_error.view(-1,9,mean_error.shape[1])
        for sequence_idx in range(unpack_temporal_sigma.shape[1]):
            for feature_idx in range(unpack_temporal_sigma.shape[2]):
                sequence_feature_data = unpack_temporal_sigma[:, sequence_idx, feature_idx]
                self.writer.add_histogram(f'Sigma_{feature_idx}/Sequence_{sequence_idx}', sequence_feature_data, global_step=epoch)

    def _run_batch(self, source, targets,epoch):
        (state,action_predictions, colors, depths) = source      
        colors = colors.to(self.gpu_id)
        depths = depths.to(self.gpu_id)        
        image = torch.cat((colors,depths), dim=1).to(self.gpu_id)   
        state = state.to(self.gpu_id)
        action_predictions = action_predictions.to(self.gpu_id)                
        intput_source = (state,action_predictions, image)

        with gpytorch.settings.use_toeplitz(False):
            self.model.train()
            self.likelihood.train()
            self.model.gp_layer.train()
            self.optimizer.zero_grad()
            gpoutput = self.model(intput_source) 
            # ground_truth_pred_pose_residuals = targets[0]
            residual_pose = targets[0][:,:,0:2]
            temporal_stacked_residual_pose = residual_pose.view(-1,residual_pose.shape[2]).to(self.gpu_id)
            loss = -self.mll(gpoutput, temporal_stacked_residual_pose)             
            loss.backward()
            self.optimizer.step()                        
            self.writer.add_scalar("loss_total", float(loss), epoch)               

        for name, param in self.model.named_parameters():
            self.writer.add_histogram(name + '/grad', param.grad, global_step=epoch)         
            if 'weight' in name:
                self.writer.add_histogram(name + '/weight', param, epoch)
            if 'bias' in name:
                self.writer.add_histogram(name + '/bias', param, epoch)
       ## TODO: why not also include loss for mathing the attention image matching            
        
        
        # with torch.no_grad(), gpytorch.settings.fast_pred_var():        
        #     mx = gpoutput.mean.detach()
        #     std = gpoutput.stddev.detach()       
        #     self.model.eval()
        #     self.likelihood.eval()
        #     self.model.gp_layer.eval()
            
            
        #     start_time = time.time()                        
        #     pred_residual = self.likelihood(self.model(intput_source) )
        #     # Record the end time
        #     end_time = time.time()
        #     mx = pred_residual.mean
        #     std = pred_residual.stddev

            # Calculate the time difference
            # time_difference = end_time - start_time
            # print("Time between calls:", time_difference, "seconds")     
            
            # self._plot_model_performance_to_tensorboard(epoch, mx, std, temporal_stacked_residual_pose)
        return loss.item()

    def _run_epoch(self, epoch):
        b_sz = next(iter(self.train_data))[0][0].shape[0]
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        if hasattr(os.environ,"LOCAL_RANK"):
            self.train_data.sampler.set_epoch(epoch)
        total_loss = 0.0
        count = 0
        for source, targets in self.train_data:                
            # source = source.to(self.gpu_id)
            # targets = targets.to(self.gpu_id)
            loss_bath = self._run_batch(source, targets,epoch)
            total_loss+=loss_bath
            count+=1
            print(f" Current batch count = {count} | Epoch {epoch} | Batchsize: {b_sz} | BATCH LOSS: {loss_bath:.6f}")



        avg_loss_non_torch = total_loss / (count+1)        
        self.writer.add_scalar('LTATT Loss/Train', avg_loss_non_torch, epoch + 1)        
        print(f" Epoch {epoch} | Batchsize: {b_sz} | AVG_LOSS: {avg_loss_non_torch:.6f}")
        


    def _save_snapshot(self, epoch):        
    
        if hasattr(os.environ,"LOCAL_RANK"):
            snapshot = {
                "MODEL_STATE": self.model.module.state_dict(),
                "EPOCHS_RUN": epoch,
            }
        else:
            snapshot = {
                "MODEL_STATE": self.model.state_dict(),
                "EPOCHS_RUN": epoch,
                "Args": self.model.args,
                "Liklihood": self.likelihood,
                "Norm_stat": self.model.train_norm_stat
            }
        
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)


def load_train_objs(args):
    preprocessed_dataset_load = False
    preprocessed_data_path = os.path.join(preprocessed_dir, f'preprocessed_data_656.pth')                    
    dirs = [train_dir] 
    if preprocessed_dataset_load:
        data_path = preprocessed_data_path # os.path.join(preprocessed_dir, f'preprocessed_data_185.pth')                
        sampGen = SampleGenerator(dirs, data_path = data_path)
    else:
        sampGen = SampleGenerator(dirs)
    train_set = sampGen.get_dataset()

    # train_set = MyTrainDataset(2048)  # load your dataset

    ################################################

    # model = LocalTemporalAttention(args = args)
    model = GPAUC(args= args, train_norm_stat= train_set.get_norm_stats())
    
    ###############################################
    # model = torch.nn.Linear(20, 1)  # load your model

    liklihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=model.gp_output_dim)  
                
    lr = 0.01
    optimizer = SGD([
        {'params' :model.rnn.parameters()},        
        {'params': model.gp_layer.hyperparameters(), 'lr': lr * 0.1},
        {'params': model.gp_layer.variational_parameters()},
        {'params': liklihood.parameters()}
    ], lr=lr, momentum=0.9, nesterov=True, weight_decay=0)


    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer, liklihood


def prepare_dataloader(dataset: Dataset, batch_size: int):
    if hasattr(os.environ,"LOCAL_RANK"):
        dataloader = DataLoader(dataset,batch_size=batch_size,pin_memory=True,shuffle=False,sampler=DistributedSampler(dataset))
    else:
        dataloader = DataLoader(dataset,batch_size=batch_size,pin_memory=True,shuffle=True)
    return dataloader


def main(save_every: int, total_epochs: int, batch_size: int):
    # ddp_setup()
    args = {'input_grid_width':153,
            'input_grid_height':128,
            'n_time_step':10, 
            'lstm_hidden_size': 14,  
            'init_fc_hidden_size':6,
            'input_state_dim':8, # [vx, vy, vz, wx, wy, wz, roll, pitch] 
            'input_action_dim':2, # [vx, delta]                 
            'batch_size':2,
            'num_epochs': 2,
            'output_residual_dim': 2, # [delx, dely, delz, delyaw]
            'distributed_train': False,
            'arnp_train': True,
            'auclstm_output_dim': 2,
            'auclstm_out_fc_hidden_size': 6                      
            }
    
    snapshot_path = 'singl_aucgp_snapshot.pth'
    
    dataset, model, optimizer,liklihood = load_train_objs(args)
    train_data = prepare_dataloader(dataset, batch_size)    
    trainer = Trainer(model, train_data, optimizer, liklihood, save_every, snapshot_path)
    trainer.train(total_epochs)
    


if __name__ == "__main__":
    save_every = 10
    total_epochs = 1000
    batch_size = 160
    main(save_every, total_epochs, batch_size)