import torch
from torch import nn 
from torch.nn import functional as F
import gpytorch
from auc.gp_encoder.gpytorch_models import IndependentMultitaskGPModelApproximate
from auc.modules.AUC_lstm_dyn_att import AUCLSTMModel
import os
import numpy as np

class GPAUC(gpytorch.Module):    
    """LSTM-based Contrasiave Auto Encoder"""
    def __init__(
        self, args, train_norm_stat = None):
        """
        args['input_size']: int, batch_size x sequence_length x input_dim
        args['hidden_size']: int, output size of LSTM VAE
        args['latent_size']: int, latent z-layer size
        num_lstm_layer: int, number of layers in LSTM
        """
        super(GPAUC, self).__init__()        
        self.args = args
        self.train_norm_stat = train_norm_stat

        self.input_grid_width = args['input_grid_width']
        self.input_grid_height = args['input_grid_height']
        self.input_state_dim = args['input_state_dim']
        self.input_action_dim = args['input_action_dim']
        self.n_time_step = args['n_time_step']
        self.init_fc_hidden_size = args['init_fc_hidden_size']
        self.lstm_hidden_size = args['lstm_hidden_size']
        self.output_residual_dim = args['output_residual_dim']
        self.lstm_input_size = self.lstm_hidden_size + self.input_action_dim
        self.auc_lstm_hidden_size = args['lstm_hidden_size']
        
        

        self.distributed_train = args['distributed_train']
        if self.distributed_train:
            self.gpu_id = int(os.environ["LOCAL_RANK"])
        else:
            self.gpu_id = 0   
        
        # dimensions        
        
        self.gp_input_size = args['output_residual_dim']
        self.gp_output_dim = args['output_residual_dim']
        self.auclstm_output = args['auclstm_output_dim']
                
        self.seq_len = args['n_time_step']

        self.rnn = AUCLSTMModel(args)        
        
        self.gp_layer = IndependentMultitaskGPModelApproximate(inducing_points_num=100,
                                                                input_dim=self.auclstm_output,
                                                                num_tasks=self.gp_output_dim)  # Independent
        
        
            
    def outputToReal(self, batch_size, pred_dist, model_output):
        with torch.no_grad():
            standardized_mean = pred_dist.mean
            batch_size
            standardized_mean_ = standardized_mean.view(batch_size,-1,standardized_mean.shape[-1])
            assert self.train_norm_stat is not None, "normalizing data requires"
                
            pred_pose_mean = torch.tensor(self.train_norm_stat['pred_pose_residuals_mean']).to(self.gpu_id)
            pred_pose_mean = pred_pose_mean[:,:2]
            pred_pose_mean = pred_pose_mean.repeat(batch_size,1,1)
            pred_pose_mean = pred_pose_mean.view(-1,2)

            pred_pose_std = torch.tensor(self.train_norm_stat['pred_pose_residuals_std']).to(self.gpu_id)
            pred_pose_std = pred_pose_std[:,:2]
            pred_pose_std = pred_pose_std.repeat(batch_size,1,1)
            pred_pose_std = pred_pose_std.view(-1,2)

        
            mean = standardized_mean * pred_pose_std + pred_pose_mean
            ## TODO: need to compute the unstandardaized std , below is incorrect
            # std = model_output.covariance_matrix.cuda() * (torch.diag(torch.pow(pred_pose_std, 2))).repeat(pred_pose_std.shape[0])
            std = pred_dist.stddev*pred_pose_std
        return mean, std



    def forward(self, input_data):    
        
        (states, action_predictions,imgs) = input_data                        
        hiddens = self.rnn((states, action_predictions,imgs))
        temporal_stacked_hiddens = torch.vstack(hiddens).to(self.gpu_id)          
        gpoutput = self.gp_layer(temporal_stacked_hiddens)
        
        return gpoutput
                
    def compute_euclidian_dist(self,A,B):
        # A = torch.randn(512, 5, 9)
        # B = torch.randn(512, 5, 9)
        # Expand dimensions to enable broadcasting
        A_expanded = A.unsqueeze(1)  # [512, 1, 4, 7]
        B_expanded = B.unsqueeze(0)  # [1, 512, 4, 7]
        # Calculate the Euclidean norm between each pair of vectors
        distances = torch.norm((A_expanded - B_expanded), dim=-1)  # [512, 512, 4]        
        # Sum the Euclidean norms over the sequence dimension
        if len(A.shape) > 2:            
            seq_sum_distances = torch.sum(distances, dim=2)  # [512, 512]            
        else:
            seq_sum_distances = distances  # [512, 512]
        # normalized_tensor = F.normalize(seq_sum_distances, dim=(0, 1))
        return seq_sum_distances   
       
    def square_exponential_kernel(self, x1, x2, length_scale=1.7, variance=0.1):
        """
        Computes the square exponential (RBF) kernel between two sets of data points.

        Args:
            x1 (Tensor): First set of data points with shape (N1, D).
            x2 (Tensor): Second set of data points with shape (N2, D).
            length_scale (float, optional): Length scale parameter. Default is 1.0.
            variance (float, optional): Variance parameter. Default is 1.0.

        Returns:
            kernel_matrix (Tensor): Kernel matrix with shape (N1, N2).
        """
        diff = x1.unsqueeze(0) - x2.unsqueeze(1)
        norm_sq = torch.sum(diff ** 2, dim=-1)        
        kernel_matrix = variance * torch.exp(-0.5 * norm_sq / (length_scale ** 2))
        return kernel_matrix

    def loss_cont(self,*args, **kwargs) -> dict:
        """
        Computes loss
        loss = contrasive_loss + reconstruction loss 
        """
        recons = args[0]
        input = args[1]
        theta = args[2]
        diff_input = input[:,1:,[0,1,2,3,4,5,6]] - input[:,0:-1,[0,1,2,3,4,5,6]]
        input_diff_mean = self.compute_euclidian_dist(diff_input,diff_input) # input_diff_mean = batch x batch 
        theta_diff_mean = self.square_exponential_kernel(theta,theta) # theta_diff_mean = batch x batch 
        upper_bound = 1.3* input_diff_mean
        lower_bound = 0.7* input_diff_mean
        deviation_loss = torch.sum(torch.relu(theta_diff_mean - upper_bound) + torch.relu(lower_bound - theta_diff_mean))

        cont_loss = deviation_loss # F.mse_loss(input_diff_mean,theta_diff_mean)
        
        
        recons_loss = F.mse_loss(recons, input)                

        cont_loss_weight_multiplier = 1e-5
        cont_loss_weighted = cont_loss*cont_loss_weight_multiplier
        recons_loss_weight = 1e-10
        recons_loss_weighted = recons_loss*recons_loss_weight
        loss = recons_loss_weighted + cont_loss_weighted
        return {
            "encoder_part_loss": loss,
            "cont_loss" : cont_loss_weighted.detach(),
            "recons_loss" : recons_loss_weighted.detach()                
        }

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        recons_loss = F.mse_loss(recons, input)
        loss = recons_loss 
        return {
            "loss": loss
        }

    