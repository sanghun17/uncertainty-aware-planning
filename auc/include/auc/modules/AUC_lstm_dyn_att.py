import torch
import torch.nn as nn
import numpy as np
import os 

'''
This module will be used by AUC_anprnn_model 
Input: state, attentioned predicted images, action prediciton 
Output: Hidden states for residual predicted positions 
'''
def ff(x,k,s):
    return (x-k)/s+1

def rr(y,k,s):
    return (y-1)*s+k

  
# Define the model architecture with LSTM-based spatial attention
class AUCLSTMModel(nn.Module):    
    def __init__(self, args):
        super(AUCLSTMModel, self).__init__()
        
        self.train_model = False

        self.input_grid_width = args['input_grid_width']
        self.input_grid_height = args['input_grid_height']
        self.input_state_dim = args['input_state_dim']
        self.input_action_dim = args['input_action_dim']
        self.n_time_step = args['n_time_step']
        self.init_fc_hidden_size = args['init_fc_hidden_size']
        self.lstm_hidden_size = args['lstm_hidden_size']        
        self.output_residual_dim = args['output_residual_dim']
        
        self.distributed_train = args['distributed_train']
        if self.distributed_train:
            self.gpu_id = int(os.environ["LOCAL_RANK"])
        else:
            self.gpu_id = 0    
        

        self.lstm_input_size = self.lstm_hidden_size + self.input_action_dim
        self.auc_lstm_hidden_size = args['lstm_hidden_size']           
        self.auc_lstm_input_dim = self.auc_lstm_hidden_size + self.input_action_dim
        assert self.auc_lstm_input_dim-self.input_action_dim-self.input_state_dim > 0, 'hidden layer should be greater than the sum of action and state dim'
        
        
        self.auc_output_fc_hidden = args['auclstm_out_fc_hidden_size']        
        self.auclstm_output = args['auclstm_output_dim']
        
        self.init_input_size = self.input_grid_width*self.input_grid_height + self.input_state_dim + self.input_action_dim        
        # #TODO ## assume we have 1 channel ############
        
        # # Convolutional layer  
        '''
        This layer takes the attended image set and convert it to 
        '''      
        # hconv_kernel = [5,3,5]
        # hconv_stride = [5,3,3]
        # wconv_kernel = [5,3,5]
        # wconv_stride = [5,5,3]
        hconv_kernel = [3,3] #[5,3,3]
        hconv_stride = [3,3] #[5,3,1]
        wconv_kernel = [5,3] #[5,7,5]
        wconv_stride = [3,3] #[5,3,1]
        ## TODO: below conv layer has same structure inside of ltatt model, can we use only one?(borrow it from ltatt)
        self.att_image_to_lstm_conv_layer = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=(hconv_kernel[0],wconv_kernel[0]), stride=(hconv_stride[0],wconv_stride[0]), padding =0),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 16, kernel_size=(hconv_kernel[1],wconv_kernel[1]), stride=(hconv_stride[1],wconv_stride[1]), padding =0),
            # nn.BatchNorm2d(16),
            nn.ReLU()
        ).to(self.gpu_id) # Optional pooling layer
        
        self.auc_conv_out_size = self._get_conv_out_size(self.att_image_to_lstm_conv_layer,self.input_grid_height,self.input_grid_width, input_channels= 4 )        
        
        self.init_att_image_to_lstm_fc_layer = nn.Sequential(            
            nn.Linear(self.auc_conv_out_size, self.auc_lstm_input_dim-self.input_action_dim-self.input_state_dim),
            # nn.BatchNorm1d(self.auc_lstm_input_dim-self.input_action_dim-self.input_state_dim),
            nn.ReLU()
        ).to(self.gpu_id)
                
        self.att_image_to_lstm_fc_layer = nn.Sequential(
            nn.Linear(self.auc_conv_out_size, self.auc_lstm_input_dim-self.input_action_dim),
            # nn.BatchNorm1d(self.auc_lstm_input_dim-self.input_action_dim),
            nn.ReLU()
        ).to(self.gpu_id)

        # Define the Second LSTM layer for estimating the residual position errors
        self.auc_lstm = nn.LSTM(input_size=self.auc_lstm_input_dim,  
                    hidden_size=self.auc_lstm_hidden_size,
                    num_layers=1,
                    batch_first=True).to(self.gpu_id)
        
        
        self.auc_output_fc = nn.Sequential(
                nn.Linear(self.auc_lstm_hidden_size, self.auc_output_fc_hidden),                  
                nn.ReLU(),
                nn.Linear(self.auc_output_fc_hidden, self.auclstm_output)                
        ).to(self.gpu_id) 
        
        self.tanh = torch.nn.Tanh()

     

    def _get_conv_out_size(self, model, width,height, input_channels = 4):
        dummy_input = torch.randn(1, input_channels, width,height, requires_grad=False).to(self.gpu_id).float()         
        conv_output = model(dummy_input)
        return conv_output.view(-1).size(0)
    
        
    def forward(self, input):
        ''' 
        state : [batch, features]
        action_predictions : [batch, seq, feature]
        images : [batch, channel, height, width] -> for train
        images : [channel, height, width] -> for inference , we need to repeat it for batched predicted actions

        '''
        state, action_predictions,images = input                

        if self.train_model is False:
            images = images.unsqueeze(dim=0)
        ## TODO: propogate mean and std for image attention module 
        ## TODO: check if the prediction is within the image.
        ## TODO: Create filter mask for such unpredictable area        
        
        hidden_outputs = []
        batch_size = state.shape[0]                
        # attended_images have [batch , sequence, image_width, image_height]
        
        #####################################################################################
        #######################   init LSTM2 for residual error #############################
        #####################################################################################
        # compute the inital features for the second LSTM
        
        init_lstm_input_state_action = torch.cat((state, action_predictions[:,0,:]), dim=1).to(self.gpu_id).float() 
        # init_lstm_input - > 
        
        ## make sure the input has [batch , channel, width, height]
        # assert len(attentioned_imgs[:,0,:,:,:].shape) == 4, 'make sure the input to conv has [batch , channel, width, height]'
        init_conv_result = self.att_image_to_lstm_conv_layer(images)
        
        ''' 
        #### image has only single batch to save memory 
        '''
        ## TODO: currently only single image is used...
        if self.train_model is False:
            init_conv_fc_result = self.init_att_image_to_lstm_fc_layer(init_conv_result.view(1,-1))
            batched_init_conv_fc_result = init_conv_fc_result.repeat(state.shape[0],1)
        else:
            batched_init_conv_fc_result = self.init_att_image_to_lstm_fc_layer(init_conv_result.view(batch_size,-1))
##########################################################################################

        init_lstm_input = torch.cat([init_lstm_input_state_action,batched_init_conv_fc_result], dim=1).to(self.gpu_id).float()
        
        # Initialize the LSTM hidden state
        h0 = torch.zeros(self.auc_lstm.num_layers, batch_size, self.auc_lstm_hidden_size).to(self.gpu_id).float()
        c0 = torch.zeros(self.auc_lstm.num_layers, batch_size, self.auc_lstm_hidden_size).to(self.gpu_id).float()
        
        # initial guess from LSTM
        init_lstm_output, (h,c) = self.auc_lstm(init_lstm_input.unsqueeze(dim=1),(h0,c0))
        
        fc_out = self.auc_output_fc(h[-1,:,:])  
        remapped_h = self.tanh(fc_out)
        hidden_outputs.append(remapped_h)
        
        #####################################################################################
        #######################    LSTM for getting hidden states error   #######################
        #####################################################################################
        
        for t in range(1,action_predictions.shape[1]):
          ## TODO: currently only single image is used
            conv_result = self.att_image_to_lstm_conv_layer(images)
            if self.train_model is False:                
                conv_fc_result = self.att_image_to_lstm_fc_layer(conv_result.view(1,-1))            
                batched_conv_fc_result = conv_fc_result.repeat(state.shape[0],1)
            else:                
                batched_conv_fc_result = self.att_image_to_lstm_fc_layer(conv_result.view(batch_size,-1))            

            
            
##########################################################################################
            

            lstm_input = torch.cat([action_predictions[:,t,:],batched_conv_fc_result], dim=1).to(self.gpu_id).float()
            lstm_output, (h,c) = self.auc_lstm(lstm_input.unsqueeze(dim=1),(h,c))        

            # hidden_outputs.append(h[-1,:,:].clone())            
            '''
            '''
            fc_out = self.auc_output_fc(h[-1,:,:])  
            remapped_h = self.tanh(fc_out) +t*2
            hidden_outputs.append(remapped_h)
        
        
        return hidden_outputs

