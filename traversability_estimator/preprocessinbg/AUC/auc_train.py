

import torch
import torch.nn as nn
from fake_data_gen import gen_img_set
from gat_utils import GATDataset
from torch.utils.data import DataLoader
from GAT_model import GridAttentionModel
from tensorboardX import SummaryWriter 
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from train_utils import animate_result

writer = SummaryWriter()

args = {'input_grid_width':64,
        'input_grid_height':64,
        'n_time_step':10, 
        'lstm_hidden_size': 12,  
        'init_fc_hidden_size':64,
        'input_state_dim':5, # [vx, vy, wz, roll, pitch] 
        'input_action_dim':2, # [vx, delta] 
        'data_size':10000,
        'batch_size':150,
        'num_epochs': 200
        }

## Generate train and test dataloader
state_input, action_input, image_input, image_output = gen_img_set(batch = args['data_size'], grid_width= args['input_grid_width'], grid_height=args['input_grid_height'], num_time_steps=args['n_time_step'], show_sample_data = False)
test_state_input, test_action_input, test_image_input, test_image_output = gen_img_set(batch = args['data_size'], grid_width= args['input_grid_width'], grid_height=args['input_grid_height'], num_time_steps=args['n_time_step'], show_sample_data = False)
gat_train_dataset = GATDataset(state_input, action_input, image_input, image_output)
gat_test_dataset = GATDataset(test_state_input, test_action_input, test_image_input, test_image_output)
train_dataloader = DataLoader(gat_train_dataset, batch_size=args['batch_size'], shuffle=True)
test_dataloader = DataLoader(gat_test_dataset, batch_size=args['batch_size'], shuffle=False)

# Initialize model and move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GridAttentionModel(args = args)
model.to(device)
# Count the number of parameters
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters in the model: {num_params}")
# Define loss function and optimizer
criterion = nn.MSELoss()  # Use MSE loss for pixel-wise comparison

# optimizer = Adam(model.parameters(), lr=0.001,weight_decay = 0.01)
optimizer = Adam(model.parameters(), lr=0.001)

scheduler = MultiStepLR(optimizer, milestones=[0.75 * args['num_epochs'], 0.85 * args['num_epochs']], gamma=0.1)

# Training loop
for epoch in range(args['num_epochs']):
    model.train()
    total_loss = 0.0
    for batch_idx, (state, action, image_in, image_out) in enumerate(train_dataloader):
        state, action, image_in, image_out = state.to(device).float(), action.to(device).float(), image_in.to(device).float(), image_out.to(device).float()

        optimizer.zero_grad()

        # Forward pass
        outputs = model(state, action,image_in)

        # Compute loss
        loss = criterion(outputs, image_out)
        total_loss += loss.item()

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / (batch_idx + 1)
    print(f"Epoch [{epoch + 1}/{args['num_epochs']}], Avg. Loss: {avg_loss:.6f}")

    # Log training loss to TensorBoard
    writer.add_scalar('Loss/Train', avg_loss, epoch + 1)
    
    # Save model checkpoint every 100 epochs
    if (epoch + 1) % 100 == 0:
        checkpoint_path = f"model_checkpoint_epoch_{epoch + 1}.pth"
        
        # Create a checkpoint dictionary including both model state and args
        checkpoint = {
            'model_state': model.state_dict(),
            'args': args
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Model checkpoint saved at epoch {epoch + 1}")

    # Intermediate evaluation every 10 epochs
    if (epoch + 1) % 10 == 0:
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_idx, (state, action, image_in, image_out) in enumerate(test_dataloader):
                state, action, image_in, image_out = state.to(device).float(), action.to(device).float(), image_in.to(device).float(), image_out.to(device).float()

                # Forward pass
                outputs = model(state, action,image_in)

                # Compute loss
                loss = criterion(outputs, image_out)
                test_loss += loss.item()

        avg_test_loss = test_loss / (batch_idx + 1)
        print(f"Test Avg. Loss: {avg_test_loss:.6f}")

        # Log test loss to TensorBoard
        writer.add_scalar('Loss/Test', avg_test_loss, epoch + 1)
        
      
          
        animate_result(image_out[0,:,:,:],outputs[0,:,:,:])

        
        
        
    
    scheduler.step()
# Close the SummaryWriter
writer.close()
