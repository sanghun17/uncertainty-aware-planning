import torch
import torchvision
import torch.optim as optim
import argparse
import matplotlib
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import model_BEV
import os
from torch.utils.data import Dataset, DataLoader, random_split

from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter  # Import SummaryWriter from torch.utils.tensorboard

matplotlib.style.use('ggplot')

writer = SummaryWriter()

class CustomDataset(Dataset):
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.data_files = []

        # Iterate through all subfolders inside the main data folder
        for subfolder in os.listdir(self.data_folder):
            subfolder_path = os.path.join(self.data_folder, subfolder)

            # Check if the item in the data folder is a subfolder
            if os.path.isdir(subfolder_path):
                # Get all the data files inside the subfolder
                subfolder_files = [os.path.join(subfolder_path, filename) for filename in os.listdir(subfolder_path)]
                self.data_files.extend(subfolder_files)

        self.transform_minus1_to_1 = transforms.Normalize(mean=[0.5], std=[0.5])  # Assuming you want to normalize to [-1, 1] range

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        file_path = self.data_files[idx]
        data_dict = torch.load(file_path)
        lidar_map = data_dict['lidar_map']
        camera_map = data_dict['camera_map']
        traversability_map = data_dict['traversability_map']

        lidar_map_mask = data_dict['lidar_map_mask']
        camera_map_mask = data_dict['camera_map_mask']
        traversability_map_mask = data_dict['traversability_map_mask']

        # Replace NaN values with 0 since nan occur unexpected behavior...
        # Actually, it does not affect to loss. since we donot use those pixels. 
        lidar_map[torch.isnan(lidar_map)] = 0
        camera_map[torch.isnan(camera_map)] = 0
        traversability_map[torch.isnan(traversability_map)] = 0

        # Ensure that all maps have the same spatial dimensions (20x20)
        assert lidar_map.size() == (20, 20), lidar_map.size()
        assert camera_map.size()  == (20, 20, 3), camera_map.size()
        assert traversability_map.size()  == (20, 20) , traversability_map.size()

        # Expand the 20x20 maps to 20x20x1
        lidar_map = lidar_map.unsqueeze(2)  # shape: 20x20x1
        traversability_map = traversability_map.unsqueeze(2)  # shape: 20x20x1
        lidar_map_mask =  lidar_map_mask.unsqueeze(2)
        traversability_map_mask= traversability_map_mask.unsqueeze(2)
        camera_map_mask=camera_map_mask.unsqueeze(2) # 20*20*1
        camera_map_mask= torch.cat([camera_map_mask,camera_map_mask,camera_map_mask], dim=2) # concentrate to 3 chneel with same mask! # 20*20*3
        
        stacked_map = torch.cat([lidar_map, camera_map, traversability_map, lidar_map_mask, camera_map_mask,traversability_map_mask], dim=2) # 20*20*10
        # Rearrange the dimensions of stacked_map
        stacked_map = stacked_map.permute(2, 0, 1)  # Change the order of dimensions (10,20,20)
        return stacked_map  

# construct the argument parser and parser the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', default=10, type=int, 
                    help='number of epochs to train the VAE for')
args = vars(parser.parse_args())

# leanring parameters
epochs = args['epochs']
batch_size = 32
lr = 0.0001
# Get Device for Training
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )
print(f"Using {device} device")

current_dir = os.path.dirname(os.path.realpath(__file__))
data_folder = '../input/data'
os.makedirs('../outputs', exist_ok=True)
absolute_data_folder = os.path.join(current_dir, data_folder)
dataset = CustomDataset(data_folder=absolute_data_folder)
dataset_size=len(dataset)
print("dataset_size: ", dataset_size)

train_data_size = int(0.8*dataset_size)
val_data_size = dataset_size-train_data_size
train_data, val_data = random_split(dataset, [train_data_size, val_data_size] )
train_loader = DataLoader(train_data, batch_size = batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_data, batch_size = batch_size, shuffle=True, drop_last=True)


model = model_BEV.LinearVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss(reduction='none')

def final_loss(bce_loss, mu, logvar):
    """
    This function will add the reconstruction loss (BCELoss) and the 
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    BCE = bce_loss 
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def fit(model, dataloader):
    model.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        # data, _ = data
        data = data.to(device)
        input_map=data[:,:5,:,:]
        input_map_mask=data[:,5:,:,:]
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(input_map)
        bce_loss = criterion(reconstruction, input_map.view(input_map.size(0), -1))
        bce_loss_mask = torch.mul(bce_loss, input_map_mask.view(input_map_mask.size(0), -1))
        bce_loss_mask = torch.sum(bce_loss_mask)
        loss = final_loss(bce_loss_mask, mu, logvar)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        # save the last batch input and output of every epoch
        if i == int(len(train_data)/dataloader.batch_size) - 1:
            num_rows = 4
            #Save the first image using the first row (channel 0) of the input and output
            input_map = input_map.view(batch_size, 5, 20, 20)[:4]
            reconstruction=reconstruction.view(batch_size, 5, 20, 20)[:4]
            input_map_mask=input_map_mask.view(batch_size, 5, 20, 20)[:4]
            data1 = input_map[:,0:1,:,:]
            reconstruction1 = reconstruction[:,0:1,:,:]
            mask1 = input_map_mask[:,0:1,:,:]
            data2 = input_map[:,1:4,:,:]
            reconstruction2 = reconstruction[:,1:4,:,:]
            mask2 = input_map_mask[:,1:4,:,:]
            data3 = input_map[:,4:5,:,:]
            reconstruction3 = reconstruction[:,4:5,:,:]
            mask3 = input_map_mask[:,4:5,:,:]
            
            # for tensorboard!
            data1_rgb = data1[1,:,:,:].expand(3,-1,-1)
            data2_rgb = data2[1,:,:,:]
            data3_rgb = data3[1,:,:,:].expand(3,-1,-1)
            mask1_rgb = mask1[1,:,:,:].expand(3,-1,-1)
            mask2_rgb = mask2[1,:,:,:]
            mask3_rgb = mask3[1,:,:,:].expand(3,-1,-1)
            reconstruction1_masked = reconstruction1[1, :, :, :] * mask1[1, :, :, :].expand(3, -1, -1)
            reconstruction2_masked = reconstruction2[1, :, :, :] * mask2[1, :, :, :]
            reconstruction3_masked = reconstruction3[1, :, :, :] * mask3[1, :, :, :].expand(3, -1, -1)
            log_image = torch.cat((torch.cat((data1_rgb, data2_rgb, data3_rgb), 1),
                                torch.cat((reconstruction1_masked, reconstruction2_masked, reconstruction3_masked), 1),
                                torch.cat((mask1_rgb, mask2_rgb, mask3_rgb), 1)),
                  2)
            writer.add_image("train image", log_image, epoch)

    train_loss = running_loss/len(dataloader.dataset)
    writer.add_scalar("Train Loss", train_loss, epoch)  # Log train loss to TensorBoard
    return train_loss

def validate(model, dataloader):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            # data, _ = data
            data = data.to(device)
            input_map=data[:,:5,:,:]
            input_map_mask=data[:,5:,:,:]
            reconstruction, mu, logvar = model(input_map)
            bce_loss = criterion(reconstruction, input_map.view(input_map.size(0), -1))
            bce_loss_mask = torch.mul(bce_loss, input_map_mask.view(input_map_mask.size(0), -1))
            bce_loss_mask = torch.sum(bce_loss_mask)
            loss = final_loss(bce_loss_mask, mu, logvar)
            running_loss += loss.item()
        
            # save the last batch input and output of every epoch
            if i == int(len(val_data)/dataloader.batch_size) - 1:
                num_rows = 4

               # Save the first image using the first row (channel 0) of the input and output
                input_map = input_map.view(batch_size, 5, 20, 20)[:4]
                reconstruction=reconstruction.view(batch_size, 5, 20, 20)[:4]
                input_map_mask=input_map_mask.view(batch_size, 5, 20, 20)[:4]
                data1 = input_map[:,0:1,:,:]
                reconstruction1 = reconstruction[:,0:1,:,:]
                mask1 = input_map_mask[:,0:1,:,:]
                data2 = input_map[:,1:4,:,:]
                reconstruction2 = reconstruction[:,1:4,:,:]
                mask2 = input_map_mask[:,1:4,:,:]
                data3 = input_map[:,4:5,:,:]
                reconstruction3 = reconstruction[:,4:5,:,:]
                mask3 = input_map_mask[:,4:5,:,:]

                # print(data1.size())
                # print(reconstruction1.size())
                # both_image1 = torch.cat((data1,reconstruction1,mask1))
                # save_image(both_image1.cpu(), f"../outputs/output_{epoch}_lidar.png", nrow=num_rows)

                # # Save the second image using rows 2 to 4 (channel 2) of the input and output
                # both_image2 = torch.cat((data2,reconstruction2,mask2))
                # save_image(both_image2.cpu(), f"../outputs/output_{epoch}_camera.png", nrow=num_rows)

                # # Save the third image using the fifth row (channel 0) of the input and output
                # both_image3 = torch.cat((data3,reconstruction3,mask3))
                # save_image(both_image3.cpu(), f"../outputs/output_{epoch}_grid.png", nrow=num_rows)
                
                # for tensorboard!

                data1_rgb = data1[1,:,:,:].expand(3,-1,-1)
                data2_rgb = data2[1,:,:,:]
                data3_rgb = data3[1,:,:,:].expand(3,-1,-1)
                mask1_rgb = mask1[1,:,:,:].expand(3,-1,-1)
                mask2_rgb = mask2[1,:,:,:]
                mask3_rgb = mask3[1,:,:,:].expand(3,-1,-1)
               
                reconstruction1_masked = reconstruction1[1, :, :, :] * mask1_rgb
                reconstruction2_masked = reconstruction2[1, :, :, :] * mask2_rgb
                reconstruction3_masked = reconstruction3[1, :, :, :] * mask3_rgb

                log_image = torch.cat((torch.cat((data1_rgb, data2_rgb, data3_rgb), 1),
                                    torch.cat((reconstruction1_masked, reconstruction2_masked, reconstruction3_masked), 1),
                                    torch.cat((mask1_rgb, mask2_rgb, mask3_rgb), 1)),
                      2)
                writer.add_image("valdiate image", log_image, epoch)
                
    val_loss = running_loss/len(dataloader.dataset)
    writer.add_scalar("Validation Loss", val_loss, epoch)  # Log validation loss to TensorBoard
    return val_loss

train_loss = []
val_loss = []
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = fit(model, train_loader)
    val_epoch_loss = validate(model, val_loader)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {val_epoch_loss:.4f}")

# Close the SummaryWriter when training is complete
writer.close()

# Save encoder and decoder weights
torch.save(model.encoder.state_dict(), 'encoder_weights.pth')
torch.save(model.decoder.state_dict(), 'decoder_weights.pth')