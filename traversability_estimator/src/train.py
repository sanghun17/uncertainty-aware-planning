import torch
import torchvision
import torch.optim as optim
import argparse
import matplotlib
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import model
import os
from torch.utils.data import Dataset, DataLoader, random_split

from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
matplotlib.style.use('ggplot')

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

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Assuming you want to normalize to [-1, 1] range
        ])

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        file_path = self.data_files[idx]
        data_dict = torch.load(file_path)
        lidar_map = data_dict['lidar_map']
        camera_map = data_dict['camera_map']
        grid_map = data_dict['grid_map']

        # Replace NaN values with 0
        lidar_map[torch.isnan(lidar_map)] = 0
        camera_map[torch.isnan(camera_map)] = 0
        grid_map[torch.isnan(grid_map)] = 0
        
        # Ensure that all maps have the same spatial dimensions (20x20)
        assert lidar_map.size() == (20, 20), lidar_map.size()
        assert camera_map.size()  == (20, 20, 3), camera_map.size()
        assert grid_map.size()  == (20, 20) , grid_map.size()

        # Expand the 20x20 maps to 20x20x1
        lidar_map = lidar_map.unsqueeze(2)  # shape: 20x20x1
        grid_map = grid_map.unsqueeze(2)  # shape: 20x20x1

        # Concatenate the three maps along the channel dimension (axis=2)
        stacked_map = torch.cat([lidar_map, camera_map, grid_map], dim=2)
        
        return stacked_map

# construct the argument parser and parser the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', default=10, type=int, 
                    help='number of epochs to train the VAE for')
args = vars(parser.parse_args())

# leanring parameters
epochs = args['epochs']
batch_size = 10
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

# transforms
transform = transforms.Compose([
    transforms.ToTensor(),
])

current_dir = os.path.dirname(os.path.realpath(__file__))
data_folder = '../input/data'
absolute_data_folder = os.path.join(current_dir, data_folder)
dataset = CustomDataset(data_folder=absolute_data_folder)
dataset_size=len(dataset)
print("dataset_size: ", dataset_size)

train_data_size = int(0.8*dataset_size)
val_data_size = dataset_size-train_data_size
train_data, val_data = random_split(dataset, [train_data_size, val_data_size] )
train_loader = DataLoader(train_data, batch_size = batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_data, batch_size = batch_size, shuffle=True, drop_last=True)


model = model.LinearVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss(reduction='sum')

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
        data = data.view(data.size(0), -1)
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(data)
        bce_loss = criterion(reconstruction, data)
        loss = final_loss(bce_loss, mu, logvar)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    train_loss = running_loss/len(dataloader.dataset)
    return train_loss

def validate(model, dataloader):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            # data, _ = data
            data = data.to(device)
            data = data.view(data.size(0), -1)
            reconstruction, mu, logvar = model(data)
            bce_loss = criterion(reconstruction, data)
            loss = final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()
        
            # save the last batch input and output of every epoch
            if i == int(len(val_data)/dataloader.batch_size) - 1:
                num_rows = 4

               # Save the first image using the first row (channel 0) of the input and output
                data = data.view(batch_size, 5, 20, 20)[:4]
                reconstruction=reconstruction.view(batch_size, 5, 20, 20)[:4]
                data1 = data[:,0:1,:,:]
                reconstruction1 = reconstruction[:,0:1,:,:]
                data2 = data[:,2:5,:,:]
                reconstruction2 = reconstruction[:,2:5,:,:]
                data3 = data[:,4:5,:,:]
                reconstruction3 = reconstruction[:,4:5,:,:]
                both_image1 = torch.cat((data1,reconstruction1))
                save_image(both_image1.cpu(), f"../outputs/output_{epoch}_lidar.png", nrow=num_rows)

                # Save the second image using rows 2 to 4 (channel 2) of the input and output
                both_image2 = torch.cat((data2,reconstruction2))
                save_image(both_image2.cpu(), f"../outputs/output_{epoch}_camera.png", nrow=num_rows)

                # Save the third image using the fifth row (channel 0) of the input and output
                both_image3 = torch.cat((data3,reconstruction3))
                save_image(both_image3.cpu(), f"../outputs/output_{epoch}_grid.png", nrow=num_rows)
    val_loss = running_loss/len(dataloader.dataset)
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