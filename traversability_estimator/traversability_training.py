from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
import os

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

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        file_path = self.data_files[idx]
        data_dict = torch.load(file_path)
        lidar_map = data_dict['lidar_map']
        camera_map = data_dict['camera_map']
        grid_map = data_dict['grid_map']
        return lidar_map, camera_map, grid_map
    
class NeuralNetwork(nn.Module):
    def __init__(self, input_channel, hidden_size, output_size, input_height, input_width):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * input_height * input_width, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
class NeuralNetwork_SH(nn.Module):
    def __init__(self, input_channel, hidden_size, output_size, input_height, input_width):
        super(NeuralNetwork_SH, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear( input_height * input_width, 1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        return x
    
if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_folder = 'data'
    absolute_data_folder = os.path.join(current_dir, data_folder)
    dataset = CustomDataset(data_folder=absolute_data_folder)
    dataset_size=len(dataset)
    print("dataset_size: ", dataset_size)
    
    trainset_size = int(0.8*dataset_size)
    valset_szie = dataset_size-trainset_size
    trainset, valset = random_split(dataset, [trainset_size, valset_szie] )
    train_loader = DataLoader(trainset, batch_size = 10, shuffle=True, drop_last=True)
    val_loader = DataLoader(valset, batch_size = 10, shuffle=True, drop_last=True)

    # Get Device for Training
    device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )
    print(f"Using {device} device")

    # Define the input channels, hidden size, and output size of the neural network.
    input_channels = 1  # Single channel (grayscale) input.
    input_height = 48  # Update this with the desired height.
    input_width = 64  # Update this with the desired width.
    hidden_size = 32  # Choose an appropriate number of hidden units.
    output_size = 1  # The output is the numerical cost in this case.
    input_size = [ ]
    # Create an instance of the CostCalculator model.
    model = NeuralNetwork(input_channels, hidden_size, output_size, input_height, input_width)
    # model = NeuralNetwork_SH(input_channels, hidden_size, output_size, input_height, input_width)
    print("model: ",model)
    
    # Define the loss function (Mean Squared Error)
    criterion = nn.MSELoss()

    # Define the optimizer (Adam optimizer with learning rate 0.001)
    optimizer = optim.Adam(model.parameters(),lr=0.01)

    # Move the model to the specified device (CPU or GPU)
    model.to(device)

    # Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        # Move the batch to the appropriate device
        batch = batch.to(device)
        # Add an additional channel dimension to the batch tensor
        batch = batch.unsqueeze(1)

        # Forward pass
        outputs = model(batch)

        # Generate the ground truth numerical cost (Replace this with your evaluation function)
       
        ground_truth_costs = FakeTraversaibiltyEvalutaion(batch)

        # Compute the loss
        loss = criterion(outputs, ground_truth_costs.view(-1, 1))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Print the average loss for this epoch
    average_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}")








