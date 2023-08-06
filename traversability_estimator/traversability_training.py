from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim

class FakeLidarDataset(Dataset):
    def __init__(self):
        self.height = 48
        self.width = 64
        self.data_size = 200
        self.samples = []
        for i in range(self.data_size):
            sample = torch.rand(self.width,self.height)
            self.samples.append(sample)
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
def FakeTraversaibiltyEvalutaion(batch):
    sum_per_sample = torch.sum(batch, dim=(2, 3))  # Sum along height and width dimensions
    # Reshape the sum tensor to have shape (batch_size, 1)
    sum_per_sample = sum_per_sample.view(-1, 1)
    return sum_per_sample
    

    

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
    dataset = FakeLidarDataset()
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








