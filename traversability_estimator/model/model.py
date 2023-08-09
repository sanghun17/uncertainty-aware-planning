import torch
import torch.nn as nn
import torch.nn.functional as F

features = 64
# define a simple linear VAE
class LinearVAE(nn.Module):
    def __init__(self):
        super(LinearVAE, self).__init__()
 
        # encoder
        self.conv_layer1 = nn.Conv2d(in_channels=5, out_channels=32, kernel_size=5, stride=2,padding=1)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2,padding=1)
        # self.conv_layer3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2)
        
        # Calculate the spatial dimensions after applying the convolutions
        self.fc1 = nn.Linear(1600, 256)
        self.fc2 = nn.Linear(256, 128)
        # decoder 
        self.dec1 = nn.Linear(in_features=features, out_features=512)
        self.dec2 = nn.Linear(in_features=512, out_features=20*20*5)

    
    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample
 
    def forward(self, x):
        # encoding
        # Apply convolutional layers
        x = F.relu(self.conv_layer1(x))
        x = F.relu(self.conv_layer2(x))
        # x = F.relu(self.conv_layer3(x))
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 2, features)
        # get `mu` and `log_var`
        mu = x[:, 0, :] # the first feature values as mean
        log_var = x[:, 1, :] # the other feature values as variance
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
 
        # decoding
        x = F.relu(self.dec1(z))
        # print("x:",x)
        reconstruction = torch.sigmoid(self.dec2(x))
        # print("reconstruction:",reconstruction)
        # print("mu:",mu)
        # print("log_var:",log_var)
        return reconstruction, mu, log_var