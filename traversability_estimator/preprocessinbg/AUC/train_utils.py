import numpy as np
import torch
import matplotlib.pyplot as plt

def animate_result(batch_images_np = None,outputs_np = None):
        if torch.is_tensor(batch_images_np):
            batch_images_np = batch_images_np.cpu().numpy()
            
        if torch.is_tensor(outputs_np):
            outputs_np = outputs_np.cpu().numpy()     
            
        # Create a figure with subplots for each image pair
        fig, axes = plt.subplots(2, batch_images_np.shape[2],figsize=( 4 * batch_images_np.shape[2],8))
        
        for j in range(1):
            axes[0, j].imshow(batch_images_np[:,:,j], cmap='gray')
            axes[0, j].set_title('Ground Truth')
            
            axes[1, j].imshow(outputs_np[:,:,j], cmap='gray')
            axes[1, j].set_title('Predicted Output')
        
        plt.tight_layout()