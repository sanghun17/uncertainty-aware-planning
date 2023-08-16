import torch
from torch.utils.data import Dataset
import os
import numpy as np
import transformations
import matplotlib.pyplot as plt


class GATDataset_fake(Dataset):    
    def __init__(self, state_input, action_input,img_input, img_output):
        self.img_input = img_input
        self.state_input = state_input
        self.action_input = action_input
        self.img_output = img_output
        
        assert len(self.img_input) == len(self.state_input) == len(self.action_input) == len(self.img_output), "All input and output must have the same length"

    # def collate_fn(self, batch):
    def __len__(self):
        return len(self.img_input)

    def __getitem__(self, idx):
        img_input = self.img_input[idx]
        state_input = self.state_input[idx]
        action_input = self.action_input[idx]
        img_output = self.img_output[idx]
        # print(action_input.size)
        return state_input, action_input,img_input,  img_output
    

class GATDataset(Dataset):    
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.data_files = []
        self.fx = 381.36246688113556
        self.fy = 381.36246688113556
        self.cx = 320.5
        self.cy = 381.36246688113556
        self.T_baselink2camera = np.array([[0, 0, 1, 0],
                                          [-1, 0, 0, 0],
                                          [0, -1, 0, 0],
                                          [0, 0, 0, 1]])

        # Iterate through all subfolders inside the main data folder
        for subfolder in os.listdir(self.data_folder):
            subfolder_path = os.path.join(self.data_folder, subfolder)

            # Check if the item in the data folder is a subfolder
            if os.path.isdir(subfolder_path):
                # Get all the data files inside the subfolder
                subfolder_files = [os.path.join(subfolder_path, filename) for filename in os.listdir(subfolder_path)]
                self.data_files.extend(subfolder_files)
            self.data_files = self.data_files[3:] # remove first, and last 3 files (0.5s recording step, 1s window)

    def __len__(self):
        return len(self.data_files)



    def __getitem__(self, idx):
        file_path = self.data_files[idx]
        data_dict = torch.load(file_path)
        camera_image = data_dict['RGB_image'] # image at t=k
        camera_image = camera_image/255 # normalize
        camera_image = camera_image[:, :, [2, 1, 0]] # channel correction..
        depth_image = data_dict['depth_image'] # image at t=k
        state = data_dict['state'] # state at t=k:k+10 (vx,vy,wz,roll, ptich)
        true_position = data_dict['true_position'] # true position at t=k:k+10 (x, y, z, roll,pitch,yaw)
        action = data_dict['action'] # control input at t=k:k+10 (vx, steering)
        output_image_list=self.ApplyAtentionInput(camera_image,true_position,0.1,action)

        # plt.close()
        # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        # axes[0].imshow(depth_image, cmap='gray')
        # axes[0].set_title('Depth Image')
        # axes[1].imshow(camera_image)
        # axes[1].set_title('Camera Image')
        # plt.tight_layout()
        # plt.show()

        return state, action, camera_image, output_image_list

    def ApplyAtentionInput(self,image, true_position,patch_size_in_meter,action):
        x0 = true_position[0,:]
        # T_translation = transformations.translation_matrix(x0[:3])
        T_translation = transformations.translation_matrix([0,0,0])
        T_rotation = transformations.euler_matrix(x0[3], x0[4], x0[5], 'sxyz')
        T_rotation[:3,:3] = T_rotation[:3,:3]/np.linalg.norm(T_rotation[:3,:3])
        T_world2baselink0 = T_translation @ T_rotation
        T_world2camera0 = T_world2baselink0 @ self.T_baselink2camera
        image_list = []
        for i,x in enumerate(true_position[1:]): # skip first since x is same with x0x
                image_copy = image.clone()
                del_x_world = x - x0
                del_x_world = (del_x_world[0], del_x_world[1], del_x_world[2],1)
                del_x_baselink = del_x_world@T_world2baselink0
                cur_action = action[i+1,:]
                del_x_camera = del_x_baselink @ self.T_baselink2camera
                # del_x_camera = del_x_camera/del_x_camera[-1]
                u = (self.fx * del_x_camera[0] / del_x_camera[2]) + self.cx
                v = (self.fy * del_x_camera[1] / del_x_camera[2]) + self.cy
                bound_pixel_x = patch_size_in_meter * self.fx / del_x_camera[2]  # get pixel length from given patch_size in meter and z
                bound_pixel_y = patch_size_in_meter * self.fy / del_x_camera[2]  # get pixel length from given patch_size in meter and z
                attention_patch=self.drawrectangular(image_copy, u, v, bound_pixel_x,bound_pixel_y)  # draw rectangle centered at u, v, length of bound pixel
                image_list.append(attention_patch)

                image_tensor = torch.stack(image_list, dim=0)
        # image_tensor_test = image_tensor.permute(0,2,3,1)
        # image_tensor_test=np.reshape(image_tensor_test, (-1, image_tensor_test.shape[2], image_tensor_test.shape[3]))
        # plt.imshow(image_tensor_test, cmap='gray')
        # plt.show()
        
        return image_tensor
        
    def drawrectangular(self,image, u, v, width, height):
        # Create a mask with zeros (black image)
        size =image.size()
        attention_patch = torch.zeros(1,size[0],size[1])
        
        # Calculate rectangle corners
        half_bound_x = width // 2
        half_bound_y = height // 2
        top_left = (int(u - half_bound_x), int(v - half_bound_y))
        bottom_right = (int(u + half_bound_x), int(v + half_bound_y))
        top_left = (max(top_left[0], 0), max(top_left[1], 0))
        bottom_right = (min(bottom_right[0], size[0]), min(bottom_right[1], size[1]))
        
        # Set the region inside the rectangle to ones (white)
        attention_patch[:,top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] = 1
        return attention_patch
            
