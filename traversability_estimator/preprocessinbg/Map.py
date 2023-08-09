import utils

import numpy as np
import matplotlib.pyplot as plt
import cv2
import tf2_ros
import sensor_msgs.point_cloud2 as pc2
import rospy
from sensor_msgs.msg import CameraInfo
import tf
from cv_bridge import CvBridge
import torch
import time
import os


class LocalMap:
    def __init__(self, width, height, cell_size,data_dict):
        self.map_size = np.array([width, height, cell_size])
        self.map = np.zeros((int(width/cell_size), int(height/cell_size)))
        self.map_mask = np.zeros_like(self.map)
        self.fig, self.ax = plt.subplots()
        self.data_dict = data_dict

    def GetMapSize(self):
        return self.map_size
    
    def UpdateMap(self,data):
        assert data.shape == self.map.shape, "Shape of input data and shape of map are not matched!"
        self.map = data
    
    def UpdateFigure(self):
        self.im.set_data(self.map)     
        self.UpdateFigureAxis()   
        self.im.set_clim(vmin=np.min(self.map), vmax=np.max(self.map))  # Update the colorbar limits
        self.cbar.update_normal(self.im)  # Update the colorbar
        self.fig.canvas.draw_idle()
    
    def UpdateFigureAxis(self):
        # Compute the new tick positions and labels based on the cell_size
        tick_step = self.map_size[2]
        x_ticks = np.arange(-self.map.shape[1] / 2 * tick_step, self.map.shape[1] / 2 * tick_step, tick_step)
        y_ticks = np.arange(-self.map.shape[0] / 2 * tick_step, self.map.shape[0] / 2 * tick_step, tick_step)

        self.ax.set_xticks(np.arange(self.map.shape[1]) + 0.5, minor=False)
        self.ax.set_yticks(np.arange(self.map.shape[0]) + 0.5, minor=False)
        self.ax.set_xticklabels(x_ticks)
        self.ax.set_yticklabels(y_ticks)
        self.ax.set_xticklabels([f'{val:.2f}' for val in x_ticks])
        self.ax.set_yticklabels([f'{val:.2f}' for val in y_ticks]) 

    def show(self):
        plt.show()

class LidarMap(LocalMap):
    def __init__(self, width, height, cell_size,camera_map,data_dict):
        super().__init__(width, height, cell_size,data_dict)
        self.im = self.ax.imshow(self.map, cmap='viridis', interpolation='none', aspect='auto')
        self.cbar = self.fig.colorbar(self.im)
        self.ax.set_title('Local LiDAR Map')

        self.projected_map = np.zeros_like(self.map)
        self.map_cnt = np.zeros_like(self.map)
        
        self.camera_map = camera_map

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        try:
            self.Tv2b = self.tf_buffer.lookup_transform("velodyne", "base_link", rospy.Time(0), rospy.Duration(2.0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn("Could not get the base_link to velodyne transformation.")
            return
        
    def BEV_Projection(self,msg):
        self.camera_map.update_image_flag = False # Stop update image while Lidar callback!
        _, _, cell_size = self.map_size
        for point in pc2.read_points(msg, skip_nans=True):
            x, y, z_orig = point[:3]
            z = z_orig - self.Tv2b.transform.translation.z + 0.25 # -.25: chasis height
            # Project LIDAR points onto the local map without transformation
            grid_x_lidar = -int(x / cell_size) + self.map.shape[0]
            grid_y_lidar = -int(y / cell_size) + self.map.shape[1] // 2
            
            if 0 <= grid_x_lidar < self.map.shape[0] and 0 <= grid_y_lidar < self.map.shape[1]:
                self.projected_map[grid_x_lidar, grid_y_lidar] += z  # -0.25: z of baselink2lidar TF
                self.map_cnt[grid_x_lidar, grid_y_lidar] = self.map_cnt[grid_x_lidar, grid_y_lidar]+1
                lidar_point = np.array([x, y, z_orig , 1.0])
                self.camera_map.BEV_Projection(lidar_point)
        
        self.map_mask = (self.map_cnt != 0)
        self.map_cnt[self.map_cnt == 0] = 1.0  # Avoid division by zero
        self.projected_map = self.projected_map / self.map_cnt
        self.UpdateMap(self.projected_map)
        self.UpdateFigure()
        # print("Lidar map upadted!")
        self.projected_map = np.zeros_like(self.map)
        self.map_cnt = np.zeros_like(self.map)

        self.camera_map.map_mask = (self.camera_map.map_cnt != 0)
        self.camera_map.map_cnt[self.camera_map.map_cnt == 0] = 1.0  # Avoid division by zero
        self.camera_map.projected_map = self.camera_map.projected_map / self.camera_map.map_cnt
        self.camera_map.UpdateMap(self.camera_map.projected_map)
        self.camera_map.UpdateFigure()
        # print("Camera map upadted!")
        self.camera_map.projected_map = np.zeros_like(self.camera_map.map)
        self.camera_map.map_cnt = np.zeros_like(self.camera_map.map)

        self.data_dict['lidar_map'] = torch.tensor(self.map, dtype=torch.float32)
        self.data_dict['camera_map'] = torch.tensor(self.camera_map.map, dtype=torch.float32)
        self.data_dict['lidar_map_mask'] = torch.tensor(self.map_mask, dtype=torch.float32)
        self.data_dict['camera_map_mask'] = torch.tensor(self.camera_map.map_mask[:,:,1], dtype=torch.float32)


        self.camera_map.update_image_flag = True


class CameraMap(LocalMap):
    def __init__(self, width, height, cell_size,data_dict):
        super().__init__(width, height, cell_size,data_dict)
        self.im = self.ax.imshow(self.map,cmap='viridis')
        self.ax.set_title('Local Camera Map')
    
        self.map = np.zeros((int(width/cell_size), int(height/cell_size),3))
        self.map_mask = np.zeros((int(width/cell_size), int(height/cell_size)))
        self.projected_map = np.zeros_like(self.map)
        self.map_cnt = np.zeros_like(self.map)

        self.image = None
        self.update_flag = True
        self.update_image_flag = True

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        try:
            self.Tv2b = self.tf_buffer.lookup_transform("velodyne", "base_link", rospy.Time(0), rospy.Duration(2.0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn("Could not get the base_link to velodyne transformation.")
            return
        try:
            self.Tb2c = self.tf_buffer.lookup_transform("base_link", "left_camera_optical_frame", rospy.Time(0), rospy.Duration(2.0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn("Could not get the base_link to camera_link transformation.")
            return
        self.Tv2c  = utils.multiply_transformations(self.Tv2b, self.Tb2c)

        # Get the camera intrinsic parameters from the CameraInfo topic
        camera_info_topic = "/left_camera/camera_info"  # Replace with your actual camera_info topic
        self.camera_info_msg = rospy.wait_for_message(camera_info_topic, CameraInfo)
        self.fx = self.camera_info_msg.K[0]
        self.fy = self.camera_info_msg.K[4]
        self.cx = self.camera_info_msg.K[2]
        self.cy = self.camera_info_msg.K[5]

        self.bridge = CvBridge()

    def BEV_Projection(self,lidar_point):
         # Transform the point from the LiDAR frame to the camera frame
        camera_point = np.dot(lidar_point,self.Tv2c)
        x_camera = camera_point[0] / camera_point[3]
        y_camera = camera_point[1] / camera_point[3]
        z_camera = camera_point[2] / camera_point[3]
        # Project the transformed 3D point back to the camera image
        u = self.fx * ((x_camera )/ z_camera) + self.cx  # optical axis is different with camera axis!!! need to transform !!
        v = self.fy * (y_camera / z_camera) + self.cy
        
        # Check if the projected point is within the camera image boundaries
        if 0 <= u < self.camera_info_msg.width and 0 <= v < self.camera_info_msg.height:
            x,y,_,_ = lidar_point
            _, _, cell_size = self.map_size
            grid_x_lidar = -int(x / cell_size) + self.map.shape[0]
            grid_y_lidar = -int(y / cell_size) + self.map.shape[1] // 2

            if 0 <= grid_x_lidar < self.projected_map.shape[0] and 0 <= grid_y_lidar < self.projected_map.shape[1]:
                r, g, b = self.get_pixel_color_from_image(int(u), int(v))
                self.projected_map[grid_x_lidar, grid_y_lidar, 0] += r  # Red channel
                self.projected_map[grid_x_lidar, grid_y_lidar, 1] += g  # Green channel
                self.projected_map[grid_x_lidar, grid_y_lidar, 2] += b  # Blue channel
                self.map_cnt[grid_x_lidar, grid_y_lidar] = self.map_cnt[grid_x_lidar, grid_y_lidar]+1
    
    def UpdateFigure(self):
        self.im.set_data(self.map)
        self.UpdateFigureAxis()   
        self.fig.canvas.draw_idle()
    
    def UpdateMap(self,data):
        assert data.shape == self.map.shape, "Shape of input data and shape of map are not matched!"
        self.map = data/255.0

    def UpdateImage(self, msg):
        if self.update_image_flag == True:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.image = cv_image
            self.update_flag = True

    def get_pixel_color_from_image(self, x, y):
        if self.image is None:
            return 0, 0, 0  
        b, g, r = self.image[y, x]  
        return r,g,b
    
    
    

class TraversabilityMap(LocalMap):
    def __init__(self, width, height, cell_size,data_dict):
        super().__init__(width, height, cell_size,data_dict)
        self.im = self.ax.imshow(self.map, cmap='viridis', interpolation='none', aspect='auto')
        self.cbar = self.fig.colorbar(self.im)
        self.ax.set_title('Traversability Map')

        current_time = time.strftime("%Y%m%d_%H%M%S")
        # Create a folder with the current time as the name
        self.folder_name = '../input/data/'+current_time
        os.makedirs(self.folder_name, exist_ok=True)

        self.update_cnt = 0

    def CalculateMap(self,msg):
        info = msg.info
        yaw = 0.0
        self.position = [info.pose.position.x , info.pose.position.y]
        length_x = info.length_x
        length_y = info.length_y
        resolution =info.resolution
        map_shape = [length_x/resolution, length_y/resolution]
        grid_map = np.array(msg.data[9].data) # pick "traversability cost layer" from multiple layers
        grid_map = grid_map.reshape( int(map_shape[1]), int(map_shape[0]))
        grid_map=utils.rotate_grid(grid_map,(yaw*180.0/3.141592)+90)
        grid_map = np.flipud(grid_map) # traversability map is initially filped... really bad code..
        grid_map = np.roll(grid_map, +3, axis=0) # traversability map and lidar,camera alignment correction... really bad code..

        width, height, _ = self.map_size
        grid_map_crop = utils.zoom_grid(grid_map,width/length_x ,height/length_y)

        if grid_map_crop.shape == (20,20):
            self.map_mask = ~np.isnan(grid_map_crop)
            self.UpdateMap(grid_map_crop)
            self.UpdateFigure()
            
            current_time = time.strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(self.folder_name, f'{current_time}.pt')

            self.data_dict['traversability_map'] = torch.tensor(self.map, dtype=torch.float32)
            self.data_dict['traversability_map_mask'] = torch.tensor(self.map_mask, dtype=torch.float32)
            
            if self.update_cnt > 5:
                torch.save(self.data_dict, file_path)
                print("dataset saved! :",file_path)

            self.update_cnt = self.update_cnt +1


        else:
            print("traversability map shape error: ", grid_map_crop.shape)
        
        return
    


