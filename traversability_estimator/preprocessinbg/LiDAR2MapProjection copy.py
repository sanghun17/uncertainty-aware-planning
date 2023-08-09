
import tf2_ros
from geometry_msgs.msg import TransformStamped
import sensor_msgs.point_cloud2 as pc2
import seaborn as sns
from matplotlib.transforms import Affine2D
from matplotlib.image import AxesImage
import matplotlib
from cv_bridge import CvBridge

from scipy.ndimage import rotate
from scipy.ndimage import zoom
import torch
from datetime import datetime
import os
import time
from torchvision.utils import save_image


# TODO: 
# camera bev projection does not work well about near object.
# about dataset save, detect anormaly. (some map is stopped to be updated.)



class LocalMap:
    def __init__(self, width, height, cell_size):
        self.map_shape = np.array([width, height, cell_size])
        self.lidar_map = np.zeros((int(width/cell_size), int(height/cell_size)))
        self.camera_map = np.zeros((int(width/cell_size), int(height/cell_size),3))

        self.fig1, self.ax1 = plt.subplots()
        self.ax1.set_title('Local LiDAR Map')  # Add a title for the plot
        self.im = self.ax1.imshow(self.lidar_map, cmap='viridis', interpolation='none', aspect='auto')
        self.cbar = self.fig1.colorbar(self.im)
        self.cell_size = cell_size

        self.fig2, self.ax2 = plt.subplots()
        self.ax2.set_title('Local Camera Map')  # Add a title for the plot
        self.im2 = self.ax2.imshow(self.camera_map,cmap='viridis')

    def GetMapShape(self):
        return self.map_shape
    
    def GetCellSize(self):
        return self.cell_size
    
    def UpdateLiDARMap(self, data):
        assert data.shape == self.lidar_map.shape, "Shape of input data and shape of map are not matched!"
        self.lidar_map = data
        self.im.set_data(self.lidar_map)  # Update the heatmap data
        self.im.set_clim(vmin=np.min(self.lidar_map), vmax=np.max(self.lidar_map))  # Update the colorbar limits

        # Compute the new tick positions and labels based on the cell_size
        constant_multiplier = self.cell_size
        x_ticks = np.arange(width/2-cell_size, -width/2-cell_size, -constant_multiplier)
        y_ticks = np.arange(height/2-cell_size, -height/2-cell_size, -constant_multiplier)

        self.ax1.set_xticks(np.arange(self.lidar_map.shape[1]) + 0.5, minor=False)
        self.ax1.set_yticks(np.arange(self.lidar_map.shape[0]) + 0.5, minor=False)
        self.ax1.set_xticklabels(x_ticks)
        self.ax1.set_yticklabels(y_ticks)
        self.ax1.set_xticklabels([f'{val:.2f}' for val in x_ticks])
        self.ax1.set_yticklabels([f'{val:.2f}' for val in y_ticks])

        data_dict['lidar_map'] = torch.tensor(self.lidar_map, dtype=torch.float32)

        
        self.cbar.update_normal(self.im)  # Update the colorbar
        self.fig1.canvas.draw_idle()

    def UpdateCameraMap(self, data):
        assert data.shape == self.camera_map.shape, "Shape of input data and shape of map are not matched!"
        self.camera_map = data/255.0
        self.im2.set_data(self.camera_map)  

        # Compute the new tick positions and labels based on the cell_size
        constant_multiplier = self.cell_size
        x_ticks = np.arange(width/2-cell_size, -width/2-cell_size, -constant_multiplier)
        y_ticks = np.arange(height/2-cell_size, -height/2-cell_size, -constant_multiplier)

        self.ax2.set_xticks(np.arange(self.camera_map.shape[1]) + 0.5, minor=False)
        self.ax2.set_yticks(np.arange(self.camera_map.shape[0]) + 0.5, minor=False)
        self.ax2.set_xticklabels(x_ticks)
        self.ax2.set_yticklabels(y_ticks)

        data_dict['camera_map'] = torch.tensor(self.camera_map, dtype=torch.float32)
        
        self.fig2.canvas.draw_idle()

    def show(self):
        plt.show()

class GridMapp:
    def __init__(self,width, height,robot_pose_handler):
        self.resolution = 0.5
        self.width = width
        self.height = height
        self.grid_map_crop = np.zeros((int(width/self.resolution), int(height/self.resolution)))
        self.robot_pose_handler = robot_pose_handler
        self.fig1, self.ax1 = plt.subplots()
        self.ax1.set_title('Local Traversability Map')  # Add a title for the plot
        self.im = self.ax1.imshow(self.grid_map_crop, cmap='viridis', interpolation='none', aspect='auto')
        self.cbar = self.fig1.colorbar(self.im)


        current_time = time.strftime("%Y%m%d_%H%M%S")

        # Create a folder with the current time as the name
        self.folder_name = '../input/data/'+current_time
        os.makedirs(self.folder_name, exist_ok=True)

    def GetResolution(self):
        return self.resolution
    
    def UpdateGridMap(self, msg):
        x, y, yaw = self.robot_pose_handler.GetPose()
        # print(self.length_x )
        info = msg.info
        self.position = [info.pose.position.x , info.pose.position.y]
        length_x = info.length_x
        length_y = info.length_y
        self.resolution =info.resolution
        map_shape = [length_x/self.resolution, length_y/self.resolution]
        grid_map = np.array(msg.data[9].data)
        grid_map = grid_map.reshape( int(map_shape[1]), int(map_shape[0]))
        grid_map=self.rotate_array(grid_map,(yaw*180.0/3.141592)+90)
        grid_map = np.flipud(grid_map)
        grid_map = np.roll(grid_map, +3, axis=0)
        self.grid_map_crop = self.grid_zoom(grid_map,self.width/length_x ,self.height/length_y)

        self.im.set_data(self.grid_map_crop)  # Update the heatmap data
        self.im.set_clim(vmin=np.min(self.grid_map_crop), vmax=np.max(self.grid_map_crop))  # Update the colorbar limits

        # Compute the new tick positions and labels based on the cell_size
        constant_multiplier = self.resolution
        x_ticks = np.arange(self.width/2-constant_multiplier, -self.width/2-constant_multiplier, -constant_multiplier)
        y_ticks = np.arange(self.height/2-constant_multiplier, -self.height/2-constant_multiplier, -constant_multiplier)

        self.ax1.set_xticks(np.arange(self.grid_map_crop.shape[1]) + 0.5, minor=False)
        self.ax1.set_yticks(np.arange(self.grid_map_crop.shape[0]) + 0.5, minor=False)
        self.ax1.set_xticklabels(x_ticks)
        self.ax1.set_yticklabels(y_ticks)
        self.ax1.set_xticklabels([f'{val:.2f}' for val in x_ticks])
        self.ax1.set_yticklabels([f'{val:.2f}' for val in y_ticks])
        
        data_dict['grid_map'] = torch.tensor(self.grid_map_crop, dtype=torch.float32)
        current_time = time.strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(self.folder_name, f'{current_time}.pt')
        grid_map = data_dict['grid_map']
        
        # Ensure that all maps have the same spatial dimensions (20x20)
        if grid_map.size()  == (20, 20) :
            torch.save(data_dict, file_path)
            print("dataset saved! :",file_path)

        self.cbar.update_normal(self.im)  # Update the colorbar
        self.fig1.canvas.draw_idle()
        return
    
    def show(self):
        plt.show()

    def rotate_array(self,data_2d, angle_deg):
        center = (np.array(data_2d.shape)+np.array([0.0, 0.0])) / 2.0
        rotation_matrix = cv2.getRotationMatrix2D(tuple(center[::-1]), angle_deg, scale=1.0)

        # Perform the rotation using warpAffine with interpolation
        rotated_array = cv2.warpAffine(data_2d, rotation_matrix, data_2d.shape[::-1], flags=cv2.INTER_LINEAR)
        return rotated_array
    
    def grid_zoom(self,data_2d,zoom_factor_x,zoom_factor_y):
      crop_rows = int(data_2d.shape[0] * (1 - zoom_factor_y))
      crop_cols = int(data_2d.shape[1] * (1 - zoom_factor_x))
      # Crop the array to remove the outer rows and columns
      cropped_array = data_2d[crop_rows//2 : -crop_rows//2, crop_cols//2 : -crop_cols//2]
    #   print(cropped_array)
      return cropped_array

class LiDAR2MapProjection:
    def __init__(self, local_map):
        self.local_map = local_map
        self.image = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Get the transformation from "base_link" to "camera_link"
        try:
            self.Tb2c = self.tf_buffer.lookup_transform("base_link", "left_camera_optical_frame", rospy.Time(0), rospy.Duration(2.0))
            print(self.Tb2c)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn("Could not get the base_link to camera_link transformation.")
            return

        try:
            self.Tv2b = self.tf_buffer.lookup_transform("velodyne", "base_link", rospy.Time(0), rospy.Duration(2.0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn("Could not get the base_link to velodyne transformation.")
            return
        
        self.Tv2c  = self.combine_transformations(self.Tv2b, self.Tb2c)
        print("self.Tv2c ",self.Tv2c)

        # Get the camera intrinsic parameters from the CameraInfo topic
        camera_info_topic = "/left_camera/camera_info"  # Replace with your actual camera_info topic
        self.camera_info_msg = rospy.wait_for_message(camera_info_topic, CameraInfo)
        self.fx = self.camera_info_msg.K[0]
        self.fy = self.camera_info_msg.K[4]
        self.cx = self.camera_info_msg.K[2]
        self.cy = self.camera_info_msg.K[5]

        self.bridge = CvBridge()

    def Projection(self, msg):
        if self.Tv2c is None:
            rospy.logwarn("Could not transform LIDAR points to camera frame.")
            return

        map_width, map_height, cell_size = self.local_map.GetMapShape()
        projected_map_lidar = np.zeros((int(map_width/cell_size), int(map_height/cell_size)))
        projected_map_camera = np.zeros((int(map_width/cell_size), int(map_height/cell_size),3))
        map_cnt_lidar = np.zeros((int(map_width/cell_size), int(map_height/cell_size)))
        map_cnt_camera = np.zeros((int(map_width/cell_size), int(map_height/cell_size)))

        for point in pc2.read_points(msg, skip_nans=True):
            x, y, z_orig = point[:3]
            z = z_orig - self.Tv2b.transform.translation.z + 0.25 # -.25: chasis height
            # Project LIDAR points onto the local map without transformation
            grid_x_lidar = -int(x / cell_size) + projected_map_lidar.shape[0]
            grid_y_lidar = -int(y / cell_size) + projected_map_lidar.shape[1] // 2
            
            if 0 <= grid_x_lidar < projected_map_lidar.shape[0] and 0 <= grid_y_lidar < projected_map_lidar.shape[1]:
                projected_map_lidar[grid_x_lidar, grid_y_lidar] += z  # -0.25: z of baselink2lidar TF
                map_cnt_lidar[grid_x_lidar, grid_y_lidar] = map_cnt_lidar[grid_x_lidar, grid_y_lidar]+1
                # Transform the point from the LiDAR frame to the camera frame
                lidar_point = np.array([x, y, z_orig , 1.0])
                camera_point = np.dot(lidar_point,self.Tv2c)
                x_camera = camera_point[0] / camera_point[3]
                y_camera = camera_point[1] / camera_point[3]
                z_camera = camera_point[2] / camera_point[3]

                # Project the transformed 3D point back to the camera image
                u = self.fx * ((x_camera )/ z_camera) + self.cx  # optical axis is different with camera axis!!! need to transform !!
                v = self.fy * (y_camera / z_camera) + self.cy

                # Check if the projected point is within the camera image boundaries
                if 0 <= u < self.camera_info_msg.width and 0 <= v < self.camera_info_msg.height:
                    # print("u, v: ", u,v)
                    r, g, b = self.get_pixel_color_from_image(int(u), int(v))  # Implement this method to extract color from image
                    if 0 <= grid_x_lidar < projected_map_camera.shape[0] and 0 <= grid_y_lidar < projected_map_camera.shape[1]:
                        projected_map_camera[grid_x_lidar, grid_y_lidar, 0] += r  # Red channel
                        projected_map_camera[grid_x_lidar, grid_y_lidar, 1] += g  # Green channel
                        projected_map_camera[grid_x_lidar, grid_y_lidar, 2] += b  # Blue channel
                        map_cnt_camera[grid_x_lidar, grid_y_lidar] = map_cnt_camera[grid_x_lidar, grid_y_lidar]+1
                        # print(r, g, b)
        # Divide the values in projected_map_lidar by the count in map_cnt_lidar
        map_cnt_lidar[map_cnt_lidar == 0] = 1.0  # Avoid division by zero
        projected_map_lidar = projected_map_lidar / map_cnt_lidar

        # Divide the values in projected_map_camera by the count in map_cnt_camera
        map_cnt_camera[map_cnt_camera == 0] = 1.0  # Avoid division by zero
        projected_map_camera = projected_map_camera / map_cnt_camera[:, :, np.newaxis]

        # Update the map using the UpdateMap method of the LocalMap object
        self.local_map.UpdateLiDARMap(projected_map_lidar)
        self.local_map.UpdateCameraMap(projected_map_camera)

    def UpdateImage(self, msg):
        # np_arr = np.frombuffer(msg.data, np.uint8)
        # cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.image = cv_image
        # (rows,cols,channels) = cv_image.shape
        # print(rows,cols,channels)
        # cv2.imshow("Image window", cv_image)

    def get_pixel_color_from_image(self, x, y):
        if self.image is None:
            return 0, 0, 0  # Return default color if the image is not available

        # Get the color (r, g, b) values from the image at the given pixel position (x, y)
        b, g, r = self.image[y, x]  # Note the order of y and x for indexing

        return r,g,b
    
    def combine_transformations(self, transform1, transform2):
        # Convert TransformStamped objects to 4x4 homogeneous transformation matrices
        matrix1 = tf.transformations.quaternion_matrix([transform1.transform.rotation.x,
                                                        transform1.transform.rotation.y,
                                                        transform1.transform.rotation.z,
                                                        transform1.transform.rotation.w])
        matrix1[:3, 3] = [transform1.transform.translation.x,
                        transform1.transform.translation.y,
                        transform1.transform.translation.z]

        matrix2 = tf.transformations.quaternion_matrix([transform2.transform.rotation.x,
                                                        transform2.transform.rotation.y,
                                                        transform2.transform.rotation.z,
                                                        transform2.transform.rotation.w])
        matrix2[:3, 3] = [transform2.transform.translation.x,
                        transform2.transform.translation.y,
                        transform2.transform.translation.z]

        # Set the bottom row to [0, 0, 0, 1] for both matrices
        matrix1[3, :] = [0, 0, 0, 1]
        matrix2[3, :] = [0, 0, 0, 1]

        # Perform matrix multiplication
        combined_matrix = np.dot(matrix1, matrix2)

        return combined_matrix


if __name__ == '__main__':
    rospy.init_node('lidar_map_projection_node')
    plt.switch_backend('Qt5Agg')
    robot_pose_handler = RobotPose() 
    width, height = 10,10 
    grid_map = GridMapp(width, height,robot_pose_handler)
    cell_size = 0.5
    local_map = LocalMap(width, height, cell_size) 
    lidar_projection = LiDAR2MapProjection(local_map) 
    
    rospy.Subscriber('/velodyne_points', PointCloud2, lidar_projection.Projection) 
    rospy.Subscriber('/ground_truth/state', Odometry, robot_pose_handler.update_pose)
    rospy.Subscriber('/left_camera/image_raw',Image, lidar_projection.UpdateImage)
    rospy.Subscriber('/traversability_estimation/traversability_map',GridMap, grid_map.UpdateGridMap)

    local_map.show()
    grid_map.show()

    rospy.spin()
