import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from nav_msgs.msg import Odometry
import tf2_ros
from geometry_msgs.msg import TransformStamped
import sensor_msgs.point_cloud2 as pc2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from matplotlib.transforms import Affine2D
from matplotlib.image import AxesImage
import matplotlib
import cv2
from cv_bridge import CvBridge
import tf.transformations
import tf
matplotlib.use('Qt5Agg')  # Set the backend to Qt5Agg

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
        # self.im.set_clim(vmin=np.min(self.lidar_map), vmax=1.0)  # Update the colorbar limits

        # Compute the new tick positions and labels based on the cell_size
        constant_multiplier = self.cell_size
        x_ticks = np.arange(0, self.lidar_map.shape[1] + 1, constant_multiplier)
        y_ticks = np.arange(0, self.lidar_map.shape[0] + 1, constant_multiplier)

        self.ax1.set_xticks(np.arange(self.lidar_map.shape[1]) + 0.5, minor=False)
        self.ax1.set_yticks(np.arange(self.lidar_map.shape[0]) + 0.5, minor=False)
        self.ax1.set_xticklabels(x_ticks)
        self.ax1.set_yticklabels(y_ticks)
        
        self.cbar.update_normal(self.im)  # Update the colorbar
        self.fig1.canvas.draw_idle()

    def UpdateCameraMap(self, data):
        assert data.shape == self.camera_map.shape, "Shape of input data and shape of map are not matched!"
        self.camera_map = data/255.0
        self.im2.set_data(self.camera_map)  # Update the heatmap data
        # print(self.camera_map.shape)

        # Compute the new tick positions and labels based on the cell_size
        constant_multiplier = self.cell_size
        x_ticks = np.arange(0, self.camera_map.shape[1] + 1, constant_multiplier)
        y_ticks = np.arange(0, self.camera_map.shape[0] + 1, constant_multiplier)

        self.ax2.set_xticks(np.arange(self.camera_map.shape[1]) + 0.5, minor=False)
        self.ax2.set_yticks(np.arange(self.camera_map.shape[0]) + 0.5, minor=False)
        self.ax2.set_xticklabels(x_ticks)
        self.ax2.set_yticklabels(y_ticks)
        
        self.fig2.canvas.draw_idle()

    def PlotLiDAROnOtherFigure(self, ax2, position, angle):
        # Get the heatmap data from the local_map
        heatmap_data = self.lidar_map
        # Remove the previous heatmap on ax2 if it exists
        for artist in ax2.get_children():
            if isinstance(artist, AxesImage) and artist.get_cmap().name == 'viridis':
                artist.remove()
        # Calculate the dimensions of the heatmap
        map_width, map_height = self.map_shape[0], self.map_shape[1]
        # Calculate the center of the heatmap
        center_x,center_y = position[0],position[1]
        # Calculate the extent of the heatmap
        extent = [center_x - map_width / 2, center_x + map_width / 2, center_y - map_height / 2, center_y + map_height / 2]
        # Plot the heatmap on ax2 using imshow with additional transformations
        im = ax2.imshow(heatmap_data, cmap='viridis', extent=extent, interpolation='none')
        # Apply the transformations to the image
        trans = Affine2D().rotate_around(center_x, center_y, angle) + ax2.transData
        im.set_transform(trans)
        ax2.set_title('Heatmap on Another Figure')  # Add a title for the plot

    # sdef PlotCameraROnOtherFigure(self, ax2, position, angle):
    #     # Get the heatmap data from the local_map
    #     heatmap_data = self.camera_map
    #     # Remove the previous heatmap on ax2 if it exists
    #     for artist in ax2.get_children():
    #         if isinstance(artist, AxesImage) and artist.get_cmap().name == 'viridis':
    #             artist.remove()
    #     # Calculate the dimensions of the heatmap
    #     map_width, map_height = self.map_shape[0], self.map_shape[1]
    #     # Calculate the center of the heatmap
    #     center_x,center_y = position[0],position[1]
    #     # Calculate the extent of the heatmap
    #     extent = [center_x - map_width / 2, center_x + map_width / 2, center_y - map_height / 2, center_y + map_height / 2]
    #     # Plot the heatmap on ax2 using imshow with additional transformations
    #     im = ax2.imshow(heatmap_data, cmap='viridis', extent=extent, interpolation='none')
    #     # Apply the transformations to the image
    #     trans = Affine2D().rotate_around(center_x, center_y, angle) + ax2.transData
    #     im.set_transform(trans)
    #     ax2.set_title('Heatmap on Another Figure')  # Add a title for the plot

    def show(self):
        plt.show()

class LiDAR2MapProjection:
    def __init__(self, local_map):
        self.local_map = local_map
        self.image = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Get the transformation from "base_link" to "camera_link"
        try:
            self.Tb2c = self.tf_buffer.lookup_transform("base_link", "camera_link", rospy.Time(0), rospy.Duration(2.0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn("Could not get the base_link to camera_link transformation.")
            return

        try:
            self.Tv2b = self.tf_buffer.lookup_transform("velodyne_base_link", "base_link", rospy.Time(0), rospy.Duration(2.0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn("Could not get the base_link to velodyne_base_link transformation.")
            return
        
        # Combine the transformations to get the LiDAR to camera transformation
        # First, transform from "base_link" to "camera_link" and store the result in the variable `self.transform_base_to_camera`
        # self.Tb2c  = tf2_geometry_msgs.do_transform_pose(tf2_geometry_msgs.PoseStamped(), self.Tb2c )
        # Next, transform from "velodyne_base_link" to "base_link" and store the result in the variable `self.transform_base_to_lidar`
        # self.Tv2b  = tf2_geometry_msgs.do_transform_pose(tf2_geometry_msgs.PoseStamped(), self.Tv2b)
        self.Tv2c  = self.combine_transformations(self.Tv2b, self.Tb2c)
        # print("self.Tv2c ")


        # Finally, combine the transformations to get the LiDAR to camera transformation
        # write code here

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
            x, y, z = point[:3]
            z = -z

            # Project LIDAR points onto the local map without transformation
            grid_x_lidar = int(x / cell_size) + projected_map_lidar.shape[0] // 2
            grid_y_lidar = int(y / cell_size) + projected_map_lidar.shape[1] // 2

            if 0 <= grid_x_lidar < projected_map_lidar.shape[0] and 0 <= grid_y_lidar < projected_map_lidar.shape[1]:
                projected_map_lidar[grid_x_lidar, grid_y_lidar] += (z + self.Tv2b.transform.translation.z)  # -0.25: z of baselink2lidar TF
                map_cnt_lidar[grid_x_lidar, grid_y_lidar] = map_cnt_lidar[grid_x_lidar, grid_y_lidar]+1
                # Transform the point from the LiDAR frame to the camera frame
                lidar_point = np.array([x, y, z, 1.0])  # Homogeneous coordinates
                camera_point = np.dot(self.Tv2c, lidar_point)
                x_camera = camera_point[0]
                y_camera = camera_point[1]
                z_camera = camera_point[2]



                # Project the transformed 3D point back to the camera image plane
                # u = self.fx * (x_camera / z_camera) + self.cx
                # v = self.fy * (y_camera / z_camera) + self.cy
                u = self.fx * (-y_camera / x_camera) + self.cx  # optical axis is different with camera axis!!! need to transform !!
                v = self.fy * (-z_camera / x_camera) + self.cy

                # print("x,y,z: ",x,y,z)
                # print("c x,y,z: ",x_camera,y_camera,z_camera)
                # print("u, v: ", u,v)

                    
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
        # print("map_cnt_lidar: ",map_cnt_lidar)
        map_cnt_lidar[map_cnt_lidar == 0] = 1.0  # Avoid division by zero
        # print("b projected_map_lidar: ",projected_map_lidar)
        projected_map_lidar = projected_map_lidar / map_cnt_lidar
        # print("a projected_map_lidar: ",projected_map_lidar)

        # Divide the values in projected_map_camera by the count in map_cnt_camera
        # print("map_cnt_camera: ",map_cnt_camera)
        map_cnt_camera[map_cnt_camera == 0] = 1.0  # Avoid division by zero
        # print("b projected_map_camera: ",projected_map_camera)
        projected_map_camera = projected_map_camera / map_cnt_camera[:, :, np.newaxis]
        # print("a projected_map_camera: ",projected_map_camera)

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
        cv2.imshow("Image window", cv_image)

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

    
class RobotPose:    
    def __init__(self):
        self.robot_pose = (0, 0)
        self.robot_yaw = 0.0

    def update_pose(self, odom_msg):
        # Get the position from the Odometry message
        self.robot_pose = (odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y)

        # Get the orientation (quaternion) from the Odometry message
        quaternion = (
            odom_msg.pose.pose.orientation.x,
            odom_msg.pose.pose.orientation.y,
            odom_msg.pose.pose.orientation.z,
            odom_msg.pose.pose.orientation.w
        )
        # Convert quaternion to RPY (roll, pitch, yaw) angles
        _, _, self.robot_yaw = tf.transformations.euler_from_quaternion(quaternion)

    def GetPose(self):
        return self.robot_pose[0], self.robot_pose[1], self.robot_yaw

class Plotter:
    def __init__(self, local_map, robot_pose_handler):
        self.local_map = local_map
        self.robot_pose_handler = robot_pose_handler
        self.max_x = 0
        self.min_x = 0
        self.max_y = 0
        self.min_y = 0
        local_map_shape=local_map.GetMapShape()
        self.local_map_width, self.local_map_height = local_map_shape[0], local_map_shape[1]

        # Create the map plot
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('Map Plot')
        self.pose_arrow = self.ax.arrow(0, 0, 0, 0, head_width=0.1, head_length=0.15, fc='red', ec='red')

    def update_pose(self):
        # Update the robot pose arrow
        arrow_length = 0.5
        x, y, yaw = self.robot_pose_handler.GetPose()
        arrow_dx = arrow_length * np.cos(yaw)
        arrow_dy = arrow_length * np.sin(yaw)

        # Remove the previous arrow
        if hasattr(self, 'pose_arrow'):
            self.pose_arrow.remove()

        # Draw the new arrow
        self.pose_arrow = self.ax.arrow(x, y, arrow_dx, arrow_dy, head_width=0.2, head_length=0.15, fc='red', ec='red')

        # Update the plot limits based on the robot pose
        self.max_x = max(self.max_x, x + arrow_dx)
        self.min_x = min(self.min_x, x + arrow_dx)
        self.max_y = max(self.max_y, y + arrow_dy)
        self.min_y = min(self.min_y, y + arrow_dy)
        self.ax.set_xlim(self.min_x - 3, self.max_x + 3)
        self.ax.set_ylim(self.min_y - 3, self.max_y + 3)

        self.fig.canvas.draw_idle()

    def update_map(self, x, y, yaw):
        local_map.PlotLiDAROnOtherFigure(self.ax, position=[x, y], angle=yaw+3.141592/2.0)
        self.fig.canvas.draw_idle()

    def update(self, frame):
        x, y, yaw = self.robot_pose_handler.GetPose()
        self.update_pose()
        self.update_map( x, y, yaw)

    def show(self):
        # Show the final plot with the updated map and pose
        plt.show()

# Rest of the code...

if __name__ == '__main__':
    rospy.init_node('lidar_map_projection_node')
    plt.switch_backend('Qt5Agg')
    # Set the map parameters
    width, height, cell_size = 10, 10, 0.5
    # Create an instance of the LocalMap class
    local_map = LocalMap(width, height, cell_size)
    # Create an instance of the LiDAR2MapProjection class and pass the LocalMap object
    lidar_projection = LiDAR2MapProjection(local_map)
    # Create an instance of the RobotPose class to handle robot's pose
    robot_pose_handler = RobotPose()

    # Create an instance of the Plotter class and pass the LocalMap and RobotPose objects
    plotter = Plotter(local_map, robot_pose_handler)

    # Subscribe to the /velodyne_points topic with the PointCloud2 message
    rospy.Subscriber('/velodyne_points', PointCloud2, lidar_projection.Projection)
    # Subscribe to the /ground_truth/state topic with the Odometry message
    rospy.Subscriber('/ground_truth/state', Odometry, robot_pose_handler.update_pose)
    rospy.Subscriber('/left_camera/image_raw',Image, lidar_projection.UpdateImage)

    anim = FuncAnimation(plotter.fig, plotter.update, interval=50, blit=False)

    # Show the animation
    plotter.show()
    local_map.show()
