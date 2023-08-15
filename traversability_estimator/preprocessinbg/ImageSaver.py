import utils
import numpy as np
import cv2
import tf2_ros
import sensor_msgs.point_cloud2 as pc2
import rospy
from sensor_msgs.msg import CameraInfo
import tf
from cv_bridge import CvBridge
import torch
from queue import Queue

class FixedSizeQueue:
    def __init__(self, size, init_value=None):
        self.size = size
        self.queue = Queue(maxsize=size)
        
        # Fill the queue with initial values
        for _ in range(size):
            self.queue.put(init_value)
    
    def push(self, item):
        if self.queue.full():
            self.queue.get()  # Remove the oldest item if the queue is full
        self.queue.put(item)
    
    def get(self):
        return list(self.queue.queue)
    
class RGBSaver():
    def __init__(self,data_dict):
        self.data_dict = data_dict
        self.queue_image = FixedSizeQueue(size=10, init_value=None)

        # Get the camera intrinsic parameters from the CameraInfo topic
        camera_info_topic = "/left_camera/camera_info"  # Replace with your actual camera_info topic
        self.camera_info_msg = rospy.wait_for_message(camera_info_topic, CameraInfo)
        self.fx = self.camera_info_msg.K[0]
        self.fy = self.camera_info_msg.K[4]
        self.cx = self.camera_info_msg.K[2]
        self.cy = self.camera_info_msg.K[5]

        self.image = np.zeros((self.camera_info_msg.height,self.camera_info_msg.width))

        self.bridge = CvBridge()

    def UpdateImage(self, msg):
        self.image  = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if self.queue_image.get()[0] is not None:
            self.data_dict['RGB_image'] =torch.tensor(self.queue_image.get()[0],dtype=torch.float32)  # Get the first item (oldest) in the queue
        self.queue_image.push(self.image)

    def show(self):
        if self.image is not None:
            cv2.imshow('RGB image',self.image)
    
class DepthSaver():
    def __init__(self, RGBSaver, data_dict):
        self.queue_depth_image = FixedSizeQueue(size=10, init_value=None)
        self.data_dict = data_dict
        self.RGBsaver= RGBSaver
        self.depth_image = np.zeros((RGBSaver.camera_info_msg.height,RGBSaver.camera_info_msg.width))
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

    def DepthGenerator(self,msg):
        self.depth_image = np.zeros_like(self.depth_image)
        for point in pc2.read_points(msg, skip_nans=True):
            x, y, z = point[:3]
            lidar_point = np.array([x, y, z , 1.0])
            camera_point = np.dot(lidar_point,self.Tv2c)
            x_camera = camera_point[0] / camera_point[3]
            y_camera = camera_point[1] / camera_point[3]
            z_camera = camera_point[2] / camera_point[3]
            if z_camera > 0.0: # front points
                # Project the transformed 3D point back to the camera image
                u = self.RGBsaver.fx * ((x_camera )/ z_camera) + self.RGBsaver.cx  # optical axis is different with camera axis!!! need to transform !!
                v = self.RGBsaver.fy * (y_camera / z_camera) + self.RGBsaver.cy
                # Check if the projected point is within the camera image boundaries
                if 0 <= u < self.RGBsaver.camera_info_msg.width and 0 <= v < self.RGBsaver.camera_info_msg.height:
                    distance =np.sqrt(z_camera*z_camera+x_camera*x_camera+y_camera*y_camera)
                    self.depth_image[int(v),int(u)]= distance
        
        self.depth_image = (self.depth_image / np.max(self.depth_image) * 255).astype(np.uint8)  # Normalize and convert to uint8
        if self.queue_depth_image.get()[0] is not None:
            self.data_dict['depth_image'] = torch.tensor(self.queue_depth_image.get()[0],dtype=torch.float32)  # Get the first item (oldest) in the queue
        
        self.queue_depth_image.push(self.depth_image)

    def show(self):
        if self.depth_image is not None:
            cv2.imshow('Depth Image', self.depth_image,)



    


