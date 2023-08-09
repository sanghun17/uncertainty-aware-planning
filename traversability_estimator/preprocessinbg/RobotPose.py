import tf.transformations
import tf

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
        # print("GetPose: ",self.robot_yaw)
        return self.robot_pose[0], self.robot_pose[1], self.robot_yaw