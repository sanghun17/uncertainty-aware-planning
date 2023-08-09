import numpy as np
import cv2
import tf

def rotate_grid(data_2d, angle_deg):
        center = (np.array(data_2d.shape)+np.array([0.0, 0.0])) / 2.0
        rotation_matrix = cv2.getRotationMatrix2D(tuple(center[::-1]), angle_deg, scale=1.0)

        # Perform the rotation using warpAffine with interpolation
        rotated_array = cv2.warpAffine(data_2d, rotation_matrix, data_2d.shape[::-1], flags=cv2.INTER_LINEAR)
        return rotated_array

def zoom_grid(data_2d,zoom_factor_x,zoom_factor_y):
      crop_rows = int(data_2d.shape[0] * (1 - zoom_factor_y))
      crop_cols = int(data_2d.shape[1] * (1 - zoom_factor_x))
      # Crop the array to remove the outer rows and columns
      cropped_array = data_2d[crop_rows//2 : -crop_rows//2, crop_cols//2 : -crop_cols//2]
      return cropped_array

def multiply_transformations(transform1, transform2):
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