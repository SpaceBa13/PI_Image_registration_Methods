import cv2
import numpy as np
import os

# -------------------------
# Camera Data Provider
# -------------------------
# Description:
#   Manages the loading of camera calibration parameters 
#   and video frames, ensuring data integrity for the 
#   SLAM pipeline.

class DataProvider:
    def __init__(self, calibration_path="calibration_results/"):
        self.k_path = os.path.join(calibration_path, "camera_matrix.npy")
        self.d_path = os.path.join(calibration_path, "dist_coeffs.npy")

    # -------------------------
    # Load Camera Parameters
    # -------------------------
    # Inputs:
    #   Path to .npy files (defined in constructor)
    #
    # Description:
    #   Retrieves the intrinsic matrix K and distortion 
    #   coefficients D calculated during calibration.
    #
    # Outputs:
    #   K    : camera intrinsic matrix
    #   dist : lens distortion coefficients
    
    def load_camera_parameters(self):
        if not os.path.exists(self.k_path) or not os.path.exists(self.d_path):
            raise FileNotFoundError("Calibration files not found. Run calibration first.")
        
        K = np.load(self.k_path)
        dist = np.load(self.d_path)
        return K, dist

    # -------------------------
    # Load Image Frame
    # -------------------------
    # Inputs:
    #   frame_path : path to the image file
    #
    # Description:
    #   Loads a frame while maintaining the original 
    #   aspect ratio (16:9) required for the K matrix.
    #
    # Outputs:
    #   frame : BGR image array
    
    def load_frame(self, frame_path):
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Error: Could not load frame at {frame_path}")
        return frame