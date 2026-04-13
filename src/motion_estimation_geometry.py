import cv2
import numpy as np

# -------------------------
# Motion Estimation Geometry
# -------------------------
# Description:
#   Extracts the relative camera motion (Rotation and 
#   Translation) between two frames using the Essential 
#   Matrix and the Cheirality constraint.

class MotionEstimator:
    def __init__(self, K):
        # -------------------------
        # Initialize Estimator
        # -------------------------
        # Inputs:
        #   K : Camera intrinsic matrix
        #
        # Description:
        #   Stores the intrinsic parameters necessary for 
        #   decomposing the Essential Matrix into R and t.
        self.K = K

    # -------------------------
    # Recover Pose
    # -------------------------
    # Inputs:
    #   pts1, pts2 : filtered coordinates from FeatureMatcher
    #
    # Description:
    #   Estimates the Essential Matrix and decomposes it 
    #   to recover the relative rotation and translation 
    #   vectors.
    #
    # Outputs:
    #   R    : rotation matrix (3x3)
    #   t    : translation vector (3x1)
    #   mask : mask of points that pass the cheirality check
    
    def recover_camera_motion(self, pts1, pts2):
        # -------------------------
        # Essential Matrix Computation
        # -------------------------
        # Description:
        #   Calculates the Essential Matrix E, which encodes 
        #   the geometric relationship between two views.
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K, 
            method=cv2.RANSAC, 
            prob=0.999, 
            threshold=1.0
        )

        # -------------------------
        # Pose Recovery
        # -------------------------
        # Description:
        #   Decomposes E into R and t. It uses the Cheirality 
        #   check to ensure the 3D points are in front of 
        #   both cameras.
        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)

        return R, t, mask_pose