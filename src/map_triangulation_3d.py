import cv2
import numpy as np

# -------------------------
# Map Triangulation 3D
# -------------------------
# Description:
#   Computes the 3D spatial coordinates of matched 
#   feature points by triangulating rays from two 
#   different camera poses.

class MapTriangulator:
    def __init__(self, K):
        self.K = K

    # -------------------------
    # Triangulate Points
    # -------------------------
    # Inputs:
    #   R, t       : Relative Rotation and Translation from MotionEstimator.
    #   pts1, pts2 : Filtered coordinates of matched points (Inliers).
    #
    # Description:
    #   Computes the 3D spatial coordinates of matched features by 
    #   intersecting light rays from two different camera poses. 
    #   It constructs Projection Matrices (P = K[R|t]) and solves 
    #   the system using a Direct Linear Transform (DLT) approach.
    #
    # Outputs:
    #   points_3d : Array of Euclidean (X, Y, Z) coordinates.
    # -------------------------
    def triangulate(self, R, t, pts1, pts2):
        # 1. Construct Camera Projection Matrices
        # Camera 1 is assumed to be at the World Origin [I | 0]
        # P1 = K * [Identity_3x3 | Zero_Vector_3x1]
        P1 = np.dot(self.K, np.hstack((np.eye(3), np.zeros((3, 1)))))

        # Camera 2 position is defined by the recovered motion [R | t]
        # P2 = K * [Rotation | Translation]
        P2 = np.dot(self.K, np.hstack((R, t)))

        # 2. Data Formatting
        # OpenCV's triangulatePoints requires (2, N) input arrays
        pts1_t = pts1.T
        pts2_t = pts2.T

        # 3. Linear Triangulation
        # Returns points in Homogeneous Coordinates (x, y, z, w)
        points_4d_hom = cv2.triangulatePoints(P1, P2, pts1_t, pts2_t)

        # 4. Perspective Division
        # Convert from Homogeneous to Euclidean coordinates (X, Y, Z)
        # We divide by the 'w' component to normalize the projection
        points_3d = points_4d_hom[:3] / points_4d_hom[3]

        # Return transposed to maintain (N, 3) consistency in the pipeline
        return points_3d.T