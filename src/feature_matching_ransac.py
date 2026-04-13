
import cv2
import numpy as np

# -------------------------
# Feature Matching & RANSAC
# -------------------------
class FeatureMatcher:
    def __init__(self, K):
        self.K = K
        # IMPORTANTE: crossCheck debe ser False para usar knnMatch
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def match(self, kp1, des1, kp2, des2):
        # -------------------------
        # KNN Matching
        # -------------------------
        # Description:
        #   Finds the 2 best matches for each descriptor to 
        #   apply Lowe's Ratio Test.
        matches = self.bf.knnMatch(des1, des2, k=2)
        
        # -------------------------
        # Lowe's Ratio Test
        # -------------------------
        # Description:
        #   Filters ambiguous matches by comparing the distance 
        #   of the best match against the second best.
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 8:
            print("Warning: Not enough good matches after Ratio Test.")
            return np.array([]), np.array([]), None

        # Convert matches to numpy arrays
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        # -------------------------
        # Essential Matrix & RANSAC
        # -------------------------
        # Inputs:
        #   pts1, pts2 : Matched feature points in image coordinates (u, v).
        #   self.K     : Camera intrinsic matrix (focal length and principal point).
        #   method     : Robust estimation algorithm (RANSAC).
        #   prob       : Confidence level (99.9%) that the estimated matrix is correct.
        #   threshold  : Maximum distance (1.0 pixel) for a point to be considered an inlier.
        #
        # Description:
        #   Estimates the Essential Matrix (E) which encapsulates the relative 
        #   rotation and translation between two camera views. It uses the 
        #   Epipolar Constraint (x' * E * x = 0) to filter out outlier matches 
        #   that do not follow the camera's physical motion.
        #
        # Outputs:
        #   E    : The 3x3 Essential Matrix.
        #   mask : Boolean array identifying valid geometric inliers.
        # -------------------------
        E, mask = cv2.findEssentialMat(
            pts1,                # Points from the first frame
            pts2,                # Points from the second frame
            self.K,              # Samsung S24 Intrinsic matrix
            method=cv2.RANSAC,   # Random Sample Consensus to ignore wrong matches
            prob=0.999,          # High probability ensures a robust model
            threshold=1.2        # Tight threshold for sub-pixel precision in SLAM
        )

        if mask is not None:
            pts1_filtered = pts1[mask.ravel() == 1]
            pts2_filtered = pts2[mask.ravel() == 1]
            return pts1_filtered, pts2_filtered, mask
        else:
            return np.array([]), np.array([]), None