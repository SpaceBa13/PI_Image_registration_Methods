import cv2

# -------------------------
# ORB Feature Extractor
# -------------------------
# Description:
#   Handles the detection of keypoints and computation 
#   of binary descriptors using the ORB algorithm.

class ORBFeatureExtractor:
    def __init__(self, n_features=2000):
        # -------------------------
        # Initialize ORB
        # -------------------------
        # Description:
        #   n_features set to 2000 to ensure enough points 
        #   for 3D triangulation and pose estimation.
        self.orb = cv2.ORB_create(nfeatures=n_features)

    # -------------------------
    # Extract Features
    # -------------------------
    # Inputs:
    #   image : BGR or Gray image array
    #
    # Description:
    #   Converts image to grayscale if necessary and 
    #   extracts keypoints and their corresponding descriptors.
    #
    # Outputs:
    #   keypoints   : list of detected CV keypoints
    #   descriptors : binary descriptors for matching
    
    def extract(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        return keypoints, descriptors