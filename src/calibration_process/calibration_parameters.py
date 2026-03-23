import cv2
import numpy as np
import glob

# -------------------------
# Chessboard configuration
# -------------------------
# Inputs:
#   chessboard_size : number of internal corners (columns, rows)
#
# Description:
#   Defines the chessboard pattern used during calibration.
#   The configuration must match the calibration pattern
#   used to capture the calibration images.
#
# Outputs:
#   objp : 3D coordinates of the chessboard corners
#          in the calibration pattern coordinate system.

chessboard_size = (9, 6)

# Prepare object points
# Description:
#   Generates the 3D coordinates for each chessboard corner
#   assuming the chessboard lies on the Z = 0 plane.
#   These coordinates represent the real-world position
#   of the calibration pattern corners.

objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# -------------------------
# Calibration point storage
# -------------------------
# Inputs:
#   None
#
# Description:
#   Lists used to accumulate the detected calibration data
#   across all calibration images.
#
# Outputs:
#   objpoints : list of 3D points in world coordinates
#   imgpoints : list of corresponding 2D image points

objpoints = []
imgpoints = []

# -------------------------
# Load calibration images
# -------------------------
# Inputs:
#   resources/calibration_frames/*.jpg
#
# Description:
#   Loads all extracted calibration frames generated
#   from the calibration video.
#
# Outputs:
#   images : list of calibration image paths

images = glob.glob("resources/calibration_frames/*.jpg")

print("Images found:", len(images))

# -------------------------
# Chessboard detection loop
# -------------------------
# Inputs:
#   calibration images
#
# Description:
#   Each calibration image is processed to detect the
#   chessboard pattern. If the pattern is successfully
#   detected, the corner positions are refined and stored
#   for camera calibration.
#
# Outputs:
#   objpoints : accumulated 3D object points
#   imgpoints : accumulated refined image corners

for fname in images:

    img = cv2.imread(fname)

    if img is None:
        continue

    # Convert image to grayscale for corner detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:

        # Store the corresponding object points
        objpoints.append(objp)

        # -------------------------
        # Subpixel corner refinement
        # -------------------------
        # Inputs:
        #   gray image
        #   detected corners
        #
        # Description:
        #   Improves the precision of detected corner positions
        #   using subpixel refinement.
        #
        # Outputs:
        #   corners_refined : refined corner coordinates

        corners_refined = cv2.cornerSubPix(
            gray,
            corners,
            (11,11),
            (-1,-1),
            (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001
            )
        )

        imgpoints.append(corners_refined)

        # Draw detected corners for visualization
        cv2.drawChessboardCorners(img, chessboard_size, corners_refined, ret)

        cv2.imshow("Corners", img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# -------------------------
# Calibrate camera
# -------------------------
# Inputs:
#   objpoints : 3D chessboard coordinates
#   imgpoints : detected 2D corner positions
#   image resolution
#
# Description:
#   Computes the intrinsic camera parameters and
#   lens distortion coefficients using OpenCV's
#   calibration algorithm.
#
# Outputs:
#   K      : camera intrinsic matrix
#   dist   : distortion coefficients
#   rvecs  : rotation vectors for each image
#   tvecs  : translation vectors for each image

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    None,
    None
)

print("\nCamera matrix (K):")
print(K)

print("\nDistortion coefficients:")
print(dist)

# -------------------------
# Save calibration parameters
# -------------------------
# Inputs:
#   K    : camera intrinsic matrix
#   dist : distortion coefficients
#
# Description:
#   Saves the camera calibration parameters so they
#   can be reused by other modules such as:
#   - image undistortion
#   - pose estimation
#   - triangulation
#   - SLAM pipeline
#
# Outputs:
#   calibration_results/camera_matrix.npy
#   calibration_results/dist_coeffs.npy

np.save("calibration_results/camera_matrix.npy", K)
np.save("calibration_results/dist_coeffs.npy", dist)

print("\nParameters saved:")
print("camera_matrix.npy")
print("dist_coeffs.npy")