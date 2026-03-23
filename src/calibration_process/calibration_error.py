import cv2
import numpy as np
import glob

# -------------------------
# Chessboard configuration
# -------------------------
# Inputs:
#   chessboard_size : number of internal chessboard corners
#                     (columns, rows)
#
# Description:
#   Defines the pattern configuration used during the
#   camera calibration process. The values must match
#   the chessboard used to capture the calibration images.
#
# Outputs:
#   objp : 3D coordinates of the chessboard corners
#          in the calibration pattern coordinate system

chessboard_size = (9,6)

objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# -------------------------
# Calibration point storage
# -------------------------
# Inputs:
#   None
#
# Description:
#   Lists used to store the 3D points in the real world
#   (object points) and the corresponding 2D image points
#   detected in the calibration images.
#
# Outputs:
#   objpoints : list of 3D chessboard points
#   imgpoints : list of 2D detected image corners

objpoints = []
imgpoints = []

# -------------------------
# Load calibration images
# -------------------------
# Inputs:
#   resources/calibration_frames/*.jpg
#
# Description:
#   Loads all calibration images extracted from the
#   calibration video.
#
# Outputs:
#   images : list of file paths to calibration frames

images = glob.glob("resources/calibration_frames/*.jpg")

# -------------------------
# Chessboard detection loop
# -------------------------
# Inputs:
#   calibration images
#
# Description:
#   Each image is processed to detect the chessboard pattern.
#   If the pattern is found, the detected corners are refined
#   using subpixel accuracy and stored for calibration.
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
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size)

    if ret:

        # Store corresponding object points
        objpoints.append(objp)

        # -------------------------
        # Subpixel corner refinement
        # -------------------------
        # Inputs:
        #   gray image
        #   detected corners
        #
        # Description:
        #   Improves corner localization accuracy by refining
        #   corner positions at the subpixel level.
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

# -------------------------
# Load calibration parameters
# -------------------------
# Inputs:
#   calibration_results/camera_matrix.npy
#   calibration_results/dist_coeffs.npy
#
# Description:
#   Loads previously computed intrinsic camera parameters
#   and distortion coefficients obtained during calibration.
#
# Outputs:
#   K    : camera intrinsic matrix
#   dist : distortion coefficients

K = np.load("calibration_results/camera_matrix.npy")
dist = np.load("calibration_results/dist_coeffs.npy")

# -------------------------
# Recompute camera poses
# -------------------------
# Inputs:
#   objpoints
#   imgpoints
#   image resolution
#
# Description:
#   Recomputes the camera pose (rotation and translation)
#   for each calibration image using the detected corner
#   correspondences.
#
# Outputs:
#   K      : intrinsic camera matrix
#   dist   : distortion coefficients
#   rvecs  : rotation vectors
#   tvecs  : translation vectors

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    None,
    None
)

# -------------------------
# Compute reprojection error
# -------------------------
# Inputs:
#   objpoints
#   imgpoints
#   camera parameters
#
# Description:
#   Measures the average distance between the observed
#   corner locations and the projected 3D points using
#   the estimated camera parameters.
#
# Outputs:
#   mean_error : average reprojection error in pixels

mean_error = 0

for i in range(len(objpoints)):

    imgpoints2, _ = cv2.projectPoints(
        objpoints[i],
        rvecs[i],
        tvecs[i],
        K,
        dist
    )

    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)

    mean_error += error

mean_error /= len(objpoints)

print("\nReprojection error:", mean_error)