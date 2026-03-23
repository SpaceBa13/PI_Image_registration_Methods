import cv2
import os

# -------------------------
# Video input configuration
# -------------------------
# Inputs:
#   video_path : path to the calibration video file
#
# Outputs:
#   VideoCapture object used to read frames from the video
#
# Description:
#   Defines the location of the calibration video that
#   contains the chessboard pattern used for camera calibration.

video_path = "resources/calibration_video/calibration_video.mp4"


# -------------------------
# Output folder configuration
# -------------------------
# Inputs:
#   output_folder : directory where extracted frames will be saved
#
# Outputs:
#   Creates the directory if it does not exist
#
# Description:
#   Frames containing valid chessboard detections will be stored
#   in this directory for later use during camera calibration.

output_folder = "resources/calibration_frames"

os.makedirs(output_folder, exist_ok=True)


# -------------------------
# Chessboard configuration
# -------------------------
# Inputs:
#   chessboard_size : number of internal corners (columns, rows)
#
# Description:
#   Defines the chessboard pattern used during calibration.
#   This must match the pattern used to generate the calibration
#   chessboard image.

chessboard_size = (9,6)


# -------------------------
# Open video file
# -------------------------
# Inputs:
#   video_path
#
# Outputs:
#   cap : OpenCV VideoCapture object
#
# Description:
#   Initializes video reading to process frames sequentially.

cap = cv2.VideoCapture(video_path)

print("Video opened:", cap.isOpened())


# -------------------------
# Frame processing parameters
# -------------------------
# Inputs:
#   None
#
# Description:
#   Controls how frames are extracted from the video.
#
# Variables:
#   saved          : number of calibration frames saved
#   frame_id       : current frame index in the video
#   max_frames     : maximum number of calibration images to store
#   frame_interval : frame spacing between saved images

saved = 0
frame_id = 0
max_frames = 40          # máximo de imágenes para calibración
frame_interval = 20      # guardar solo cada 20 frames


# -------------------------
# Video processing loop
# -------------------------
# Inputs:
#   video frames from the calibration video
#
# Description:
#   Each frame is processed to detect the chessboard pattern.
#   If the pattern is detected, the corners are refined and
#   selected frames are saved for calibration.
#
# Outputs:
#   Saved calibration images in resources/calibration_frames

while True:

    ret, frame = cap.read()

    if not ret or saved >= max_frames:
        break

    # Convert frame to grayscale for corner detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # -------------------------
    # Chessboard detection
    # -------------------------
    # Inputs:
    #   grayscale frame
    #   chessboard_size
    #
    # Description:
    #   Attempts to detect the calibration chessboard pattern
    #   using OpenCV's chessboard detection algorithm.
    #
    # Outputs:
    #   ret_corners : boolean indicating successful detection
    #   corners     : detected chessboard corner coordinates

    ret_corners, corners = cv2.findChessboardCorners(
        gray,
        chessboard_size,
        cv2.CALIB_CB_ADAPTIVE_THRESH +
        cv2.CALIB_CB_NORMALIZE_IMAGE +
        cv2.CALIB_CB_FAST_CHECK
    )

    if ret_corners and frame_id % frame_interval == 0:

        # -------------------------
        # Subpixel corner refinement
        # -------------------------
        # Inputs:
        #   grayscale image
        #   detected corners
        #
        # Description:
        #   Improves the precision of the detected chessboard
        #   corners using subpixel refinement.
        #
        # Outputs:
        #   refined corner coordinates

        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11,11),
            (-1,-1),
            (
                cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001
            )
        )

        # Save the selected calibration frame
        filename = f"{output_folder}/frame_{saved}.jpg"
        cv2.imwrite(filename, frame)

        saved += 1

        # Draw detected chessboard corners for visualization
        cv2.drawChessboardCorners(frame, chessboard_size, corners, ret_corners)

        print("Saved:", filename)

    # Display the video with detection overlay
    cv2.imshow("Video", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

    frame_id += 1


# -------------------------
# Release resources
# -------------------------
# Description:
#   Frees the video capture object and closes all
#   OpenCV visualization windows.

cap.release()
cv2.destroyAllWindows()

print("Frames saved:", saved)