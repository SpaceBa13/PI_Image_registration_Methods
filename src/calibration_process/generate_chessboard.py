import numpy as np
import cv2

# -----------------------------
# CONFIGURATION
# -----------------------------

# Number of internal corners (OpenCV format)
cols = 9
rows = 6

# Square size in pixels (for screen)
square_size_px = 120

# -----------------------------
# COMPUTE IMAGE SIZE
# -----------------------------

# number of squares
board_cols = cols + 1
board_rows = rows + 1

width = board_cols * square_size_px
height = board_rows * square_size_px

# -----------------------------
# CREATE CHESSBOARD
# -----------------------------

board = np.zeros((height, width), dtype=np.uint8)

for y in range(board_rows):
    for x in range(board_cols):

        if (x + y) % 2 == 0:
            cv2.rectangle(
                board,
                (x * square_size_px, y * square_size_px),
                ((x + 1) * square_size_px, (y + 1) * square_size_px),
                255,
                -1
            )

# -----------------------------
# SAVE IMAGE
# -----------------------------

filename = "chessboard_9x6_screen.png"

cv2.imwrite(filename, board)

print("Chessboard generated:", filename)
print("Image size:", width, "x", height, "pixels")
print("Square size:", square_size_px, "px")