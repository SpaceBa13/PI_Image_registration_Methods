import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import ORB, match_descriptors
from skimage.transform import AffineTransform, warp, resize
from skimage.measure import ransac

# ---------------------------
# Load images
# ---------------------------
image1 = cv2.imread('1_control.jpeg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('2_control.jpeg', cv2.IMREAD_GRAYSCALE)

# Check if images are loaded
if image1 is None or image2 is None:
    raise ValueError("One or both images failed to load. Check the file paths.")

# Resize images to be the same size
image1 = resize(image1, (500, 500), anti_aliasing=True)
image2 = resize(image2, (500, 500), anti_aliasing=True)

# ---------------------------
# Initialize ORB detector and extract keypoints/descriptors
# ---------------------------
orb = ORB(n_keypoints=500)

orb.detect_and_extract(image1)
keypoints1 = orb.keypoints
descriptors1 = orb.descriptors

orb.detect_and_extract(image2)
keypoints2 = orb.keypoints
descriptors2 = orb.descriptors

# ---------------------------
# Match descriptors
# ---------------------------
matches = match_descriptors(descriptors1, descriptors2, cross_check=True)

# Extract matched keypoints
src = keypoints1[matches[:, 0]]
dst = keypoints2[matches[:, 1]]

# ---------------------------
# Custom function to plot matches
# ---------------------------
def plot_matches_custom(ax, image1, image2, keypoints1, keypoints2, matches, inliers):
    ax.imshow(np.hstack([image1, image2]), cmap='gray')
    ax.axis('off')
    offset = image1.shape[1]

    for idx, (i, j) in enumerate(matches):
        y1, x1 = keypoints1[i]
        y2, x2 = keypoints2[j]

        if inliers[idx]:
            color = 'lime'   # match bueno
        else:
            color = 'red'    # match malo

        ax.plot([x1, x2 + offset], [y1, y2], color=color, linewidth=0.7)
        ax.plot(x1, y1, 'o', color=color, markersize=3)
        ax.plot(x2 + offset, y2, 'o', color=color, markersize=3)


# ---------------------------
# Verify number of matches
# ---------------------------
print(f"Number of matches: {len(matches)}")
if len(matches) < 4:
    raise ValueError("Not enough matches to compute a reliable transformation")

# ---------------------------
# Compute affine transformation using RANSAC
# ---------------------------
model_robust, inliers = ransac((dst, src),
                               AffineTransform, min_samples=4,
                               residual_threshold=2, max_trials=1000)

# Warp image
registered_image = warp(image1, model_robust.inverse, output_shape=image2.shape)


# Plot keypoints and matches
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
plot_matches_custom(ax, image1, image2, keypoints1, keypoints2, matches, inliers)
ax.set_title("Matches (verde = buenos, rojo = malos)")
plt.show()


# ---------------------------
# ORB Similarity Score
# ---------------------------

num_matches = len(matches)
num_inliers = np.sum(inliers)

num_keypoints = min(len(keypoints1), len(keypoints2))

similarity_score = num_inliers / num_keypoints

print("Matches:", num_matches)
print("Inliers:", num_inliers)
print("Similarity score:", similarity_score)


# ---------------------------
# Display results
# ---------------------------
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title('Image 1')
plt.imshow(image1, cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Image 2')
plt.imshow(image2, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Registered Image')
plt.imshow(registered_image, cmap='gray')

plt.show()