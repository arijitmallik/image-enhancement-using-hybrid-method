import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the two images
img1 = cv2.imread('/Users/arijitmalik/Documents/final/fusion method/test11.jpg')
img2 = cv2.imread('/Users/arijitmalik/Documents/final/fusion method/fused_image.png')
         # Image to align

# Convert to grayscale for feature detection
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# ORB detector
orb = cv2.ORB_create(5000)

# Detect and compute keypoints/descriptors
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

# Matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Sort matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# Extract location of good matches
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)

# Find homography
H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC)

# Warp the second image to align with the first
height, width, _ = img1.shape
aligned_img2 = cv2.warpPerspective(img2, H, (width, height))

# Blend the two images
blended = cv2.addWeighted(img1, 0.5, aligned_img2, 0.5, 0)

# Save the result
output_path = '/Users/arijitmalik/Documents/final/fusion method/test1.jpg'
cv2.imwrite(output_path, blended)

# Display (optional)
plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
plt.title('Aligned and Superimposed')
plt.axis('off')
plt.show()
