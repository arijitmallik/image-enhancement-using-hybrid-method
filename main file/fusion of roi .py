import cv2
import numpy as np
from rembg import remove

# Load original images
background = cv2.imread('originaltest5.jpg')
input_image = cv2.imread('fused_image.png')

# Remove background
input_image_rgba = remove(input_image)  # Output is RGBA (with alpha channel)

# Convert to OpenCV format
input_image_rgba = cv2.imdecode(np.frombuffer(input_image_rgba, np.uint8), cv2.IMREAD_UNCHANGED)

# Resize the removed-background image
region_width, region_height = 200, 200
resized_rgba = cv2.resize(input_image_rgba, (region_width, region_height))

# Get center position to paste
h1, w1 = background.shape[:2]
start_x = w1 // 2 - region_width // 2
start_y = h1 // 2 - region_height // 2

# Separate BGR and alpha channel
bgr = resized_rgba[:, :, :3]
alpha = resized_rgba[:, :, 3] / 255.0

# Overlay on background using alpha blending
for c in range(3):
    background[start_y:start_y + region_height, start_x:start_x + region_width, c] = \
        (1 - alpha) * background[start_y:start_y + region_height, start_x:start_x + region_width, c] + \
        alpha * bgr[:, :, c]

# Save and show result
cv2.imwrite('final_overlay.jpg', background)
cv2.imshow("Final Result", background)
cv2.waitKey(0)
cv2.destroyAllWindows()
