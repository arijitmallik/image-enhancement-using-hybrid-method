import cv2
import numpy as np
import os

# Load image with alpha channel (transparency preserved)
image = cv2.imread('output13.png', cv2.IMREAD_UNCHANGED)

# Validate image loading
if image is None:
    raise ValueError("Image could not be loaded. Check the path or file format.")

# Separate channels
if image.shape[2] == 4:
    bgr = image[:, :, :3]
    alpha = image[:, :, 3]
else:
    bgr = image
    alpha = None

# Convert BGR to RGB
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
image_float = rgb.astype(np.float32) / 255.0  # Scale to [0, 1]

# Output directory
output_dir = 'transformed_images'
os.makedirs(output_dir, exist_ok=True)

# Gamma values
gamma_values = [0.6, 0.7, 0.8, 0.9, 1.0]

# Log alpha values
alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5]

# ----------------- GAMMA TRANSFORMED IMAGES -----------------
for gamma in gamma_values:
    gamma_corrected = np.power(image_float, gamma)
    gamma_corrected = np.clip(gamma_corrected * 255.0, 0, 255).astype(np.uint8)

    # Re-attach alpha channel if it exists
    if alpha is not None:
        merged = cv2.cvtColor(gamma_corrected, cv2.COLOR_RGB2BGR)
        output = cv2.merge((merged, alpha))
    else:
        output = cv2.cvtColor(gamma_corrected, cv2.COLOR_RGB2BGR)

    filename = f'gamma_{gamma:.1f}.png'
    cv2.imwrite(os.path.join(output_dir, filename), output)

# ----------------- LOG TRANSFORMED IMAGES -----------------
for a in alpha_values:
    # Use scaled image_float in [0, 1]
    log_transformed = np.log(1 + a * image_float) / np.log(1 + a)
    log_transformed = np.clip(log_transformed * 255.0, 0, 255).astype(np.uint8)

    if alpha is not None:
        merged = cv2.cvtColor(log_transformed, cv2.COLOR_RGB2BGR)
        output = cv2.merge((merged, alpha))
    else:
        output = cv2.cvtColor(log_transformed, cv2.COLOR_RGB2BGR)

    filename = f'log_alpha_{a:.1f}.png'
    cv2.imwrite(os.path.join(output_dir, filename), output)

print("✅ Saved 5 gamma and 5 log-transformed images with preserved transparency.")
