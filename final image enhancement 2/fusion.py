import cv2
import numpy as np

# === Image paths ===
image_paths = [
    'log_0.1_enhanced.png',
    'log_0.2_enhanced.png',
    'log_0.3_enhanced.png',
    'log_0.4_enhanced.png',
    'log_0.5_enhanced.png',
    'gamma_0.6_enhanced.png',
    'gamma_0.7_enhanced.png',
    'gamma_0.8_enhanced.png',
    'gamma_0.9_enhanced.png',
    'gamma_1.0_enhanced.png'
]

# === Load images ===
images = [cv2.imread(p) for p in image_paths]
for idx, img in enumerate(images):
    if img is None:
        print(f"❌ Error loading image: {image_paths[idx]}")
        exit()

images = [img.astype(np.float32) for img in images]

# === Fusion Weights ===
weights = [
    0.092717, 0.129365, 0.159857, 0.192174, 0.213029,
    0.066167, 0.045042, 0.037403, 0.032901, 0.029251
]

# === Weighted Fusion ===
fused_image = np.zeros_like(images[0])
for img, weight in zip(images, weights):
    fused_image += img * weight

fused_image = np.clip(fused_image, 0, 255).astype(np.uint8)

# === Step 1: CLAHE on Lightness Channel ===
lab = cv2.cvtColor(fused_image, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
l_clahe = clahe.apply(l)
lab_enhanced = cv2.merge((l_clahe, a, b))
contrast_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

# === Step 2: Boost Saturation ===
hsv = cv2.cvtColor(contrast_enhanced, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)
s = cv2.multiply(s, 1.2)  # increase saturation
s = np.clip(s, 0, 255).astype(np.uint8)
hsv_enhanced = cv2.merge((h, s, v))
color_boosted = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

# === Step 3: Gamma Correction ===
gamma = 1.1  # >1 brightens slightly, <1 darkens
gamma_corrected = np.power(color_boosted / 255.0, gamma) * 255.0
gamma_corrected = np.clip(gamma_corrected, 0, 255).astype(np.uint8)

# === Step 4: Smart Sharpening Filter ===
sharpen_kernel = np.array([[0, -0.5, 0],
                           [-0.5, 3, -0.5],
                           [0, -0.5, 0]])
sharpened = cv2.filter2D(gamma_corrected, -1, sharpen_kernel)

# === Step 5: Final Denoising ===
final = cv2.fastNlMeansDenoisingColored(sharpened, None, 10, 10, 7, 21)

# === Save the final result ===
output_path = '/Users/arijitmalik/Documents/final/final image enhancement 2/fused_enhanced_final_pro.jpg'
cv2.imwrite(output_path, final)
print(f"\n✅ Final enhanced image saved at:\n{output_path}")

# === Show result ===
cv2.imshow('Final Enhanced Image', final)
cv2.waitKey(0)
cv2.destroyAllWindows()
