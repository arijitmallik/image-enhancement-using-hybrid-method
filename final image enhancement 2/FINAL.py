import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from skimage import img_as_float
import math

# Function to calculate PSNR
def calculate_psnr(original, distorted):
    original = img_as_float(original)
    distorted = img_as_float(distorted)
    mse_value = mse(original, distorted)
    if mse_value == 0:
        return float('inf')  # PSNR is infinite when there is no error
    return psnr(original, distorted)

# Function to calculate MSE
def calculate_mse(original, distorted):
    original = img_as_float(original)
    distorted = img_as_float(distorted)
    return mse(original, distorted)

# Function to calculate SSIM
def calculate_ssim(original, distorted):
    original = img_as_float(original)
    distorted = img_as_float(distorted)
    multichannel = len(original.shape) == 3 and original.shape[2] == 3
    height, width = original.shape[:2]
    print(f"Image dimensions: {height}x{width}")
    min_dim = min(height, width)
    if min_dim < 3:
        print("Warning: Image is too small for SSIM calculation. Returning 1.0.")
        return 1.0
    win_size = min(7, min_dim)
    if win_size % 2 == 0:
        win_size -= 1
    print(f"Using win_size: {win_size}")
    try:
        return ssim(original, distorted, channel_axis=2 if multichannel else None, 
                    win_size=win_size, data_range=1.0)
    except TypeError:
        return ssim(original, distorted, multichannel=multichannel, 
                    win_size=win_size, data_range=1.0)

# Function to calculate standard deviation
def calculate_standard_deviation(image):
    return np.std(image)

# Function to calculate contrast
def calculate_contrast(image):
    mean = np.mean(image)
    std_dev = np.std(image)
    return std_dev / mean if mean != 0 else 0

# Function to calculate entropy
def calculate_entropy(image):
    histogram, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])
    histogram = histogram / histogram.sum()
    histogram = histogram[histogram > 0]
    return -np.sum(histogram * np.log2(histogram)) if histogram.size > 0 else 0

# Function to calculate image quality metrics comparing original to enhanced/distorted
def calculate_image_quality(original, comparison):
    original_float = img_as_float(original)
    comparison_float = img_as_float(comparison)

    # PSNR
    psnr_value = calculate_psnr(original_float, comparison_float)
    print(f'PSNR: {psnr_value:.2f} dB')

    # MSE
    mse_value = calculate_mse(original_float, comparison_float)
    print(f'MSE: {mse_value:.4f}')

    # SSIM
    ssim_value = calculate_ssim(original_float, comparison_float)
    print(f'SSIM: {ssim_value:.4f}')

    # Standard Deviation (of original image)
    std_dev = calculate_standard_deviation(original)
    print(f'Standard Deviation: {std_dev:.2f}')

    # Contrast (of original image)
    contrast = calculate_contrast(original)
    print(f'Contrast: {contrast:.2f}')

    # Entropy (of original image)
    entropy_value = calculate_entropy(original)
    print(f'Entropy: {entropy_value:.2f}')

# Example enhancement function (brightness adjustment)
def enhance_image(image):
    # Increase brightness and contrast (example enhancement)
    enhanced = cv2.convertScaleAbs(image, alpha=1.2, beta=30)
    return enhanced

def main():
    # Load image
    image_path = "superimposed.jpg" # Ensure this path is correct
    image = cv2.imread(image_path)

    # Check if the image is valid
    if image is None:
        print("Error: Could not load image. Check the file path.")
        return

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply enhancement
    enhanced_image = enhance_image(image_rgb)

    # Calculate quality metrics comparing original to enhanced
    print("Comparing original to enhanced image:")
    calculate_image_quality(image_rgb, enhanced_image)

if __name__ == "__main__":
    main()