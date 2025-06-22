import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os

def calculate_quality(original_path, enhanced_path):
    """Calculate image quality metrics between original and enhanced images."""
    # Read images
    original = cv2.imread(original_path)
    enhanced = cv2.imread(enhanced_path)

    # Check if images are loaded
    if original is None:
        print(f"Error: Image not found at {original_path}")
        return None
    if enhanced is None:
        print(f"Error: Image not found at {enhanced_path}")
        return None

    # Convert BGR to RGB
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

    # Print image dimensions for debugging
    print(f"Original image shape: {original_rgb.shape}")
    print(f"Enhanced image shape: {enhanced_rgb.shape}")

    # Resize enhanced image if dimensions don't match
    if original_rgb.shape != enhanced_rgb.shape:
        enhanced_rgb = cv2.resize(
            enhanced_rgb,
            (original_rgb.shape[1], original_rgb.shape[0]),
            interpolation=cv2.INTER_AREA
        )

    # Determine appropriate win_size
    min_dim = min(original_rgb.shape[0], original_rgb.shape[1])
    win_size = min(7, max(3, min_dim // 2 * 2 + 1))  # Ensure odd number

    # Calculate reference metrics
    try:
        psnr_value = psnr(original_rgb, enhanced_rgb, data_range=255)
        ssim_value = ssim(
            original_rgb,
            enhanced_rgb,
            win_size=win_size,
            channel_axis=2,
            data_range=255
        )
    except ValueError as e:
        print(f"Error calculating reference metrics: {str(e)}")
        return None

    # Display metrics
    print("\n📈 Quality Metrics:")
    print(f"PSNR: {psnr_value:.2f} dB (Higher is better)")
    print(f"SSIM: {ssim_value:.4f} (0-1, 1 is perfect)")

    return {
        'psnr': psnr_value,
        'ssim': ssim_value
    }

def main():
    """Main function to run the image quality assessment with hardcoded filenames."""
    # Hardcoded image filenames
    original_filename = "7.jpg"
    enhanced_filename = "image.png"

    # Get script directory
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()  # Fallback for when __file__ is not defined

    # Construct full paths
    original_path = os.path.join(script_dir, original_filename)
    enhanced_path = os.path.join(script_dir, enhanced_filename)

    # Check if files exist
    if not os.path.isfile(original_path):
        print(f"Error: Original image not found at {original_path}")
        return
    if not os.path.isfile(enhanced_path):
        print(f"Error: Enhanced image not found at {enhanced_path}")
        return

    print("🔹 Image Quality Assessment 🔹\n")
    print(f"Original image: {original_filename}")
    print(f"Enhanced image: {enhanced_filename}")
    
    calculate_quality(original_path, enhanced_path)

if __name__ == "__main__":
    main()
