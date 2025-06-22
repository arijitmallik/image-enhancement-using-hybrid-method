import cv2
import numpy as np
from rembg import remove
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.measure import shannon_entropy
from skimage import color

# Add print statement to check if the script runs
print("Starting script...")

try:
    # Load image
    input_path = 'fused_image.png'  # Update to your actual image path
    original = cv2.imread(input_path)

    # Check if the image was loaded correctly
    if original is None:
        raise FileNotFoundError(f"Could not load image at {input_path}")
    
    print(f"Loaded image from {input_path}.")

    # Convert original image to RGBA and remove background
    original_pil = Image.open(input_path).convert("RGBA")
    removed_pil = remove(original_pil)

    # Convert the removed image to numpy array
    removed_np = np.array(removed_pil)
    
    print("Background removed successfully.")
    
    # Extract the alpha channel
    alpha_channel = removed_np[:, :, 3]
    
    # Create mask (person's area)
    mask = (alpha_channel > 0).astype(np.uint8) * 255

    # Dilate the mask slightly for better coverage
    kernel = np.ones((5, 5), np.uint8)
    mask_dilated = cv2.dilate(mask, kernel, iterations=2)

    # Create white background
    white_background = np.ones_like(original) * 255  # White background

    # Use the inverse of the mask to extract the background
    mask_inv = cv2.bitwise_not(mask_dilated)
    background_part = cv2.bitwise_and(original, original, mask=mask_inv)

    # Fill the person's area with white
    person_white_part = cv2.bitwise_and(white_background, white_background, mask=mask_dilated)

    # Combine background with the white-filled person's area
    final_background = cv2.add(background_part, person_white_part)

    # Save the result
    cv2.imwrite("background_only_white.png", final_background)  # Background with person area white
    print("Saved background image with white-filled person area.")

    # Create the final person on white background
    removed_np_rgb = removed_np[:, :, :3]
    person_mask = (alpha_channel > 0)[:, :, np.newaxis]
    white_bg = np.ones_like(removed_np_rgb) * 255
    person_on_white = removed_np_rgb * person_mask + white_bg * (1 - person_mask)
    person_on_white = person_on_white.astype(np.uint8)

    # Save the final image with the person on a white background
    cv2.imwrite("person_only_white_bg.png", cv2.cvtColor(person_on_white, cv2.COLOR_RGB2BGR))
    print("Saved person on white background image.")

    # Calculate PSNR (Peak Signal-to-Noise Ratio)
    psnr_value = psnr(original, final_background)
    print(f"PSNR: {psnr_value:.2f} dB")

    # Calculate Contrast (standard deviation of pixel values)
    gray_image = cv2.cvtColor(final_background, cv2.COLOR_BGR2GRAY)
    contrast_value = gray_image.std()
    print(f"Contrast: {contrast_value:.2f}")

    # Convert images to float (for SSIM and Entropy calculations)
    original_float = original.astype(np.float32) / 255.0
    final_background_float = final_background.astype(np.float32) / 255.0

    # Calculate SSIM (Structural Similarity Index) with data_range and window size adjustment
    ssim_value = ssim(
        original_float, 
        final_background_float, 
        data_range=final_background_float.max() - final_background_float.min(),
        win_size=5,  # Use a smaller window size (odd value) for SSIM
        channel_axis=2  # Ensure it processes the color channels correctly
    )
    print(f"SSIM: {ssim_value:.2f}")

    # Calculate Entropy (Shannon Entropy)
    entropy_value = shannon_entropy(final_background)
    print(f"Entropy: {entropy_value:.2f}")

except Exception as e:
    print(f"Error occurred: {e}")
