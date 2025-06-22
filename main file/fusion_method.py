import cv2
import numpy as np
import os

def load_images(image_paths):
    """Load images from given paths and verify they exist."""
    images = []
    for path in image_paths:
        # Verify file exists first
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file not found: {path}")
        
        # Load the image
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to load image (might be corrupted or wrong format): {path}")
        images.append(img)
    return images

def calculate_quality_metrics(images):
    """Calculate quality scores using variance of Laplacian (sharpness) as a proxy for RGB images."""
    scores = []
    for img in images:
        # Convert image to grayscale to calculate sharpness (variance of Laplacian)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray_img, cv2.CV_64F).var()
        scores.append(laplacian_var)
    # Normalize scores to (0, 1]
    max_score = max(scores) if max(scores) > 0 else 1  # Prevent division by zero
    norm_scores = [s / max_score for s in scores]
    return norm_scores

def calculate_iq(quality_scores):
    """Calculate image quality values (currently using scores directly)."""
    return quality_scores

def calculate_weights(iq_values):
    """Calculate weights based on image quality scores."""
    total = sum(iq_values)
    # Handle case where all scores are zero
    if total == 0:
        return [1.0/len(iq_values)] * len(iq_values)  # Equal weights
    return [iq / total for iq in iq_values]

def fuse_images(images, weights):
    """Fuse images by applying weights."""
    fused_img = np.zeros_like(images[0], dtype=np.float32)
    for img, w in zip(images, weights):
        fused_img += img.astype(np.float32) * w
    fused_img = np.clip(fused_img, 0, 255).astype(np.uint8)
    return fused_img

def verify_image_paths(image_paths):
    """Verify all image paths exist before processing."""
    missing_files = [path for path in image_paths if not os.path.exists(path)]
    if missing_files:
        print("Error: The following image files were not found:")
        for path in missing_files:
            print(f"- {path}")
        return False
    return True

def main(image_paths, output_path='fused_image.png'):
    """Main function to perform image fusion."""
    # Verify we have exactly 10 images
    if len(image_paths) != 10:
        raise ValueError("Exactly 10 images are required.")
    
    # Verify all files exist before proceeding
    if not verify_image_paths(image_paths):
        return

    try:
        # Load and process images
        images = load_images(image_paths)
        
        # Verify all images have the same dimensions
        shapes = [img.shape for img in images]
        if len(set(shapes)) != 1:
            raise ValueError("All images must have the same dimensions")

        quality_scores = calculate_quality_metrics(images)
        iq_values = calculate_iq(quality_scores)
        weights = calculate_weights(iq_values)
        fused_img = fuse_images(images, weights)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, fused_img)

        print("\nImage fusion completed successfully!")
        print("Sharpness-based scores (higher is better):", quality_scores)
        print("Weights:", weights)
        print(f"Fused image saved to: {output_path}")
        
    except Exception as e:
        print(f"\nError during image processing: {str(e)}")

if __name__ == "__main__":
    # Define image paths
    image_paths = [
        '/Users/arijitmalik/Documents/final/fusion method/log_alpha_0.1_enhanced.png',
        '/Users/arijitmalik/Documents/final/fusion method/log_alpha_0.2_enhanced.png',
        '/Users/arijitmalik/Documents/final/fusion method/log_alpha_0.3_enhanced.png',
        '/Users/arijitmalik/Documents/final/fusion method/gamma_0.6_enhanced.png',
        '/Users/arijitmalik/Documents/final/fusion method/gamma_0.7_enhanced.png',
        '/Users/arijitmalik/Documents/final/fusion method/gamma_0.8_enhanced.png',
        '/Users/arijitmalik/Documents/final/fusion method/gamma_0.9_enhanced.png',
        '/Users/arijitmalik/Documents/final/fusion method/gamma_1.0_enhanced.png',
        '/Users/arijitmalik/Documents/final/fusion method/log_alpha_0.4_enhanced.png',
        '/Users/arijitmalik/Documents/final/fusion method/log_alpha_0.5_enhanced.png'
    ]
    
    # Output path
    output_path = '/Users/arijitmalik/Documents/final/fusion method/fused_image.png'
    
    # Run the main function
    main(image_paths, output_path)