import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage import exposure
from math import log10
import time
import matplotlib.pyplot as plt
from tabulate import tabulate

def calculate_entropy(image):
    """Calculate the Shannon entropy of an image."""
    # Convert to uint8 if not already
    if image.dtype != np.uint8:
        image = (255 * (image - image.min()) / (image.max() - image.min())).astype(np.uint8)
    
    # Calculate histogram and normalize
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    
    # Calculate entropy, avoiding log(0)
    entropy = -np.sum(hist * np.log2(hist + np.finfo(float).eps))
    return entropy

def calculate_contrast(image):
    """Calculate RMS contrast of an image."""
    return image.std()

def calculate_psnr(original, processed):
    """Calculate Peak Signal-to-Noise Ratio between two images."""
    mse = np.mean((original.astype(float) - processed.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * log10(PIXEL_MAX / np.sqrt(mse))

def calculate_mse(original, processed):
    """Calculate Mean Squared Error between two images."""
    return np.mean((original.astype(float) - processed.astype(float)) ** 2)

def normalize_images(original, processed):
    """Ensure images have same dimensions and type."""
    # Convert to grayscale if needed
    if len(original.shape) == 3:
        original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    if len(processed.shape) == 3:
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    
    # Resize if different shapes
    if original.shape != processed.shape:
        processed = cv2.resize(processed, (original.shape[1], original.shape[0]))
    
    return original, processed

def display_histograms(original, processed, title1="Original", title2="Processed"):
    """Display histograms of both images for visual comparison."""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(original.ravel(), 256, [0, 256])
    plt.title(f'Histogram - {title1}')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.hist(processed.ravel(), 256, [0, 256])
    plt.title(f'Histogram - {title2}')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

def assess_image_quality(original_path, processed_path, show_histograms=False):
    """Main function to assess image quality metrics."""
    start_time = time.time()
    
    # Load images
    original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    processed = cv2.imread(processed_path, cv2.IMREAD_GRAYSCALE)
    
    if original is None or processed is None:
        raise ValueError("One or both images could not be loaded. Check file paths.")
    
    # Normalize images
    original, processed = normalize_images(original, processed)
    
    # Calculate metrics
    metrics = {
        'Entropy (Original)': calculate_entropy(original),
        'Entropy (Processed)': calculate_entropy(processed),
        'Contrast (Original)': calculate_contrast(original),
        'Contrast (Processed)': calculate_contrast(processed),
        'SSIM': ssim(original, processed, data_range=processed.max()-processed.min()),
        'PSNR': calculate_psnr(original, processed),
        'MSE': calculate_mse(original, processed)
    }
    
    # Calculate execution time
    metrics['Execution Time (s)'] = time.time() - start_time
    
    # Display histograms if requested
    if show_histograms:
        display_histograms(original, processed, original_path, processed_path)
    
    return metrics

if __name__ == "__main__":
    # File paths
    original_path = 'img_1.png'
    processed_path = 'img_1_kindle.png'
    
    try:
        # Assess image quality
        metrics = assess_image_quality(original_path, processed_path, show_histograms=True)
        
        # Print results in a nice table
        print("\nImage Quality Assessment Results:")
        print(tabulate([(k, v) for k, v in metrics.items()], headers=['Metric', 'Value'], tablefmt='grid'))
        
        # Additional interpretation
        print("\nInterpretation:")
        print(f"- Entropy change: {'Increased' if metrics['Entropy (Processed)'] > metrics['Entropy (Original)'] else 'Decreased'} by {abs(metrics['Entropy (Processed)'] - metrics['Entropy (Original)']):.2f}")
        print(f"- Contrast change: {'Increased' if metrics['Contrast (Processed)'] > metrics['Contrast (Original)'] else 'Decreased'} by {abs(metrics['Contrast (Processed)'] - metrics['Contrast (Original)']):.2f}")
        print(f"- SSIM (1 is perfect): {metrics['SSIM']:.4f}")
        print(f"- PSNR (higher is better): {metrics['PSNR']:.2f} dB")
        print(f"- MSE (lower is better): {metrics['MSE']:.2f}")
        
    except Exception as e:
        print(f"Error: {e}")