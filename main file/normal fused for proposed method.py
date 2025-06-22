import cv2
import numpy as np

def load_images(image_paths):
    """Load images from the given file paths."""
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {path}")
        images.append(img)
    return images

def fuse_images(images):
    """Fuse images by averaging them."""
    fused_img = np.zeros_like(images[0], dtype=np.float32)
    for img in images:
        fused_img += img.astype(np.float32)
    fused_img /= len(images)  # Average the images
    fused_img = np.clip(fused_img, 0, 255).astype(np.uint8)
    return fused_img

def main(image_paths, output_path='fused_image.png'):
    if len(image_paths) != 2:
        raise ValueError("Exactly 10 images are required.")

    images = load_images(image_paths)
    fused_img = fuse_images(images)
    cv2.imwrite(output_path, fused_img)

    print(f"Fused image saved to: {output_path}")

if __name__ == "__main__":
    image_paths = [
        #  '/Users/arijitmalik/Documents/final/fusion method/log_alpha_0.1.png',
        # '/Users/arijitmalik/Documents/final/fusion method/log_alpha_0.2.png',
        # '/Users/arijitmalik/Documents/final/fusion method/log_alpha_0.3.png',
        # '/Users/arijitmalik/Documents/final/fusion method/gamma_0.6.png',
        # '/Users/arijitmalik/Documents/final/fusion method/gamma_0.7.png',
        # '/Users/arijitmalik/Documents/final/fusion method/gamma_0.8.png',
        # '/Users/arijitmalik/Documents/final/fusion method/gamma_0.9.png',
        # '/Users/arijitmalik/Documents/final/fusion method/gamma_1.0.png',
        '/Users/arijitmalik/Documents/final/fusion method/test10_illumination.jpg',
        '/Users/arijitmalik/Documents/final/fusion method/fused_image.png'
    ]
    main(image_paths, output_path='/Users/arijitmalik/Documents/final/fusion method/fused_image_final.png')
