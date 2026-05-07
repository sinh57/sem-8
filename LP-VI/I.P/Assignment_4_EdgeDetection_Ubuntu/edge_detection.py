import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Spyder compatibility - use Qt backend for interactive plots
import matplotlib
matplotlib.use('Qt5Agg')

# Load image - uses relative path (bird_image.jpeg must be in same folder)
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, 'bird_image.jpeg')

# Check if file exists
if not os.path.exists(image_path):
    print(f"Error: Image file not found at {image_path}")
    print("Please ensure bird_image.jpeg is in the same directory as this script")
    exit()

image = cv2.imread(image_path)

if image is None:
    print("Error: Could not read the image. Please check the file format.")
    exit()

def cv2_imshow(img, title=""):
    """Display image using matplotlib - optimized for Spyder"""
    plt.figure(figsize=(8, 6))
    if len(img.shape) == 2:  # Grayscale
        plt.imshow(img, cmap='gray')
    else:  # Color
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()

# Show original image
print("\n=== Original Image ===")
cv2_imshow(image, "Original Image")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# =========================================================
# 1) SOBEL EDGE DETECTION
# =========================================================
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
magnitude = (magnitude / np.max(magnitude) * 255).astype(np.uint8)

_, sobel_edges = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)

print("\n=== Sobel Edge Detection ===")
cv2_imshow(sobel_edges, "Sobel Edge Detection")

# =========================================================
# 2) CANNY EDGE DETECTION
# =========================================================
# Apply Gaussian Blur
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

# Apply Canny
canny_edges = cv2.Canny(blurred, 100, 200)

print("\n=== Canny Edge Detection ===")
cv2_imshow(canny_edges, "Canny Edge Detection")

# =========================================================
# 3) PREWITT EDGE DETECTION
# =========================================================
# Define Prewitt kernels
kernelx = np.array([[1, 0, -1],
                    [1, 0, -1],
                    [1, 0, -1]])

kernely = np.array([[1, 1, 1],
                    [0, 0, 0],
                    [-1, -1, -1]])

prewitt_x = cv2.filter2D(gray, -1, kernelx)
prewitt_y = cv2.filter2D(gray, -1, kernely)

prewitt = np.sqrt(prewitt_x**2 + prewitt_y**2)
prewitt = (prewitt / np.max(prewitt) * 255).astype(np.uint8)

_, prewitt_edges = cv2.threshold(prewitt, 50, 255, cv2.THRESH_BINARY)

print("\n=== Prewitt Edge Detection ===")
cv2_imshow(prewitt_edges, "Prewitt Edge Detection")

# =========================================================
# FINAL COMPARISON (All Results Together)
# =========================================================
# Resize images for same display size (optional)
sobel_resized = cv2.resize(sobel_edges, (300, 300))
canny_resized = cv2.resize(canny_edges, (300, 300))
prewitt_resized = cv2.resize(prewitt_edges, (300, 300))

# Combine horizontally
comparison = np.hstack((sobel_resized, canny_resized, prewitt_resized))

print("\n=== Comparison: Sobel | Canny | Prewitt ===")
cv2_imshow(comparison, "Edge Detection Comparison")

print("\n✓ Script completed successfully!")
