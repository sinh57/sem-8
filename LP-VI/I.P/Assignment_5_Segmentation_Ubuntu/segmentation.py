import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Spyder compatibility - use Qt backend for interactive plots
import matplotlib
matplotlib.use('Qt5Agg')

# Load image - uses relative path (football_image.jpg must be in same folder)
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, 'football_image.jpg')

# Check if file exists
if not os.path.exists(image_path):
    print(f"Error: Image file not found at {image_path}")
    print("Please ensure football_image.jpg is in the same directory as this script")
    exit()

image = cv2.imread(image_path)

if image is None:
    print("Error: Could not read the image. Please check the file format.")
    exit()

def cv2_imshow(img, title=""):
    """Display image using matplotlib - optimized for Spyder"""
    plt.figure(figsize=(10, 8))
    if len(img.shape) == 2:  # Grayscale
        plt.imshow(img, cmap='gray')
    else:  # Color
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    if title:
        plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.show()

# Display original image
print("\n=== Original Football Image ===")
cv2_imshow(image, "Original Football Image")

# =========================================================
# 1) THRESHOLD BASED IMAGE SEGMENTATION
# =========================================================
print("\n=== Threshold Based Segmentation ===")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Set the threshold value
threshold_value = 120

# Threshold the image to create a binary image
_, binary_image = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

cv2_imshow(binary_image, "Threshold Segmentation (Binary)")

# =========================================================
# 2) WATERSHED IMAGE SEGMENTATION
# =========================================================
print("\n=== Watershed Image Segmentation ===")

# Blur the image to reduce noise
gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Threshold the image with Otsu's method
_, binary_watershed = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Create a kernel for morphological operations
kernel = np.ones((3, 3), np.uint8)

# Perform morphological opening to remove small objects
opening = cv2.morphologyEx(binary_watershed, cv2.MORPH_OPEN, kernel, iterations=2)

# Create a mask for the background
background_mask = cv2.dilate(opening, kernel, iterations=3)

# Create a mask for the foreground
foreground_mask = cv2.subtract(binary_watershed, opening)

# Find the markers for the watershed transformation
_, markers = cv2.connectedComponents(foreground_mask)

# Add one to all labels so that the background is not 0, but 1
markers = markers + 1

# Set the background to 0
markers[background_mask == 255] = 0

# Apply watershed algorithm
watershed_result = cv2.watershed(image.copy(), markers)

# Create a colored watershed image for better visualization
watershed_colored = watershed_result.copy().astype(np.uint8)
watershed_colored = cv2.applyColorMap(watershed_colored, cv2.COLORMAP_JET)

cv2_imshow(watershed_colored, "Watershed Segmentation Result")

# =========================================================
# COMPARISON
# =========================================================
print("\n=== Segmentation Methods Comparison ===")

# Resize for comparison
binary_resized = cv2.resize(binary_image, (400, 300))
opening_resized = cv2.resize(opening, (400, 300))

# Create comparison
comparison = np.hstack((
    cv2.cvtColor(binary_resized, cv2.COLOR_GRAY2BGR),
    cv2.cvtColor(opening_resized, cv2.COLOR_GRAY2BGR)
))

# Display comparison
plt.figure(figsize=(14, 6))
plt.imshow(cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB))
plt.title("Threshold vs Opening", fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.show()

print("\n✓ Image segmentation completed successfully!")
