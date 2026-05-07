"""
ASSIGNMENT 2: HISTOGRAM EQUALIZATION
====================================
Display histogram, equalized histogram, and equalized image
Techniques: Histogram calculation and equalization
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Spyder compatibility
import matplotlib
matplotlib.use('Qt5Agg')

print("=" * 70)
print("ASSIGNMENT 2: HISTOGRAM EQUALIZATION")
print("=" * 70)

# Load image
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, 'bird_image.jpeg')

if not os.path.exists(image_path):
    print(f"Error: Image file not found at {image_path}")
    print("Please ensure bird_image.jpeg is in the same directory as this script")
    exit()

image = cv2.imread(image_path)
if image is None:
    print("Error: Could not read the image.")
    exit()

def cv2_imshow(img, title=""):
    """Display image using matplotlib"""
    plt.figure(figsize=(10, 8))
    if len(img.shape) == 2:  # Grayscale
        plt.imshow(img, cmap='gray')
    else:  # Color
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    if title:
        plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# =========================================================
# LOAD AND DISPLAY ORIGINAL IMAGE
# =========================================================
print("\n" + "=" * 70)
print("STEP 1: ORIGINAL IMAGE")
print("=" * 70)
print(f"Image Shape: {image.shape}")

cv2_imshow(image, "Original Color Image (Bird)")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(f"Grayscale Image Shape: {gray.shape}")

cv2_imshow(gray, "Original Grayscale Image (Bird)")

# =========================================================
# STEP 1: ORIGINAL HISTOGRAM
# =========================================================
print("\n" + "=" * 70)
print("STEP 2: ORIGINAL HISTOGRAM")
print("=" * 70)

# Calculate histogram for original grayscale image
hist_original = cv2.calcHist([gray], [0], None, [256], [0, 256])

# Plot the original histogram
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(gray.ravel(), bins=256, range=[0, 256], color='black', alpha=0.7)
plt.title('Original Histogram (Grayscale)', fontsize=12, fontweight='bold')
plt.xlabel('Pixel Intensity')
plt.ylabel('Number of Pixels')
plt.xlim([0, 256])
plt.grid(True, alpha=0.3)

# =========================================================
# STEP 2: HISTOGRAM EQUALIZATION
# =========================================================
print("\nPerforming Histogram Equalization...")

equalized_gray = cv2.equalizeHist(gray)

# Calculate histogram for equalized image
hist_equalized = cv2.calcHist([equalized_gray], [0], None, [256], [0, 256])

# Plot the equalized histogram
plt.subplot(1, 2, 2)
plt.hist(equalized_gray.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.7)
plt.title('Equalized Histogram (Grayscale)', fontsize=12, fontweight='bold')
plt.xlabel('Pixel Intensity')
plt.ylabel('Number of Pixels')
plt.xlim([0, 256])
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =========================================================
# STEP 3: DISPLAY EQUALIZED IMAGES
# =========================================================
print("\n" + "=" * 70)
print("STEP 3: EQUALIZED IMAGE")
print("=" * 70)

# Display equalized grayscale image
cv2_imshow(equalized_gray, "Equalized Grayscale Image (Bird)")

# =========================================================
# DETAILED HISTOGRAM COMPARISON
# =========================================================
print("\nCreating detailed histogram comparison...")

fig = plt.figure(figsize=(16, 12))

# Original Grayscale Image
plt.subplot(2, 3, 1)
plt.imshow(gray, cmap='gray')
plt.title('Original Grayscale', fontsize=12, fontweight='bold')
plt.axis('off')

# Original Histogram
plt.subplot(2, 3, 2)
plt.hist(gray.ravel(), bins=256, range=[0, 256], color='black', alpha=0.7)
plt.title('Original Histogram', fontsize=12, fontweight='bold')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.xlim([0, 256])
plt.grid(True, alpha=0.3)

# Original Stats
plt.subplot(2, 3, 3)
plt.text(0.1, 0.8, 'ORIGINAL IMAGE STATISTICS', fontsize=12, fontweight='bold', transform=plt.gca().transAxes)
plt.text(0.1, 0.65, f'Mean: {gray.mean():.2f}', fontsize=11, transform=plt.gca().transAxes)
plt.text(0.1, 0.55, f'Std Dev: {gray.std():.2f}', fontsize=11, transform=plt.gca().transAxes)
plt.text(0.1, 0.45, f'Min: {gray.min()}', fontsize=11, transform=plt.gca().transAxes)
plt.text(0.1, 0.35, f'Max: {gray.max()}', fontsize=11, transform=plt.gca().transAxes)
plt.axis('off')

# Equalized Grayscale Image
plt.subplot(2, 3, 4)
plt.imshow(equalized_gray, cmap='gray')
plt.title('Equalized Grayscale', fontsize=12, fontweight='bold')
plt.axis('off')

# Equalized Histogram
plt.subplot(2, 3, 5)
plt.hist(equalized_gray.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.7)
plt.title('Equalized Histogram', fontsize=12, fontweight='bold')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.xlim([0, 256])
plt.grid(True, alpha=0.3)

# Equalized Stats
plt.subplot(2, 3, 6)
plt.text(0.1, 0.8, 'EQUALIZED IMAGE STATISTICS', fontsize=12, fontweight='bold', transform=plt.gca().transAxes)
plt.text(0.1, 0.65, f'Mean: {equalized_gray.mean():.2f}', fontsize=11, transform=plt.gca().transAxes)
plt.text(0.1, 0.55, f'Std Dev: {equalized_gray.std():.2f}', fontsize=11, transform=plt.gca().transAxes)
plt.text(0.1, 0.45, f'Min: {equalized_gray.min()}', fontsize=11, transform=plt.gca().transAxes)
plt.text(0.1, 0.35, f'Max: {equalized_gray.max()}', fontsize=11, transform=plt.gca().transAxes)
plt.axis('off')

plt.tight_layout()
plt.show()

# =========================================================
# SIDE BY SIDE COMPARISON
# =========================================================
print("\nCreating side-by-side comparison...")

fig = plt.figure(figsize=(16, 6))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Color Image', fontsize=12, fontweight='bold')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(gray, cmap='gray')
plt.title('Original Grayscale', fontsize=12, fontweight='bold')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(equalized_gray, cmap='gray')
plt.title('Equalized Grayscale', fontsize=12, fontweight='bold')
plt.axis('off')

plt.tight_layout()
plt.show()

# =========================================================
# ANALYSIS
# =========================================================
print("\n" + "=" * 70)
print("HISTOGRAM EQUALIZATION ANALYSIS")
print("=" * 70)
print(f"\nOriginal Image:")
print(f"  Mean Intensity: {gray.mean():.2f}")
print(f"  Standard Deviation: {gray.std():.2f}")
print(f"  Min Intensity: {gray.min()}")
print(f"  Max Intensity: {gray.max()}")

print(f"\nEqualized Image:")
print(f"  Mean Intensity: {equalized_gray.mean():.2f}")
print(f"  Standard Deviation: {equalized_gray.std():.2f}")
print(f"  Min Intensity: {equalized_gray.min()}")
print(f"  Max Intensity: {equalized_gray.max()}")

print("\nBenefits of Histogram Equalization:")
print("  ✓ Improves contrast in the image")
print("  ✓ Enhances visibility of details")
print("  ✓ Distributes pixel intensities more evenly")
print("  ✓ Better for image processing operations")

print("\n" + "=" * 70)
print("✓ Assignment 2 Completed Successfully!")
print("=" * 70)
