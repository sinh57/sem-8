"""
ASSIGNMENT 1: IMAGE SUBSAMPLING AND RESAMPLING
==============================================
Subsample an image to multiple sizes, then resample back to original
Techniques: Bilinear Interpolation (INTER_LINEAR)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Spyder compatibility
import matplotlib
matplotlib.use('Qt5Agg')

print("=" * 70)
print("ASSIGNMENT 1: IMAGE SUBSAMPLING AND RESAMPLING")
print("=" * 70)

# Load image
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, 'tiger_image.jpeg')

if not os.path.exists(image_path):
    print(f"Error: Image file not found at {image_path}")
    print("Please ensure tiger_image.jpeg is in the same directory as this script")
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

# Get original image dimensions
height, width = image.shape[:2]
print(f"\nOriginal Image Dimensions: {width} x {height}")

# Display original image
print("\nDisplaying Original Image...")
cv2_imshow(image, "Original Tiger Image")

# =========================================================
# SUBSAMPLING: CREATE REDUCED SIZES
# =========================================================
print("\n" + "=" * 70)
print("SUBSAMPLING: REDUCING IMAGE SIZES")
print("=" * 70)

subsampled_images = {}
sizes = [512, 256, 128, 64, 32]

for size in sizes:
    print(f"\nSubsampling to {size}x{size}...")
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    subsampled_images[size] = resized
    print(f"Subsampled Image Size: {resized.shape[1]} x {resized.shape[0]}")
    cv2_imshow(resized, f"Subsampled: {size}x{size}")

# =========================================================
# RESAMPLING: RESTORE TO ORIGINAL SIZE
# =========================================================
print("\n" + "=" * 70)
print("RESAMPLING: RESTORING TO ORIGINAL SIZE (1024x1024)")
print("=" * 70)

resampled_images = {}
original_size = (1024, 1024)

for size in sizes:
    print(f"\nResampling from {size}x{size} back to 1024x1024...")
    resampled = cv2.resize(subsampled_images[size], original_size, interpolation=cv2.INTER_LINEAR)
    resampled_images[size] = resampled
    print(f"Resampled Image Size: {resampled.shape[1]} x {resampled.shape[0]}")
    cv2_imshow(resampled, f"Resampled from {size}x{size} to 1024x1024")

# =========================================================
# COMPARISON: ORIGINAL vs RESAMPLED
# =========================================================
print("\n" + "=" * 70)
print("COMPARISON: ORIGINAL vs RESAMPLED IMAGES")
print("=" * 70)

# Compare from smallest subsample
print("\nComparing Original vs Resampled from 32x32 subsample...")
fig = plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image (1024x1024)', fontsize=12, fontweight='bold')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(resampled_images[32], cv2.COLOR_BGR2RGB))
plt.title('Resampled from 32x32', fontsize=12, fontweight='bold')
plt.axis('off')

plt.tight_layout()
plt.show()

# =========================================================
# ALL SUBSAMPLED SIZES COMPARISON
# =========================================================
print("\nDisplaying all subsampled sizes in one view...")
fig = plt.figure(figsize=(16, 12))

sizes_display = [512, 256, 128, 64, 32]
for idx, size in enumerate(sizes_display, 1):
    plt.subplot(2, 3, idx)
    plt.imshow(cv2.cvtColor(subsampled_images[size], cv2.COLOR_BGR2RGB))
    plt.title(f'{size}x{size} Subsampled', fontsize=12, fontweight='bold')
    plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original 1024x1024', fontsize=12, fontweight='bold')
plt.axis('off')

plt.tight_layout()
plt.show()

# =========================================================
# QUALITY ANALYSIS
# =========================================================
print("\n" + "=" * 70)
print("IMAGE QUALITY ANALYSIS")
print("=" * 70)

def calculate_mse(original, resampled):
    """Calculate Mean Squared Error"""
    mse = np.mean((original.astype(float) - resampled.astype(float)) ** 2)
    return mse

print("\nMean Squared Error (MSE) - Lower is better:")
print("-" * 70)
for size in sizes:
    mse = calculate_mse(image, resampled_images[size])
    psnr = 10 * np.log10((255 ** 2) / mse) if mse > 0 else float('inf')
    print(f"From {size:3d}x{size:3d}: MSE = {mse:10.2f}, PSNR = {psnr:6.2f} dB")

print("\n" + "=" * 70)
print("✓ Assignment 1 Completed Successfully!")
print("=" * 70)
