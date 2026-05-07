"""
ASSIGNMENT 3: IMAGE ENHANCEMENT
===============================
Contrast Stretching and Intensity Level Slicing
Techniques: Linear contrast stretching, intensity range slicing
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Spyder compatibility
import matplotlib
matplotlib.use('Qt5Agg')

print("=" * 70)
print("ASSIGNMENT 3: IMAGE ENHANCEMENT TECHNIQUES")
print("=" * 70)

# Load image
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, 'nature.png')

if not os.path.exists(image_path):
    print(f"Error: Image file not found at {image_path}")
    print("Please ensure nature.png is in the same directory as this script")
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

# Display original image
print(f"\nImage Shape: {image.shape}")
print("\nDisplaying Original Image...")
cv2_imshow(image, "Original Nature Image")

# =========================================================
# TECHNIQUE 1: CONTRAST STRETCHING
# =========================================================
print("\n" + "=" * 70)
print("TECHNIQUE 1: CONTRAST STRETCHING")
print("=" * 70)

# Split the image into BGR channels
B, G, R = cv2.split(image)

# Function to perform contrast stretching on a channel
def contrast_stretch(channel):
    """Apply linear contrast stretching to a channel"""
    min_val = channel.min()
    max_val = channel.max()
    
    if max_val == min_val:
        return channel
    
    # Linear transformation: new_val = ((old_val - min) / (max - min)) * 255
    stretched = ((channel.astype(np.float32) - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return stretched

# Apply contrast stretching to each channel
print("\nApplying contrast stretching to each channel...")
B_stretched = contrast_stretch(B)
G_stretched = contrast_stretch(G)
R_stretched = contrast_stretch(R)

# Merge the channels back
stretched_image = cv2.merge((B_stretched, G_stretched, R_stretched))

print(f"Original - Min: {image.min()}, Max: {image.max()}")
print(f"Stretched - Min: {stretched_image.min()}, Max: {stretched_image.max()}")

print("\nDisplaying Contrast Stretched Image...")
cv2_imshow(image, "Original Image")
cv2_imshow(stretched_image, "Contrast Stretched Image")

# =========================================================
# CONTRAST STRETCHING ANALYSIS
# =========================================================
print("\nCreating contrast stretching comparison...")

fig = plt.figure(figsize=(16, 6))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image', fontsize=12, fontweight='bold')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(stretched_image, cv2.COLOR_BGR2RGB))
plt.title('Contrast Stretched Image', fontsize=12, fontweight='bold')
plt.axis('off')

plt.subplot(1, 3, 3)
# Calculate and display histograms
original_hist = cv2.calcHist([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256])
stretched_hist = cv2.calcHist([cv2.cvtColor(stretched_image, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256])

plt.plot(original_hist, label='Original', color='blue', linewidth=2)
plt.plot(stretched_hist, label='Stretched', color='red', linewidth=2)
plt.title('Histogram Comparison', fontsize=12, fontweight='bold')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.legend()
plt.xlim([0, 256])
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =========================================================
# TECHNIQUE 2: INTENSITY LEVEL SLICING
# =========================================================
print("\n" + "=" * 70)
print("TECHNIQUE 2: INTENSITY LEVEL SLICING")
print("=" * 70)

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print("\nDisplaying Original Grayscale Image...")
cv2_imshow(gray, "Original Grayscale Image")

# Define intensity range for slicing
lower_level = 100
upper_level = 200

print(f"\nIntensity range selected: [{lower_level}, {upper_level}]")

# Method 1: Extract only pixels in the range
mask = cv2.inRange(gray, lower_level, upper_level)
sliced_result_1 = cv2.bitwise_and(image, image, mask=mask)

print("\nDisplaying Sliced Image (pixels in range remain, others become black)...")
cv2_imshow(gray, "Original Grayscale Image")
cv2_imshow(mask, "Mask (white = in range, black = out of range)")
cv2_imshow(sliced_result_1, "Intensity Level Sliced Result (Color)")

# Method 2: Highlight pixels in the range (set to white, others to black)
sliced_binary = np.zeros_like(gray)
sliced_binary[mask != 0] = 255

print("\nDisplaying Binary Sliced Result...")
cv2_imshow(sliced_binary, "Binary Intensity Slicing Result")

# =========================================================
# INTENSITY LEVEL SLICING ANALYSIS
# =========================================================
print("\nCreating intensity level slicing comparison...")

fig = plt.figure(figsize=(16, 10))

# Row 1: Grayscale and Mask
plt.subplot(2, 3, 1)
plt.imshow(gray, cmap='gray')
plt.title('Original Grayscale', fontsize=12, fontweight='bold')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(mask, cmap='gray')
plt.title(f'Mask: [{lower_level}, {upper_level}]', fontsize=12, fontweight='bold')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(sliced_binary, cmap='gray')
plt.title('Binary Sliced Result', fontsize=12, fontweight='bold')
plt.axis('off')

# Row 2: Histograms and results
plt.subplot(2, 3, 4)
plt.hist(gray.ravel(), bins=256, range=[0, 256], color='black', alpha=0.7)
plt.axvline(lower_level, color='red', linestyle='--', linewidth=2, label='Range')
plt.axvline(upper_level, color='red', linestyle='--', linewidth=2)
plt.fill_betweenx([0, 10000], lower_level, upper_level, alpha=0.2, color='green')
plt.title('Histogram with Slicing Range', fontsize=12, fontweight='bold')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.xlim([0, 256])
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 5)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Color Image', fontsize=12, fontweight='bold')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(cv2.cvtColor(sliced_result_1, cv2.COLOR_BGR2RGB))
plt.title('Sliced Result (Color)', fontsize=12, fontweight='bold')
plt.axis('off')

plt.tight_layout()
plt.show()

# =========================================================
# MULTIPLE INTENSITY LEVELS
# =========================================================
print("\n" + "=" * 70)
print("INTENSITY LEVEL SLICING - MULTIPLE RANGES")
print("=" * 70)

# Define multiple ranges for more detailed slicing
ranges = [
    (0, 50, "Very Dark (0-50)"),
    (50, 100, "Dark (50-100)"),
    (100, 150, "Medium Dark (100-150)"),
    (150, 200, "Medium Bright (150-200)"),
    (200, 255, "Bright (200-255)")
]

fig = plt.figure(figsize=(18, 10))

for idx, (lower, upper, label) in enumerate(ranges, 1):
    mask_temp = cv2.inRange(gray, lower, upper)
    
    plt.subplot(2, 5, idx)
    plt.imshow(mask_temp, cmap='gray')
    plt.title(label, fontsize=10, fontweight='bold')
    plt.axis('off')

# Show combined visualization
plt.subplot(2, 5, 6)
plt.imshow(gray, cmap='gray')
plt.title('Original Grayscale', fontsize=10, fontweight='bold')
plt.axis('off')

plt.tight_layout()
plt.show()

# =========================================================
# SUMMARY
# =========================================================
print("\n" + "=" * 70)
print("IMAGE ENHANCEMENT SUMMARY")
print("=" * 70)

print("\n1. CONTRAST STRETCHING:")
print("   - Stretches pixel values to use full 0-255 range")
print("   - Formula: new_val = ((old_val - min) / (max - min)) * 255")
print("   - Improves image contrast and visibility")
print(f"   - Original range: [{image.min()}, {image.max()}]")
print(f"   - Stretched range: [{stretched_image.min()}, {stretched_image.max()}]")

print("\n2. INTENSITY LEVEL SLICING:")
print(f"   - Extracts pixels within range [{lower_level}, {upper_level}]")
print("   - Other pixels are set to black (0)")
print("   - Useful for highlighting specific intensity regions")
print(f"   - Pixels in range: {np.sum(mask != 0)}")
print(f"   - Pixels out of range: {np.sum(mask == 0)}")

print("\n" + "=" * 70)
print("✓ Assignment 3 Completed Successfully!")
print("=" * 70)
