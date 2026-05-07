"""
IMAGE COMPRESSION MINI PROJECT
===============================
Compression Techniques:
1. Run Length Encoding (RLE)
2. Discrete Cosine Transform (DCT)
3. Scalar Quantization

Instructions: Attach print of program, input image & output
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Spyder compatibility
import matplotlib
matplotlib.use('Qt5Agg')

print("=" * 70)
print("IMAGE COMPRESSION MINI PROJECT")
print("Three Compression Techniques: RLE, DCT, Scalar Quantization")
print("=" * 70)

# Load image
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, 'camel_image.webp')

if not os.path.exists(image_path):
    print(f"Error: Image file not found at {image_path}")
    print("Please ensure camel_image.webp is in the same directory as this script")
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
print("\n" + "=" * 70)
print("ORIGINAL IMAGE")
print("=" * 70)
cv2_imshow(image, "Original Image (Camel)")

# =============================================================================
# TECHNIQUE 1: RUN LENGTH ENCODING (RLE) IMAGE COMPRESSION
# =============================================================================
print("\n" + "=" * 70)
print("TECHNIQUE 1: RUN LENGTH ENCODING (RLE) COMPRESSION")
print("=" * 70)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Compression
compressed_rle = []
height, width = gray_image.shape
cur_pixel = -1
cur_run = 0

for row in range(height):
    for col in range(width):
        pixel = gray_image[row, col]
        
        if pixel == cur_pixel:
            cur_run += 1
        else:
            if cur_pixel != -1:
                compressed_rle.append((cur_pixel, cur_run))
            cur_pixel = pixel
            cur_run = 1

if cur_pixel != -1:
    compressed_rle.append((cur_pixel, cur_run))

# Decompression
decompressed_rle = np.zeros((height, width), dtype=np.uint8)
row, col = 0, 0

for pixel, run_length in compressed_rle:
    remaining = run_length
    while remaining > 0 and row < height:
        can_fit = min(remaining, width - col)
        decompressed_rle[row, col:col + can_fit] = pixel
        col += can_fit
        remaining -= can_fit
        
        if col >= width:
            col = 0
            row += 1

print(f"Original Image Size: {height * width} pixels")
print(f"Compressed Data Points: {len(compressed_rle)}")
compression_ratio_rle = (len(compressed_rle) * 2) / (height * width) * 100
print(f"RLE Compression Ratio: {compression_ratio_rle:.2f}%")

print("\nRLE - Original Image:")
cv2_imshow(gray_image, "RLE - Original (Grayscale)")

print("\nRLE - Decompressed Image:")
cv2_imshow(decompressed_rle, "RLE - Decompressed Image")


# =============================================================================
# TECHNIQUE 2: DISCRETE COSINE TRANSFORM (DCT) IMAGE COMPRESSION
# =============================================================================
print("\n" + "=" * 70)
print("TECHNIQUE 2: DISCRETE COSINE TRANSFORM (DCT) COMPRESSION")
print("=" * 70)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
compressed_dct = gray_image.copy().astype(np.float32)
height, width = gray_image.shape

# Apply DCT compression in 8x8 blocks
block_size = 8
for i in range(0, height, block_size):
    for j in range(0, width, block_size):
        block = gray_image[i:i+block_size, j:j+block_size].astype(np.float32)
        
        # Apply DCT
        dct_block = cv2.dct(block)
        
        # Keep only top-left frequencies (low frequency)
        dct_block[block_size//4:, :] = 0  # Zero out high frequencies
        dct_block[:, block_size//4:] = 0
        
        # Apply inverse DCT
        idct_block = cv2.idct(dct_block)
        compressed_dct[i:i+block_size, j:j+block_size] = idct_block

compressed_dct = np.uint8(np.clip(compressed_dct, 0, 255))

print(f"DCT Block Size: {block_size}x{block_size}")
print(f"Compression achieved by keeping only low frequencies")

print("\nDCT - Original Image:")
cv2_imshow(gray_image, "DCT - Original (Grayscale)")

print("\nDCT - Compressed Image:")
cv2_imshow(compressed_dct, "DCT - Compressed Image")


# =============================================================================
# TECHNIQUE 3: SCALAR QUANTIZATION IMAGE COMPRESSION
# =============================================================================
print("\n" + "=" * 70)
print("TECHNIQUE 3: SCALAR QUANTIZATION COMPRESSION")
print("=" * 70)

def quantize(image_input, levels):
    """Quantize image to specified number of levels"""
    flat_image = image_input.flatten().astype(np.float32)
    min_val = flat_image.min()
    max_val = flat_image.max()
    
    # Create intervals
    interval = (max_val - min_val) / levels
    intervals = [min_val + i * interval for i in range(levels + 1)]
    
    # Calculate mean for each interval
    means = [sum(intervals[i:i+2]) / 2 for i in range(levels)]
    
    # Quantize by replacing with nearest mean
    quantized = np.array([means[min(np.searchsorted(intervals, val) - 1, levels - 1)] 
                         for val in flat_image])
    
    return quantized.reshape(image_input.shape).astype(np.uint8)

quantization_levels = 16
quantized_image = quantize(gray_image, quantization_levels)

print(f"Original Bit Depth: 8 bits (256 levels)")
print(f"Quantized Bit Depth: {int(np.log2(quantization_levels))} bits ({quantization_levels} levels)")
compression_ratio_quant = (int(np.log2(quantization_levels)) / 8) * 100
print(f"Quantization Compression Ratio: {compression_ratio_quant:.2f}%")

print("\nQuantization - Original Image:")
cv2_imshow(gray_image, "Quantization - Original (Grayscale)")

print("\nQuantization - Compressed Image:")
cv2_imshow(quantized_image, "Quantization - Compressed Image")


# =============================================================================
# COMPARISON OF ALL THREE METHODS
# =============================================================================
print("\n" + "=" * 70)
print("COMPARISON OF ALL THREE COMPRESSION TECHNIQUES")
print("=" * 70)

# Resize for comparison
size = (400, 300)
orig_resized = cv2.resize(gray_image, size)
rle_resized = cv2.resize(decompressed_rle, size)
dct_resized = cv2.resize(compressed_dct, size)
quant_resized = cv2.resize(quantized_image, size)

# Create comparison figure
plt.figure(figsize=(16, 12))

plt.subplot(2, 2, 1)
plt.imshow(orig_resized, cmap='gray')
plt.title('Original Image', fontsize=12, fontweight='bold')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(rle_resized, cmap='gray')
plt.title(f'RLE Compression\n({compression_ratio_rle:.2f}%)', fontsize=12, fontweight='bold')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(dct_resized, cmap='gray')
plt.title('DCT Compression\n(Low Frequency)', fontsize=12, fontweight='bold')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(quant_resized, cmap='gray')
plt.title(f'Scalar Quantization\n({compression_ratio_quant:.2f}%)', fontsize=12, fontweight='bold')
plt.axis('off')

plt.tight_layout()
plt.show()

# Print summary
print("\n" + "=" * 70)
print("COMPRESSION SUMMARY")
print("=" * 70)
print(f"1. RLE Compression:        {compression_ratio_rle:.2f}% (lossy)")
print(f"2. DCT Compression:        ~50% (lossy, keeps low frequencies)")
print(f"3. Quantization:           {compression_ratio_quant:.2f}% (lossy, 16->256 levels)")
print("=" * 70)
print("\n✓ Image Compression Mini Project Completed Successfully!")
print("=" * 70)
