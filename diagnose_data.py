"""
Diagnostic script to analyze alphabet recognition dataset
Identifies issues causing low accuracy
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from bidict import bidict

# Your encoder (0-indexed)
ENCODER = bidict({
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
    'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11,
    'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17,
    'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23,
    'Y': 24, 'Z': 25
})

print("=" * 60)
print("ALPHABET RECOGNITION DATASET DIAGNOSTIC")
print("=" * 60)

# Load data
print("\n1. Loading data...")
labels_raw = np.load('data/labels.npy', allow_pickle=True)
imgs = np.load('data/images.npy')

print(f"   Total samples: {len(labels_raw)}")
print(f"   Image shape: {imgs.shape}")
print(f"   Image dtype: {imgs.dtype}")

# Check for issues
print("\n2. Checking data quality...")

# 2.1 Check image statistics
print(f"\n   Image Statistics:")
print(f"   - Min pixel value: {imgs.min()}")
print(f"   - Max pixel value: {imgs.max()}")
print(f"   - Mean pixel value: {imgs.mean():.3f}")
print(f"   - Std dev: {imgs.std():.3f}")

# 2.2 Check for all-black or all-white images
black_images = (imgs < 5).all(axis=(1, 2))
white_images = (imgs > 250).all(axis=(1, 2))
print(f"\n   All-black images: {black_images.sum()}")
print(f"   All-white images: {white_images.sum()}")

# 2.3 Check for NaN or Inf
print(f"   NaN values: {np.isnan(imgs).sum()}")
print(f"   Inf values: {np.isinf(imgs).sum()}")

# 2.4 Check label distribution
print(f"\n3. Label Distribution:")
label_counts = Counter(labels_raw)
sorted_labels = sorted(label_counts.items())

total_samples = len(labels_raw)
for letter, count in sorted_labels:
    percentage = (count / total_samples) * 100
    bar = '█' * (count // 5)
    print(f"   {letter}: {count:3d} ({percentage:5.1f}%) {bar}")

# Check for imbalance
min_count = min(label_counts.values())
max_count = max(label_counts.values())
imbalance_ratio = max_count / min_count
print(f"\n   Imbalance ratio: {imbalance_ratio:.2f}x (max/min)")
if imbalance_ratio > 3:
    print(f"   ⚠️  HIGH IMBALANCE - Some classes have 3x+ more samples!")
elif imbalance_ratio > 1.5:
    print(f"   ⚠️  MODERATE IMBALANCE - Could affect training")
else:
    print(f"   ✓ Good balance")

# 2.5 Check for duplicate samples
print(f"\n4. Checking for duplicates...")
unique_samples = len(set(map(tuple, imgs.reshape(len(imgs), -1))))
duplicate_count = len(imgs) - unique_samples
print(f"   Total samples: {len(imgs)}")
print(f"   Unique samples: {unique_samples}")
print(f"   Duplicates: {duplicate_count}")
if duplicate_count > len(imgs) * 0.1:
    print(f"   ⚠️  More than 10% duplicates - Could indicate overfitting!")

# 2.6 Visualize samples per letter
print(f"\n5. Visualizing data distribution...")
fig, axes = plt.subplots(4, 7, figsize=(14, 8))
fig.suptitle('Sample Images from Each Letter', fontsize=16)

letters = list(ENCODER.keys())
for idx, letter in enumerate(letters):
    row = idx // 7
    col = idx % 7
    ax = axes[row, col]
    
    # Find an image with this label
    letter_indices = np.where(labels_raw == letter)[0]
    if len(letter_indices) > 0:
        sample_idx = letter_indices[0]
        ax.imshow(imgs[sample_idx], cmap='gray')
        ax.set_title(f'{letter} (n={len(letter_indices)})')
    ax.axis('off')

plt.tight_layout()
plt.show()

# 2.7 Analyze image variance per letter
print(f"\n6. Image quality per letter:")
print(f"   Letter | Count | Mean Intensity | Variance")
print(f"   " + "-" * 45)

for letter in letters:
    letter_indices = np.where(labels_raw == letter)[0]
    if len(letter_indices) > 0:
        letter_imgs = imgs[letter_indices]
        mean_intensity = letter_imgs.mean()
        variance = letter_imgs.var()
        print(f"   {letter:6s} | {len(letter_indices):5d} | {mean_intensity:14.3f} | {variance:8.3f}")

# 2.8 Recommendations
print(f"\n7. RECOMMENDATIONS:")
print(f"   " + "=" * 45)

issues = []

if duplicate_count > len(imgs) * 0.1:
    issues.append("High duplicate rate - Remove duplicate samples")

if imbalance_ratio > 3:
    issues.append("High class imbalance - Use weighted loss or oversampling")
elif imbalance_ratio > 1.5:
    issues.append("Moderate imbalance - Consider weighted loss")

if black_images.sum() > 5 or white_images.sum() > 5:
    issues.append("Found all-black/white images - Check data quality")

if imgs.min() < 0 or imgs.max() > 255:
    issues.append("Invalid pixel values - Check normalization")

if len(imgs) < 2000:
    issues.append("Small dataset (< 2000 samples) - Consider data augmentation")

if len(set(label_counts.values())) < len(ENCODER) * 0.8:
    issues.append("Some letters have very few samples - Need more balanced data")

if issues:
    for i, issue in enumerate(issues, 1):
        print(f"   {i}. {issue}")
else:
    print(f"   ✓ Data looks good! Other optimization needed:")
    print(f"   - Try data augmentation (rotation, scaling, noise)")
    print(f"   - Try different model architecture")
    print(f"   - Increase training epochs")
    print(f"   - Collect more training data")

print(f"\n" + "=" * 60)
