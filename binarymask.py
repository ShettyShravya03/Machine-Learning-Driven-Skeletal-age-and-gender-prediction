"""
Vertebrae Binary Mask Generator for Google Colab
Extracts C2, C3, C4 vertebrae from color-annotated images

Usage in Colab:
1. Upload this script
2. Mount Google Drive or upload your images
3. Run the script with your folder paths
"""

import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

# ===========================
# CONFIGURATION
# ===========================

# Set your folder paths here
RAW_FOLDER = r"LAST Dataset\TESTIMG"           # Raw images folder
ANNOTATED_FOLDER = r"LAST Dataset\SEGIMG_NEW"      # Annotated images folder
OUTPUT_FOLDER = r"LAST Dataset\BINIMG2"           # Output folder for binary masks

# Color ranges in HSV for each vertebra
# Purple for C2
C2_LOWER = np.array([130, 50, 50])
C2_UPPER = np.array([170, 255, 255])

# Red for C3 (two ranges due to HSV wraparound)
C3_LOWER_1 = np.array([0, 50, 50])
C3_UPPER_1 = np.array([10, 255, 255])
C3_LOWER_2 = np.array([170, 50, 50])
C3_UPPER_2 = np.array([180, 255, 255])

# Cyan for C4
C4_LOWER = np.array([80, 50, 50])
C4_UPPER = np.array([100, 255, 255])

# Processing parameters
MIN_COMPONENT_AREA = 500      # Minimum area to keep — raised to filter small blobs like corner circles
BORDER_THICKNESS = 5          # Border artifact removal thickness
SPATIAL_OUTLIER_FACTOR = 0.35 # Components whose centroid is more than this fraction of image
                               # diagonal away from the median centroid are dropped
BORDER_EROSION_PX = 3         # Pixels to erode from each filled vertebra BEFORE combining.
                               # Strips the annotation border line so adjacent vertebrae
                               # don't touch/merge. Increase if annotation borders are thicker.


# ===========================
# CORE FUNCTIONS
# ===========================

def extract_color_mask(hsv_image, lower, upper):
    """Extract mask for a specific color range"""
    return cv2.inRange(hsv_image, lower, upper)


def fill_vertebra(mask):
    """Fill holes in vertebra mask using contour filling"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    for contour in contours:
        cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)
    return filled


def strip_annotation_border(filled_mask, erosion_px=BORDER_EROSION_PX):
    """
    Erode a single filled vertebra mask to remove the annotation border line.

    Why per-vertebra erosion before combining:
      - The colored annotation is a thick drawn border, not just a 1-px outline.
      - When two vertebrae sit close together their border pixels overlap/touch
        after filling, causing them to merge into one blob in the combined mask.
      - Eroding each vertebra individually BEFORE the bitwise_or merge shrinks
        every shape inward by `erosion_px` pixels, creating a guaranteed gap
        between adjacent vertebrae.
      - Using MORPH_ELLIPSE avoids sharp corners that rectangular kernels produce.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * erosion_px + 1, 2 * erosion_px + 1)
    )
    return cv2.erode(filled_mask, kernel, iterations=1)


def remove_small_components(mask, min_area=100):
    """Remove small connected components (noise)"""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255

    return cleaned


def remove_border_artifacts(mask, border_thickness=5):
    """Remove components touching the image border"""
    h, w = mask.shape

    # Create border mask
    border_mask = np.zeros_like(mask)
    border_mask[:border_thickness, :] = 255
    border_mask[-border_thickness:, :] = 255
    border_mask[:, :border_thickness] = 255
    border_mask[:, -border_thickness:] = 255

    # Find components
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Identify border-touching labels
    border_labels = set()
    for i in range(1, num_labels):
        component_mask = (labels == i).astype(np.uint8) * 255
        if cv2.bitwise_and(component_mask, border_mask).any():
            border_labels.add(i)

    # Create clean mask
    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):
        if i not in border_labels:
            cleaned[labels == i] = 255

    return cleaned


def remove_spatial_outliers(mask, outlier_factor=SPATIAL_OUTLIER_FACTOR):
    """
    Remove components whose centroid is spatially isolated from the main vertebrae cluster.

    Strategy:
      1. Find the centroid of every connected component.
      2. Compute the median centroid across all components (robust center estimate).
      3. Compute the image diagonal as a reference distance scale.
      4. Drop any component whose centroid is farther than
         (outlier_factor × diagonal) from the median centroid.

    This eliminates stray blobs (like the corner circle) without using
    area ranking or any assumption about which vertebra is biggest.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:          # Nothing but background
        return mask

    h, w = mask.shape
    diagonal = np.sqrt(h**2 + w**2)
    max_dist = outlier_factor * diagonal

    # Centroids of foreground components (skip label 0 = background)
    fg_centroids = centroids[1:]          # shape: (N, 2)  [cx, cy]
    fg_labels    = np.arange(1, num_labels)

    if len(fg_centroids) == 0:
        return mask

    # Robust centre: median of all component centroids
    median_cx = np.median(fg_centroids[:, 0])
    median_cy = np.median(fg_centroids[:, 1])

    # Keep only components within max_dist of the median centroid
    cleaned = np.zeros_like(mask)
    for label, (cx, cy) in zip(fg_labels, fg_centroids):
        dist = np.sqrt((cx - median_cx)**2 + (cy - median_cy)**2)
        if dist <= max_dist:
            cleaned[labels == label] = 255

    return cleaned


def process_single_image(annotated_path, output_path):
    """Process a single annotated image to extract vertebrae mask"""
    # Read annotated image
    annotated_img = cv2.imread(annotated_path)
    if annotated_img is None:
        return False, "Could not read image"

    # Convert to HSV
    hsv = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2HSV)

    # Extract C2 (purple) — fill then strip the border line
    c2_mask = extract_color_mask(hsv, C2_LOWER, C2_UPPER)
    c2_filled = strip_annotation_border(fill_vertebra(c2_mask))

    # Extract C3 (red - combine two ranges) — fill then strip the border line
    c3_mask_1 = extract_color_mask(hsv, C3_LOWER_1, C3_UPPER_1)
    c3_mask_2 = extract_color_mask(hsv, C3_LOWER_2, C3_UPPER_2)
    c3_mask = cv2.bitwise_or(c3_mask_1, c3_mask_2)
    c3_filled = strip_annotation_border(fill_vertebra(c3_mask))

    # Extract C4 (cyan) — fill then strip the border line
    c4_mask = extract_color_mask(hsv, C4_LOWER, C4_UPPER)
    c4_filled = strip_annotation_border(fill_vertebra(c4_mask))

    # Combine all vertebrae
    combined_mask = cv2.bitwise_or(c2_filled, c3_filled)
    combined_mask = cv2.bitwise_or(combined_mask, c4_filled)

    # Morphological cleaning
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_small)

    # Remove small noise
    combined_mask = remove_small_components(combined_mask, min_area=MIN_COMPONENT_AREA)

    # Remove border artifacts
    combined_mask = remove_border_artifacts(combined_mask, border_thickness=BORDER_THICKNESS)

    # Remove spatially isolated blobs (e.g. corner circles, stray annotations)
    combined_mask = remove_spatial_outliers(combined_mask, outlier_factor=SPATIAL_OUTLIER_FACTOR)

    # Final opening
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small)

    # Ensure binary output (0 and 255)
    combined_mask = (combined_mask > 0).astype(np.uint8) * 255

    # Save mask
    cv2.imwrite(output_path, combined_mask)

    return True, "Success"


def visualize_sample(annotated_path, mask_path):
    """Visualize annotated image and generated mask side by side"""
    annotated = cv2.imread(annotated_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if annotated is None or mask is None:
        print("Error loading images for visualization")
        return

    # Convert BGR to RGB for matplotlib
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(annotated_rgb)
    axes[0].set_title('Annotated Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Binary Mask', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


# ===========================
# MAIN PROCESSING
# ===========================

def process_batch():
    """Process all images in the annotated folder"""

    # Create output directory
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

    # Get list of images
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    image_files = [f for f in os.listdir(ANNOTATED_FOLDER)
                   if Path(f).suffix.lower() in image_extensions]

    if not image_files:
        print("❌ No images found in annotated folder!")
        return

    print("="*60)
    print("VERTEBRAE MASK EXTRACTION")
    print("="*60)
    print(f"Annotated folder: {ANNOTATED_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Total images: {len(image_files)}")
    print("="*60)
    print()

    # Process each image
    success_count = 0
    failed_count = 0
    failed_files = []

    for idx, image_file in enumerate(image_files, 1):
        annotated_path = os.path.join(ANNOTATED_FOLDER, image_file)
        output_path = os.path.join(OUTPUT_FOLDER, image_file)

        success, message = process_single_image(annotated_path, output_path)

        if success:
            print(f"[{idx}/{len(image_files)}] ✓ {image_file}")
            success_count += 1
        else:
            print(f"[{idx}/{len(image_files)}] ❌ {image_file} - {message}")
            failed_count += 1
            failed_files.append(image_file)

    # Print summary
    print()
    print("="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Total images: {len(image_files)}")
    print(f"✓ Successfully processed: {success_count}")
    print(f"❌ Failed: {failed_count}")

    if failed_files:
        print("\nFailed files:")
        for f in failed_files:
            print(f"  - {f}")

    print("="*60)

    # Show sample visualization
    if success_count > 0:
        print("\nGenerating sample visualization...")
        sample_file = image_files[0]
        visualize_sample(
            os.path.join(ANNOTATED_FOLDER, sample_file),
            os.path.join(OUTPUT_FOLDER, sample_file)
        )


# ===========================
# RUN PROCESSING
# ===========================

if __name__ == "__main__":
    print("\n🔬 Vertebrae Binary Mask Generator")
    print("=" * 60)

    # Check if folders exist
    if not os.path.exists(ANNOTATED_FOLDER):
        print(f"❌ Error: Annotated folder not found: {ANNOTATED_FOLDER}")
        print("\nPlease:")
        print("1. Upload your annotated images to Colab")
        print("2. Update ANNOTATED_FOLDER path in the script")
        print("3. Run again")
    else:
        # Process all images
        process_batch()

        print("\n✅ Done! Check the BINMASK folder for your binary masks.")