"""
Vertebrae Binary Mask Generator
Extracts C2, C3, C4 vertebrae from color-annotated X-ray images
and outputs clean binary masks for segmentation training.

Took a while to get the color ranges right — HSV is choosy.
Red especially wraps around in HSV so needs two ranges.
"""

import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt


# ===========================
# CONFIGURATION
# ===========================

RAW_FOLDER = r"LAST Dataset\TESTIMG"
ANNOTATED_FOLDER = r"LAST Dataset\SEGIMG_NEW"
OUTPUT_FOLDER = r"LAST Dataset\BINIMG2"

# HSV ranges — tuned manually by eyeballing misclassified masks
# small saturation/value floor of 50 to ignore near-white/black noise

# C2 = purple
C2_LOWER = np.array([130, 50, 50])
C2_UPPER = np.array([170, 255, 255])

# C3 = red — red wraps around 0° in HSV so needs two ranges
# missed a lot of C3 until I added the second range
C3_LOWER_1 = np.array([0, 50, 50])
C3_UPPER_1 = np.array([10, 255, 255])
C3_LOWER_2 = np.array([170, 50, 50])
C3_UPPER_2 = np.array([180, 255, 255])

# C4 = cyan
C4_LOWER = np.array([80, 50, 50])
C4_UPPER = np.array([100, 255, 255])

# 500 worked well — anything smaller was catching
# annotation corner circles and scan artifacts
MIN_COMPONENT_AREA = 500

# 5px covers the collimator edges in most scans
BORDER_THICKNESS = 5

# 0.35 × diagonal = generous enough to keep all three vertebrae
# but drops truly stray blobs. went higher (0.5) first,
# was keeping too many artifacts
SPATIAL_OUTLIER_FACTOR = 0.35

# annotation borders are ~3px thick in our dataset
# eroding before merging stops adjacent vertebrae from touching
BORDER_EROSION_PX = 3


# ===========================
# CORE FUNCTIONS
# ===========================

def extract_color_mask(hsv_image, lower, upper):
    # straightforward HSV threshold
    # doing this in HSV not BGR because BGR ranges
    # for the same color vary too much across scans
    return cv2.inRange(hsv_image, lower, upper)


def fill_vertebra(mask):
    # annotation only draws the border, not the fill
    # need filled regions to extract morphological features later
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filled = np.zeros_like(mask)
    for contour in contours:
        cv2.drawContours(filled, [contour], -1, 255,
                         thickness=cv2.FILLED)
    return filled


def strip_annotation_border(filled_mask, erosion_px=BORDER_EROSION_PX):
    """
    Erodes each filled vertebra mask before combining them.

    The annotation tool draws a thick colored border (~3px).
    After filling, adjacent vertebrae sit close enough that
    their borders touch — they merge into one blob in the combined mask,
    which kills the feature extraction downstream.

    Eroding per vertebra BEFORE the bitwise_or merge creates a
    guaranteed gap between them. MORPH_ELLIPSE avoids the
    sharp corners that a rectangular kernel leaves behind.

    Tried doing this after combining — didn't work,
    the merged blob erodes as one shape and you lose
    the individual vertebra boundaries entirely.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * erosion_px + 1, 2 * erosion_px + 1)
    )
    return cv2.erode(filled_mask, kernel, iterations=1)


def remove_small_components(mask, min_area=100):
    # connected components gives us individual blobs
    # anything under min_area is noise — corner dots,
    # annotation artifacts, scan dust
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):  # skip 0 = background
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def remove_border_artifacts(mask, border_thickness=5):
    # anything touching the image edge is almost certainly
    # not a vertebra — collimator edges, scan frame artifacts
    h, w = mask.shape

    border_mask = np.zeros_like(mask)
    border_mask[:border_thickness, :] = 255
    border_mask[-border_thickness:, :] = 255
    border_mask[:, :border_thickness] = 255
    border_mask[:, -border_thickness:] = 255

    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    border_labels = set()
    for i in range(1, num_labels):
        component_mask = (labels == i).astype(np.uint8) * 255
        if cv2.bitwise_and(component_mask, border_mask).any():
            border_labels.add(i)

    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):
        if i not in border_labels:
            cleaned[labels == i] = 255

    return cleaned


def remove_spatial_outliers(mask, outlier_factor=SPATIAL_OUTLIER_FACTOR):
    """
    Drops blobs whose centroid is too far from the vertebra cluster.

    Some annotated images had a small reference circle in the corner
    that survived area filtering (it was big enough).
    Area-based removal wasn't reliable — circle size varied per scan.

    This approach finds the median centroid of all surviving blobs
    and drops anything too far from it. Works because C2/C3/C4
    are always spatially clustered — stray blobs are outliers.

    Using median not mean because one big outlier would pull
    the mean toward it and cause us to drop a real vertebra instead.
    """
    num_labels, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return mask  # nothing to filter

    h, w = mask.shape
    diagonal = np.sqrt(h**2 + w**2)
    max_dist = outlier_factor * diagonal

    fg_centroids = centroids[1:]  # skip background label 0
    fg_labels = np.arange(1, num_labels)

    if len(fg_centroids) == 0:
        return mask

    # median is robust — one stray blob doesn't shift it much
    median_cx = np.median(fg_centroids[:, 0])
    median_cy = np.median(fg_centroids[:, 1])

    cleaned = np.zeros_like(mask)
    for label, (cx, cy) in zip(fg_labels, fg_centroids):
        dist = np.sqrt((cx - median_cx)**2 + (cy - median_cy)**2)
        if dist <= max_dist:
            cleaned[labels == label] = 255

    return cleaned


def process_single_image(annotated_path, output_path):
    annotated_img = cv2.imread(annotated_path)
    if annotated_img is None:
        return False, "Could not read image"

    hsv = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2HSV)

    # extract each vertebra, fill, then strip border BEFORE combining
    # order matters — strip first, combine second
    c2_mask = extract_color_mask(hsv, C2_LOWER, C2_UPPER)
    c2_filled = strip_annotation_border(fill_vertebra(c2_mask))

    # C3 needs two masks because red wraps around in HSV
    c3_mask_1 = extract_color_mask(hsv, C3_LOWER_1, C3_UPPER_1)
    c3_mask_2 = extract_color_mask(hsv, C3_LOWER_2, C3_UPPER_2)
    c3_mask = cv2.bitwise_or(c3_mask_1, c3_mask_2)
    c3_filled = strip_annotation_border(fill_vertebra(c3_mask))

    c4_mask = extract_color_mask(hsv, C4_LOWER, C4_UPPER)
    c4_filled = strip_annotation_border(fill_vertebra(c4_mask))

    # merge all three vertebrae into one binary mask
    combined_mask = cv2.bitwise_or(c2_filled, c3_filled)
    combined_mask = cv2.bitwise_or(combined_mask, c4_filled)

    # closing fills tiny gaps between pixels from the HSV threshold
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined_mask = cv2.morphologyEx(
        combined_mask, cv2.MORPH_CLOSE, kernel_small
    )

    # cleaning pipeline — order matters here too
    # small components first, then border, then spatial outliers
    combined_mask = remove_small_components(
        combined_mask, min_area=MIN_COMPONENT_AREA
    )
    combined_mask = remove_border_artifacts(
        combined_mask, border_thickness=BORDER_THICKNESS
    )
    combined_mask = remove_spatial_outliers(
        combined_mask, outlier_factor=SPATIAL_OUTLIER_FACTOR
    )

    # final opening removes single-pixel noise that survived everything else
    combined_mask = cv2.morphologyEx(
        combined_mask, cv2.MORPH_OPEN, kernel_small
    )

    # force strict binary — some morphological ops leave 254s
    combined_mask = (combined_mask > 0).astype(np.uint8) * 255

    cv2.imwrite(output_path, combined_mask)
    return True, "Success"


def visualize_sample(annotated_path, mask_path):
    # quick sanity check — always run this on a few samples
    # before trusting the full batch output
    annotated = cv2.imread(annotated_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if annotated is None or mask is None:
        print("Error loading images for visualization")
        return

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

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
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    image_files = [
        f for f in os.listdir(ANNOTATED_FOLDER)
        if Path(f).suffix.lower() in image_extensions
    ]

    if not image_files:
        print("No images found in annotated folder")
        return

    print("=" * 60)
    print("VERTEBRAE MASK EXTRACTION")
    print("=" * 60)
    print(f"Annotated: {ANNOTATED_FOLDER}")
    print(f"Output:    {OUTPUT_FOLDER}")
    print(f"Images:    {len(image_files)}")
    print("=" * 60)

    success_count = 0
    failed_count = 0
    failed_files = []

    for idx, image_file in enumerate(image_files, 1):
        annotated_path = os.path.join(ANNOTATED_FOLDER, image_file)
        output_path = os.path.join(OUTPUT_FOLDER, image_file)

        success, message = process_single_image(
            annotated_path, output_path
        )

        if success:
            print(f"[{idx}/{len(image_files)}] ✓ {image_file}")
            success_count += 1
        else:
            print(f"[{idx}/{len(image_files)}] ✗ {image_file} — {message}")
            failed_count += 1
            failed_files.append(image_file)

    print()
    print("=" * 60)
    print(f"Done: {success_count} succeeded, {failed_count} failed")
    if failed_files:
        print("Failed files:")
        for f in failed_files:
            print(f"  {f}")
    print("=" * 60)

    # always visualize at least one — catches bad color range issues early
    if success_count > 0:
        sample_file = image_files[0]
        visualize_sample(
            os.path.join(ANNOTATED_FOLDER, sample_file),
            os.path.join(OUTPUT_FOLDER, sample_file)
        )


# ===========================
# RUN
# ===========================

if __name__ == "__main__":
    if not os.path.exists(ANNOTATED_FOLDER):
        print(f"Annotated folder not found: {ANNOTATED_FOLDER}")
        print("Update ANNOTATED_FOLDER path and rerun")
    else:
        process_batch()