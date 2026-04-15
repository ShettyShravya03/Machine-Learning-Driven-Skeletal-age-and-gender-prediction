# =============================================================================
#  ENHANCED Feature Extraction – C2/C3/C4 Vertebrae (CVM Analysis)
#  Adds: endplate concavity, ant/post heights, axis ratios, slope angles,
#        convex hull features, disc gap heights, inter-vertebral distances
# =============================================================================

import os
import cv2
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from scipy.spatial import ConvexHull

# =============================================================================
#  HELPERS
# =============================================================================

def load_binary_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {mask_path}")
    return (mask > 0).astype(np.uint8)

def clean_binary_mask(mask):
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    return mask

def extract_age_from_filename(filename):
    name = os.path.splitext(filename)[0]
    age_str = name[-2:]
    if not age_str.isdigit():
        raise ValueError(f"Cannot extract age from filename: {filename}")
    return int(age_str)

def extract_gender_from_filename(filename):
    name = os.path.splitext(filename)[0]
    return int(name[-3])

# =============================================================================
#  NEW HELPER: endplate concavity depth
# =============================================================================

def endplate_concavity(region_mask, position="inferior"):
    """
    Measures the maximum concavity depth of the superior or inferior endplate.
    Fits a straight line between the two endpoints of the endplate row
    and measures the maximum perpendicular deviation inward.

    Parameters
    ----------
    region_mask : 2D binary ndarray  (cropped to vertebra bounding box)
    position    : "superior" or "inferior"

    Returns
    -------
    concavity_depth : float  (pixels; 0 if flat or convex)
    """
    rows, cols = np.where(region_mask > 0)
    if len(rows) == 0:
        return 0.0

    if position == "superior":
        target_row = rows.min()
    else:
        target_row = rows.max()

    endplate_cols = cols[rows == target_row]
    if len(endplate_cols) < 3:
        return 0.0

    c_min, c_max = endplate_cols.min(), endplate_cols.max()

    # Scan all rows near the endplate to find the deepest indentation
    span = max(1, (rows.max() - rows.min()) // 6)
    if position == "superior":
        row_range = range(target_row, min(target_row + span, rows.max()))
    else:
        row_range = range(max(rows.min(), target_row - span), target_row + 1)

    max_depth = 0.0
    for r in row_range:
        cols_at_r = cols[rows == r]
        if len(cols_at_r) == 0:
            continue
        c_left, c_right = cols_at_r.min(), cols_at_r.max()
        # Midpoint on the straight line between bounding corners
        line_mid_col = (c_min + c_max) / 2
        actual_mid_col = (c_left + c_right) / 2
        depth = abs(line_mid_col - actual_mid_col)
        if depth > max_depth:
            max_depth = depth

    return float(max_depth)

# =============================================================================
#  NEW HELPER: anterior / posterior body heights
# =============================================================================

def ant_post_heights(region_mask):
    """
    Returns:
      anterior_height  – column extent at leftmost 10% of width
      posterior_height – column extent at rightmost 10% of width
      ant_post_ratio   – anterior / posterior height
    """
    rows, cols = np.where(region_mask > 0)
    if len(cols) == 0:
        return 0.0, 0.0, 0.0

    c_min, c_max = cols.min(), cols.max()
    width = c_max - c_min
    band = max(1, int(width * 0.10))

    # Anterior (left side in lateral X-ray = ventral)
    ant_mask = cols <= (c_min + band)
    ant_rows = rows[ant_mask]
    ant_h = float(ant_rows.max() - ant_rows.min() + 1) if len(ant_rows) > 1 else 0.0

    # Posterior (right side = dorsal)
    post_mask = cols >= (c_max - band)
    post_rows = rows[post_mask]
    post_h = float(post_rows.max() - post_rows.min() + 1) if len(post_rows) > 1 else 0.0

    ratio = ant_h / post_h if post_h > 0 else 0.0
    return ant_h, post_h, ratio

# =============================================================================
#  NEW HELPER: inferior endplate slope angle
# =============================================================================

def endplate_slope_angle(region_mask, position="inferior"):
    """
    Fits a line through the inferior (or superior) endplate pixels
    and returns the angle in degrees from horizontal.
    """
    rows, cols = np.where(region_mask > 0)
    if len(rows) == 0:
        return 0.0

    target_row = rows.max() if position == "inferior" else rows.min()
    span = max(1, (rows.max() - rows.min()) // 8)

    if position == "inferior":
        mask = rows >= (target_row - span)
    else:
        mask = rows <= (target_row + span)

    r_ep, c_ep = rows[mask], cols[mask]
    if len(r_ep) < 3:
        return 0.0

    # Linear regression: row = f(col)
    try:
        slope, _, _, _, _ = np.polyfit(c_ep, r_ep, 1, full=False), None, None, None, None
        slope = np.polyfit(c_ep, r_ep, 1)[0]
        angle = np.degrees(np.arctan(slope))
    except Exception:
        angle = 0.0
    return float(angle)

# =============================================================================
#  NEW HELPER: convex hull features
# =============================================================================

def convex_hull_features(region_mask):
    """
    Returns:
      convex_hull_area  – area of the convex hull
      hull_fill_ratio   – actual area / hull area  (1 = perfect convex)
      hull_perimeter    – perimeter of convex hull
    """
    rows, cols = np.where(region_mask > 0)
    if len(rows) < 4:
        return 0.0, 0.0, 0.0
    points = np.column_stack([cols, rows]).astype(float)
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume          # in 2D, ConvexHull.volume = area
        hull_perim = hull.area           # in 2D, ConvexHull.area = perimeter
        actual_area = region_mask.sum()
        fill_ratio = actual_area / hull_area if hull_area > 0 else 0.0
        return float(hull_area), float(fill_ratio), float(hull_perim)
    except Exception:
        return 0.0, 0.0, 0.0

# =============================================================================
#  NEW HELPER: minor/major axis ratio  (from ellipse fit)
# =============================================================================

def minor_major_ratio(r):
    """skimage regionprops region → minor/major axis length ratio"""
    if r.major_axis_length > 0:
        return r.minor_axis_length / r.major_axis_length
    return 0.0

# =============================================================================
#  MAIN FEATURE EXTRACTION
# =============================================================================

def extract_features_from_mask(binary_mask, image_name):
    features = {}
    features["Image"]  = image_name
    features["Age"]    = extract_age_from_filename(image_name)
    features["Gender"] = extract_gender_from_filename(image_name)

    labeled  = label(binary_mask)
    regions  = regionprops(labeled)
    regions  = [r for r in regions if r.area > 100]

    if len(regions) < 3:
        print(f"⚠️  Less than 3 vertebrae detected in {image_name}")
        return None

    regions   = sorted(regions, key=lambda r: r.centroid[0])
    vertebrae = ["C2", "C3", "C4"]

    for v, r in zip(vertebrae, regions[:3]):
        minr, minc, maxr, maxc = r.bbox
        height    = maxr - minr
        width     = maxc - minc
        area      = r.area
        perimeter = r.perimeter

        # ── Original features ──────────────────────────────────────────────
        features[f"{v} Area"]         = area
        features[f"{v} Perimeter"]    = perimeter
        features[f"{v} Height"]       = height
        features[f"{v} Width"]        = width
        features[f"{v} Aspect Ratio"] = width / height if height else 0
        features[f"{v} Circularity"]  = (
            4 * np.pi * area / (perimeter ** 2) if perimeter else 0
        )
        features[f"{v} Eccentricity"] = r.eccentricity
        features[f"{v} Solidity"]     = r.solidity
        features[f"{v} Extent"]       = r.extent

        # ── NEW: minor/major axis ratio ────────────────────────────────────
        features[f"{v} Minor/Major Ratio"] = minor_major_ratio(r)

        # ── NEW: orientation angle (from ellipse) ──────────────────────────
        features[f"{v} Orientation"] = float(r.orientation)

        # Crop the vertebra binary mask for local analyses
        crop = binary_mask[minr:maxr, minc:maxc]

        # ── NEW: anterior / posterior height & ratio ───────────────────────
        ant_h, post_h, ap_ratio = ant_post_heights(crop)
        features[f"{v} Anterior Height"]      = ant_h
        features[f"{v} Posterior Height"]     = post_h
        features[f"{v} Ant/Post Height Ratio"]= ap_ratio

        # ── NEW: endplate concavity ────────────────────────────────────────
        features[f"{v} Superior Concavity"] = endplate_concavity(crop, "superior")
        features[f"{v} Inferior Concavity"] = endplate_concavity(crop, "inferior")

        # ── NEW: inferior endplate slope angle ────────────────────────────
        features[f"{v} Inferior Slope Angle"] = endplate_slope_angle(crop, "inferior")
        features[f"{v} Superior Slope Angle"] = endplate_slope_angle(crop, "superior")

        # ── NEW: convex hull features ──────────────────────────────────────
        hull_area, hull_fill, hull_perim = convex_hull_features(crop)
        features[f"{v} Convex Hull Area"]   = hull_area
        features[f"{v} Hull Fill Ratio"]    = hull_fill
        features[f"{v} Hull Perimeter"]     = hull_perim

    # ── Original inter-vertebral ratio features ────────────────────────────
    features["C3/C2 Area Ratio"]      = features["C3 Area"]   / features["C2 Area"]   if features["C2 Area"]   else 0
    features["C4/C2 Area Ratio"]      = features["C4 Area"]   / features["C2 Area"]   if features["C2 Area"]   else 0
    features["C4/C3 Area Ratio"]      = features["C4 Area"]   / features["C3 Area"]   if features["C3 Area"]   else 0
    features["C3/C2 Height Ratio"]    = features["C3 Height"] / features["C2 Height"] if features["C2 Height"] else 0
    features["C4/C3 Height Ratio"]    = features["C4 Height"] / features["C3 Height"] if features["C3 Height"] else 0
    features["C4/C2 Height Ratio"]    = features["C4 Height"] / features["C2 Height"] if features["C2 Height"] else 0
    features["C4/C2 Solidity Ratio"]  = features["C4 Solidity"] / features["C2 Solidity"] if features["C2 Solidity"] else 0
    features["C4/C3 Solidity Ratio"]  = features["C4 Solidity"] / features["C3 Solidity"] if features["C3 Solidity"] else 0

    # ── NEW: inter-vertebral disc gap (distance between bounding boxes) ────
    r_sorted = sorted(regions[:3], key=lambda r: r.centroid[0])
    gap_c2c3 = r_sorted[1].bbox[0] - r_sorted[0].bbox[2]   # top of C3 − bottom of C2
    gap_c3c4 = r_sorted[2].bbox[0] - r_sorted[1].bbox[2]   # top of C4 − bottom of C3
    features["C2-C3 Gap (px)"] = max(0, gap_c2c3)
    features["C3-C4 Gap (px)"] = max(0, gap_c3c4)
    features["Gap Ratio C3/C2"] = (
        features["C3-C4 Gap (px)"] / features["C2-C3 Gap (px)"]
        if features["C2-C3 Gap (px)"] > 0 else 0
    )

    # ── NEW: Solidity trend (C2→C3→C4 slope) ──────────────────────────────
    s2 = features["C2 Solidity"]
    s3 = features["C3 Solidity"]
    s4 = features["C4 Solidity"]
    features["Solidity Trend (C2→C4)"] = (s4 - s2) / 2   # linear slope

    # ── NEW: Circularity trend ─────────────────────────────────────────────
    ci2 = features["C2 Circularity"]
    ci3 = features["C3 Circularity"]
    ci4 = features["C4 Circularity"]
    features["Circularity Trend (C2→C4)"] = (ci4 - ci2) / 2

    # ── NEW: normalized area by image pixel area (scale-invariant) ─────────
    total_mask_px = binary_mask.sum()
    for v in vertebrae:
        features[f"{v} Norm Area"] = features[f"{v} Area"] / total_mask_px if total_mask_px else 0

    return features

# =============================================================================
#  CONFIG
# =============================================================================
MASK_DIR    = r"LAST Dataset\BINIMG"
OUTPUT_FILE = r"final_enhanced.xlsx"

rows = []

for file in sorted(os.listdir(MASK_DIR)):
    if file.lower().endswith(".png"):
        mask_path = os.path.join(MASK_DIR, file)
        mask = load_binary_mask(mask_path)
        mask = clean_binary_mask(mask)
        row = extract_features_from_mask(mask, file)
        if row is not None:
            rows.append(row)

df = pd.DataFrame(rows)
df.to_excel(OUTPUT_FILE, index=False)

print(f"Total samples   : {len(df)}")
print(f"Total features  : {len(df.columns) - 3}  (excl. Image, Age, Gender)")
print(f"Age range       : {df['Age'].min()} – {df['Age'].max()}")
print(f"Gender dist     : Male={(df['Gender']==0).sum()}  Female={(df['Gender']==1).sum()}")
print(f"✅ Enhanced feature extraction complete → {OUTPUT_FILE}")