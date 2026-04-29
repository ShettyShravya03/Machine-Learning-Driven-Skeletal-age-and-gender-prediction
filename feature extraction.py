# =============================================================================
#  Feature Extraction – C2/C3/C4 Vertebrae (CVM Analysis)
#
#  Takes binary masks from the mask generator and pulls out
#  morphological measurements per vertebra + inter-vertebral relationships.
#
#  Went through 3 versions of this — first used only basic regionprops,
#  then added endplate features after reading orthodontic literature on
#  what clinicians actually measure. The hull and gap features came last
#  when MAE was still high and I needed more discriminative signal.
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
    # boolean threshold — some saved masks had 254s from morphological ops
    return (mask > 0).astype(np.uint8)


def clean_binary_mask(mask):
    # close first to seal tiny gaps between pixels,
    # then open to remove single-pixel noise that survived saving
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    return mask


def extract_age_from_filename(filename):
    # filenames end in age — e.g. patient_F_25.png → age 25
    # last two characters before extension
    name = os.path.splitext(filename)[0]
    age_str = name[-2:]
    if not age_str.isdigit():
        raise ValueError(f"Cannot extract age from filename: {filename}")
    return int(age_str)


def extract_gender_from_filename(filename):
    # third character from end — 0=Male, 1=Female
    # e.g. patient_025.png → name[-3] = '0'
    name = os.path.splitext(filename)[0]
    return int(name[-3])


# =============================================================================
#  ENDPLATE CONCAVITY
# =============================================================================

def endplate_concavity(region_mask, position="inferior"):
    """
    Measures how concave the superior or inferior endplate is.

    Clinically this matters — endplate concavity increases during
    adolescent growth and is one of the key CVM stage indicators.
    Missed this feature in v1 and MAE dropped after adding it.

    Strategy: find the endplate row, draw a straight line between
    its leftmost and rightmost points, measure max deviation inward.
    A flat endplate → depth ≈ 0. Concave → depth > 0.
    """
    rows, cols = np.where(region_mask > 0)
    if len(rows) == 0:
        return 0.0

    target_row = rows.min() if position == "superior" else rows.max()

    endplate_cols = cols[rows == target_row]
    if len(endplate_cols) < 3:
        return 0.0  # too few pixels to measure meaningfully

    c_min, c_max = endplate_cols.min(), endplate_cols.max()

    # scan a band near the endplate — 1/6 of vertebra height
    # going too deep picks up body features, not just the endplate
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
        line_mid = (c_min + c_max) / 2
        actual_mid = (c_left + c_right) / 2
        depth = abs(line_mid - actual_mid)
        if depth > max_depth:
            max_depth = depth

    return float(max_depth)


# =============================================================================
#  ANTERIOR / POSTERIOR HEIGHT
# =============================================================================

def ant_post_heights(region_mask):
    """
    Measures vertebra height at the anterior (front) and posterior (back) edges.

    In lateral X-rays, anterior = left side of the vertebra in the image.
    The anterior/posterior height ratio changes significantly across CVM stages —
    early stages show more wedging, mature stages are more rectangular.

    Using 10% band at each edge — narrow enough to capture the true edge,
    wide enough to not be thrown off by a single noisy pixel.
    """
    rows, cols = np.where(region_mask > 0)
    if len(cols) == 0:
        return 0.0, 0.0, 0.0

    c_min, c_max = cols.min(), cols.max()
    width = c_max - c_min
    band = max(1, int(width * 0.10))

    ant_mask = cols <= (c_min + band)
    ant_rows = rows[ant_mask]
    ant_h = float(ant_rows.max() - ant_rows.min() + 1) \
        if len(ant_rows) > 1 else 0.0

    post_mask = cols >= (c_max - band)
    post_rows = rows[post_mask]
    post_h = float(post_rows.max() - post_rows.min() + 1) \
        if len(post_rows) > 1 else 0.0

    # ratio is more scale-invariant than raw heights
    # two patients with different image resolutions still compare
    ratio = ant_h / post_h if post_h > 0 else 0.0
    return ant_h, post_h, ratio


# =============================================================================
#  ENDPLATE SLOPE ANGLE
# =============================================================================

def endplate_slope_angle(region_mask, position="inferior"):
    """
    Angle of the endplate from horizontal, in degrees.

    Tilted endplates indicate vertebral rotation during growth —
    clinically relevant for distinguishing CVM stages 3 and 4.
    Added this after noticing the model was confusing those two stages.

    Uses linear regression on the endplate pixel coordinates.
    np.polyfit on (col, row) gives slope in row/col units → arctan → degrees.
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

    try:
        slope = np.polyfit(c_ep, r_ep, 1)[0]
        angle = np.degrees(np.arctan(slope))
    except Exception:
        # degenerate case — vertical line, no slope defined
        angle = 0.0

    return float(angle)


# =============================================================================
#  CONVEX HULL FEATURES
# =============================================================================

def convex_hull_features(region_mask):
    """
    Convex hull area, fill ratio, and perimeter.

    Hull fill ratio (actual area / hull area) captures how irregular
    the vertebra boundary is. Mature vertebrae are more rectangular
    and closer to convex — higher fill ratio. Immature vertebrae have
    more irregular edges — lower fill ratio.

    Note: in scipy 2D, ConvexHull.volume = area and .area = perimeter.
    Confusing naming but that's scipy's convention for n-dimensional hulls.
    """
    rows, cols = np.where(region_mask > 0)
    if len(rows) < 4:
        return 0.0, 0.0, 0.0  # need at least 4 points for a hull

    points = np.column_stack([cols, rows]).astype(float)
    try:
        hull = ConvexHull(points)
        hull_area  = hull.volume   # area in 2D
        hull_perim = hull.area     # perimeter in 2D
        actual_area = region_mask.sum()
        fill_ratio = actual_area / hull_area if hull_area > 0 else 0.0
        return float(hull_area), float(fill_ratio), float(hull_perim)
    except Exception:
        # QHull fails on collinear points — return zeros rather than crash
        return 0.0, 0.0, 0.0


# =============================================================================
#  MINOR/MAJOR AXIS RATIO
# =============================================================================

def minor_major_ratio(r):
    # compact way to get shape elongation from regionprops ellipse fit
    # ratio close to 1 = circular, close to 0 = very elongated
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

    # drop tiny regions — anything under 100px is annotation noise
    # not a real vertebra
    regions = [r for r in regions if r.area > 100]

    if len(regions) < 3:
        # happens when two vertebrae merge in the mask
        # or segmentation missed one entirely
        # returning None so the batch loop can skip and log it
        print(f"Warning: fewer than 3 vertebrae in {image_name}")
        return None

    # sort top to bottom by centroid row — C2 is highest in the image
    regions   = sorted(regions, key=lambda r: r.centroid[0])
    vertebrae = ["C2", "C3", "C4"]

    for v, r in zip(vertebrae, regions[:3]):
        minr, minc, maxr, maxc = r.bbox
        height    = maxr - minr
        width     = maxc - minc
        area      = r.area
        perimeter = r.perimeter

        # --- basic shape features ---
        # these were in v1 — still the strongest predictors per SHAP
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

        # axis ratio from fitted ellipse — more stable than raw aspect ratio
        features[f"{v} Minor/Major Ratio"] = minor_major_ratio(r)

        # orientation from ellipse fit — captures vertebra tilt
        features[f"{v} Orientation"] = float(r.orientation)

        # crop to vertebra bounding box for local measurements
        # working on the crop not the full mask avoids
        # accidentally picking up pixels from adjacent vertebrae
        crop = binary_mask[minr:maxr, minc:maxc]

        # --- anterior/posterior heights ---
        # added in v2 after reading Baccetti et al. CVM staging criteria
        ant_h, post_h, ap_ratio = ant_post_heights(crop)
        features[f"{v} Anterior Height"]       = ant_h
        features[f"{v} Posterior Height"]      = post_h
        features[f"{v} Ant/Post Height Ratio"] = ap_ratio

        # --- endplate concavity ---
        # one of the primary CVM stage markers in clinical literature
        # C3 inferior concavity specifically separates stage 2 from 3
        features[f"{v} Superior Concavity"]   = endplate_concavity(crop, "superior")
        features[f"{v} Inferior Concavity"]   = endplate_concavity(crop, "inferior")

        # --- endplate slope ---
        # helps distinguish stages 3/4 which were getting confused
        features[f"{v} Inferior Slope Angle"] = endplate_slope_angle(crop, "inferior")
        features[f"{v} Superior Slope Angle"] = endplate_slope_angle(crop, "superior")

        # --- convex hull features ---
        # fill ratio was more predictive than I expected —
        # showed up in top 10 SHAP features for age
        hull_area, hull_fill, hull_perim = convex_hull_features(crop)
        features[f"{v} Convex Hull Area"]  = hull_area
        features[f"{v} Hull Fill Ratio"]   = hull_fill
        features[f"{v} Hull Perimeter"]    = hull_perim

    # --- inter-vertebral ratios ---
    # ratios are more robust than raw values — normalise out
    # image scale differences between scanning devices
    features["C3/C2 Area Ratio"]   = features["C3 Area"]   / features["C2 Area"]   if features["C2 Area"]   else 0
    features["C4/C2 Area Ratio"]   = features["C4 Area"]   / features["C2 Area"]   if features["C2 Area"]   else 0
    features["C4/C3 Area Ratio"]   = features["C4 Area"]   / features["C3 Area"]   if features["C3 Area"]   else 0
    features["C3/C2 Height Ratio"] = features["C3 Height"] / features["C2 Height"] if features["C2 Height"] else 0
    features["C4/C3 Height Ratio"] = features["C4 Height"] / features["C3 Height"] if features["C3 Height"] else 0
    features["C4/C2 Height Ratio"] = features["C4 Height"] / features["C2 Height"] if features["C2 Height"] else 0
    features["C4/C2 Solidity Ratio"] = features["C4 Solidity"] / features["C2 Solidity"] if features["C2 Solidity"] else 0
    features["C4/C3 Solidity Ratio"] = features["C4 Solidity"] / features["C3 Solidity"] if features["C3 Solidity"] else 0

    # --- inter-vertebral disc gaps ---
    # gap = space between bounding boxes of adjacent vertebrae
    # disc height changes are a known CVM maturity indicator
    # using bbox not centroid distance — more anatomically accurate
    r_sorted = sorted(regions[:3], key=lambda r: r.centroid[0])
    gap_c2c3 = r_sorted[1].bbox[0] - r_sorted[0].bbox[2]  # C3 top − C2 bottom
    gap_c3c4 = r_sorted[2].bbox[0] - r_sorted[1].bbox[2]  # C4 top − C3 bottom

    # max(0, gap) — negative means vertebrae overlap in mask,
    # which happens when erosion wasn't enough. clamp to 0.
    features["C2-C3 Gap (px)"]  = max(0, gap_c2c3)
    features["C3-C4 Gap (px)"]  = max(0, gap_c3c4)
    features["Gap Ratio C3/C2"] = (
        features["C3-C4 Gap (px)"] / features["C2-C3 Gap (px)"]
        if features["C2-C3 Gap (px)"] > 0 else 0
    )

    # --- trend features: C2→C4 slope ---
    # captures systematic change across vertebrae rather than per-vertebra values
    # linear slope over 3 points = (last - first) / 2
    s2, s3, s4 = features["C2 Solidity"], features["C3 Solidity"], features["C4 Solidity"]
    features["Solidity Trend (C2→C4)"] = (s4 - s2) / 2

    ci2, ci3, ci4 = features["C2 Circularity"], features["C3 Circularity"], features["C4 Circularity"]
    features["Circularity Trend (C2→C4)"] = (ci4 - ci2) / 2

    # --- scale-invariant normalised area ---
    # raw pixel area varies with image resolution and patient size
    # dividing by total mask pixels makes it resolution-independent
    total_mask_px = binary_mask.sum()
    for v in vertebrae:
        features[f"{v} Norm Area"] = (
            features[f"{v} Area"] / total_mask_px if total_mask_px else 0
        )

    return features


# =============================================================================
#  BATCH PROCESSING
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
        # None means fewer than 3 vertebrae detected —
        # logged inside the function, skipped here

df = pd.DataFrame(rows)
df.to_excel(OUTPUT_FILE, index=False)

print(f"Total samples  : {len(df)}")
print(f"Total features : {len(df.columns) - 3}  (excl. Image, Age, Gender)")
print(f"Age range      : {df['Age'].min()} – {df['Age'].max()}")
print(f"Gender dist    : Male={(df['Gender']==0).sum()}  Female={(df['Gender']==1).sum()}")
print(f"Saved to       : {OUTPUT_FILE}")