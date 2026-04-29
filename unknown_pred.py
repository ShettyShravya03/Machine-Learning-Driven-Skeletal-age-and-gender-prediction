"""
unknown_pred.py  –  CVM Full Inference Pipeline  v8  (FIXED)

Fixes over v7:
  1. Feature names now exactly match feature_extraction.py (training):
       Gap C2_C3 / Gap C3_C4           → C2-C3 Gap (px) / C3-C4 Gap (px)
       Circularity Trend (C2->C4)       → Circularity Trend (C2→C4)  [Unicode →]
       {v} Concavity (wrong formula)    → removed
       Missing: Solidity Trend (C2→C4), Gap Ratio C3/C2, C2-C3 Gap (px),
                {v} Superior/Inferior Concavity, {v} Anterior/Posterior Height,
                {v} Ant/Post Height Ratio, C2 Norm Area
  2. Hull Perimeter: now uses scipy ConvexHull (was incorrectly set to
     regular perimeter).
  3. Norm Area: now area / total_mask_pixels for all 3 vertebrae (was
     bbox_h*bbox_w / C2_bbox — a completely different quantity).
  4. Slope angles: now use proper linear regression over endplate pixels
     (was just ±orientation angle from regionprops ellipse fit).
  5. Added startup feature audit — crashes loudly if any required feature
     name is missing instead of silently using 0.0.
  6. Area ratios added: C3/C2, C4/C2, C4/C3 (were missing from v7).

Architecture reverse-engineered from the TRAINING script to match the saved
checkpoint exactly.

ConvBlock layout (7 sequential entries):
  .block.0  Conv2d  (bias=False)
  .block.1  BatchNorm2d
  .block.2  ReLU
  .block.3  Identity  (dropout_p=0 for enc/dec) | Dropout2d (bottleneck)
  .block.4  Conv2d  (bias=False)
  .block.5  BatchNorm2d
  .block.6  ReLU

AttentionGate Conv2d:  bias=False
Decoder concat:  cat([skip, x])
"""

import cv2
import torch
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from skimage.measure import label, regionprops
from scipy.spatial import ConvexHull
import torch.nn as nn
import torch.nn.functional as F
import base64
import io
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# =====================================================================
#  CONFIG
# =====================================================================
UNET_MODEL_PATH   = "attention_unet_scratch_best.pth"
AGE_MODEL_PATH    = "best_age_model.pkl"
AGE_FEAT_PATH     = "age_selected_features.pkl"
GENDER_MODEL_PATH = "best_gender_model.pkl"
GENDER_FEAT_PATH  = "gender_selected_features.pkl"

IMG_SIZE  = 256
THRESHOLD = 0.5
DEVICE    = torch.device("cpu")

print(f"Using device: {DEVICE}")


# =====================================================================
#  ARCHITECTURE  –  must match the training script exactly
# =====================================================================

class ConvBlock(nn.Module):
    """
    Matches training ConvBlock exactly:
      block.0  Conv2d      (bias=False)
      block.1  BatchNorm2d
      block.2  ReLU
      block.3  Identity    (dropout_p=0.0 for encoders/decoders)
               Dropout2d   (dropout_p=0.4 for bottleneck)
      block.4  Conv2d      (bias=False)
      block.5  BatchNorm2d
      block.6  ReLU
    """
    def __init__(self, in_ch, out_ch, dropout_p=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),   # .block.0
            nn.BatchNorm2d(out_ch),                                  # .block.1
            nn.ReLU(inplace=True),                                   # .block.2
            nn.Dropout2d(dropout_p) if dropout_p > 0.0
                else nn.Identity(),                                  # .block.3
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),   # .block.4
            nn.BatchNorm2d(out_ch),                                  # .block.5
            nn.ReLU(inplace=True),                                   # .block.6
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    """
    Matches training AttentionGate exactly.
    All Conv2d layers use bias=False, matching the saved checkpoint.
    g  = gating signal (upsampled decoder feature)
    x  = skip-connection feature map (encoder)
    """
    def __init__(self, g_ch, x_ch, inter_ch):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(g_ch,   inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(x_ch,   inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1,      kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g_up = F.interpolate(self.W_g(g), size=x.shape[2:],
                             mode="bilinear", align_corners=False)
        att  = self.relu(g_up + self.W_x(x))
        att  = self.psi(att)
        return x * att


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1,
                 features=(64, 128, 256, 512), bottleneck_dropout=0.4):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────
        self.encoders = nn.ModuleList()
        self.pools    = nn.ModuleList()
        ch = in_channels
        for feat in features:
            self.encoders.append(ConvBlock(ch, feat))
            self.pools.append(nn.MaxPool2d(2))
            ch = feat

        # ── Bottleneck ────────────────────────────────────────────────
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2,
                                    dropout_p=bottleneck_dropout)

        # ── Decoder + Attention Gates ─────────────────────────────────
        self.upconvs   = nn.ModuleList()
        self.att_gates = nn.ModuleList()
        self.decoders  = nn.ModuleList()
        ch = features[-1] * 2
        for feat in reversed(features):
            inter_ch = feat // 2
            self.upconvs.append(nn.ConvTranspose2d(ch, feat, 2, stride=2))
            self.att_gates.append(
                AttentionGate(g_ch=feat, x_ch=feat, inter_ch=inter_ch)
            )
            self.decoders.append(ConvBlock(feat * 2, feat))
            ch = feat

        # ── Output head ───────────────────────────────────────────────
        self.head = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up, att, dec, skip in zip(
            self.upconvs, self.att_gates, self.decoders, reversed(skips)
        ):
            x    = up(x)
            skip = att(g=x, x=skip)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:],
                                  mode="bilinear", align_corners=False)
            x = dec(torch.cat([skip, x], dim=1))   # skip FIRST, then x

        return self.head(x)


# =====================================================================
#  LOAD ALL MODELS
# =====================================================================
print("Loading models ...")

unet = AttentionUNet().to(DEVICE)
unet.load_state_dict(torch.load(UNET_MODEL_PATH, map_location=DEVICE))
unet.eval()
print(f"  Attention U-Net  : loaded from {UNET_MODEL_PATH}")

age_model    = joblib.load(AGE_MODEL_PATH)
age_feats    = joblib.load(AGE_FEAT_PATH)

gender_model = joblib.load(GENDER_MODEL_PATH)
gender_feats = joblib.load(GENDER_FEAT_PATH)

print(f"  Age model        : {type(age_model).__name__}  ({len(age_feats)} features)")
print(f"  Gender model     : {type(gender_model).__name__}  ({len(gender_feats)} features)")
print("All models loaded successfully ✓")


# =====================================================================
#  FEATURE AUDIT  –  runs once at startup
#  Crashes loudly if any required feature name is not produced by
#  _extract_all_features(), instead of silently falling back to 0.0
#  and giving corrupt predictions.
# =====================================================================
# These are all features that _extract_all_features() is expected to emit.
# Update this set if you add or rename features.
_PRODUCED_FEATURES: set | None = None   # populated after first extraction call

def _audit_features(computed: dict) -> None:
    """Called once to verify all model features can be found."""
    missing_age    = [f for f in age_feats    if f not in computed]
    missing_gender = [f for f in gender_feats if f not in computed]
    if missing_age or missing_gender:
        lines = []
        if missing_age:
            lines.append(f"  Age model needs: {missing_age}")
        if missing_gender:
            lines.append(f"  Gender model needs: {missing_gender}")
        raise KeyError(
            "FEATURE AUDIT FAILED — the following features required by the "
            "models are not produced by _extract_all_features().\n"
            + "\n".join(lines)
            + "\nFix the feature names so they exactly match feature_extraction.py."
        )
    print(f"Feature audit passed ✓  "
          f"(age: {len(age_feats)} features, gender: {len(gender_feats)} features)")


# =====================================================================
#  HELPER FUNCTIONS
#  Ported directly from feature_extraction.py so inference features
#  are computed identically to training features.
# =====================================================================

def _endplate_concavity(region_mask: np.ndarray, position: str = "inferior") -> float:
    """
    Measures the maximum concavity depth of the superior or inferior endplate.
    Fits a straight line between the two endpoints of the endplate row
    and measures the maximum perpendicular deviation inward.
    """
    rows, cols = np.where(region_mask > 0)
    if len(rows) == 0:
        return 0.0

    target_row = rows.min() if position == "superior" else rows.max()
    endplate_cols = cols[rows == target_row]
    if len(endplate_cols) < 3:
        return 0.0

    c_min, c_max = endplate_cols.min(), endplate_cols.max()
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
        line_mid_col   = (c_min + c_max) / 2
        actual_mid_col = (c_left + c_right) / 2
        depth = abs(line_mid_col - actual_mid_col)
        if depth > max_depth:
            max_depth = depth

    return float(max_depth)


def _ant_post_heights(region_mask: np.ndarray):
    """
    Returns:
      anterior_height  – column extent at leftmost 10 % of width
      posterior_height – column extent at rightmost 10 % of width
      ant_post_ratio   – anterior / posterior height
    """
    rows, cols = np.where(region_mask > 0)
    if len(cols) == 0:
        return 0.0, 0.0, 0.0

    c_min, c_max = cols.min(), cols.max()
    width = c_max - c_min
    band  = max(1, int(width * 0.10))

    ant_mask  = cols <= (c_min + band)
    ant_rows  = rows[ant_mask]
    ant_h     = float(ant_rows.max() - ant_rows.min() + 1) if len(ant_rows) > 1 else 0.0

    post_mask = cols >= (c_max - band)
    post_rows = rows[post_mask]
    post_h    = float(post_rows.max() - post_rows.min() + 1) if len(post_rows) > 1 else 0.0

    ratio = ant_h / post_h if post_h > 0 else 0.0
    return ant_h, post_h, ratio


def _endplate_slope_angle(region_mask: np.ndarray, position: str = "inferior") -> float:
    """
    Fits a regression line through the inferior (or superior) endplate pixels
    and returns the angle in degrees from horizontal.
    """
    rows, cols = np.where(region_mask > 0)
    if len(rows) == 0:
        return 0.0

    target_row = rows.max() if position == "inferior" else rows.min()
    span       = max(1, (rows.max() - rows.min()) // 8)

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
        angle = 0.0
    return float(angle)


def _convex_hull_features(region_mask: np.ndarray):
    """
    Returns:
      hull_area  – area of the convex hull   (scipy ConvexHull.volume in 2-D)
      fill_ratio – actual area / hull area   (1 = perfectly convex)
      hull_perim – perimeter of convex hull  (scipy ConvexHull.area in 2-D)
    """
    rows, cols = np.where(region_mask > 0)
    if len(rows) < 4:
        return 0.0, 0.0, 0.0
    points = np.column_stack([cols, rows]).astype(float)
    try:
        hull        = ConvexHull(points)
        hull_area   = hull.volume        # in 2-D: volume = area
        hull_perim  = hull.area          # in 2-D: area   = perimeter
        actual_area = float(region_mask.sum())
        fill_ratio  = actual_area / hull_area if hull_area > 0 else 0.0
        return float(hull_area), float(fill_ratio), float(hull_perim)
    except Exception:
        return 0.0, 0.0, 0.0


# =====================================================================
#  IMAGE UTILITIES
# =====================================================================

def _img_to_base64(np_img: np.ndarray) -> str:
    pil = Image.fromarray(np_img)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _create_overlay(orig_gray: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    rgb     = cv2.cvtColor(orig_gray, cv2.COLOR_GRAY2RGB)
    overlay = rgb.copy()
    overlay[binary_mask > 0] = [56, 189, 248]
    return cv2.addWeighted(rgb, 0.55, overlay, 0.45, 0)


# =====================================================================
#  FULL FEATURE EXTRACTION
#  Feature names MUST stay in sync with feature_extraction.py
# =====================================================================

def _extract_all_features(clean_mask: np.ndarray,
                           regions: list,
                           vertebrae_labels: list) -> tuple[dict, list]:
    """
    Extract the same feature set as feature_extraction.py.
    Returns (features_dict, vert_info_list).
    """
    features  = {}
    vert_info = []
    row_data  = {}   # lightweight per-vertebra data for inter-vertebra features

    total_mask_px = float(clean_mask.sum())

    for v, r in zip(vertebrae_labels, regions):
        minr, minc, maxr, maxc = r.bbox
        h    = maxr - minr
        w    = maxc - minc
        area = float(r.area)
        peri = float(r.perimeter) if r.perimeter > 0 else 1e-6

        ar           = w / h if h > 0 else 0.0
        circularity  = 4 * np.pi * area / (peri ** 2)
        eccentricity = float(r.eccentricity)
        solidity     = float(r.solidity)
        extent       = float(r.extent)
        orientation  = float(r.orientation)
        angle_deg    = float(np.degrees(orientation))
        minor_major  = (r.minor_axis_length / r.major_axis_length
                        if r.major_axis_length > 0 else 0.0)

        # Crop binary mask to vertebra bounding box (same as training)
        crop = clean_mask[minr:maxr, minc:maxc]

        # ── Convex hull features (FIX: use scipy, not convex_area proxy) ──
        hull_area, hull_fill, hull_perim = _convex_hull_features(crop)

        # ── Anterior / posterior heights ───────────────────────────────────
        ant_h, post_h, ap_ratio = _ant_post_heights(crop)

        # ── Endplate concavity ─────────────────────────────────────────────
        sup_concavity = _endplate_concavity(crop, "superior")
        inf_concavity = _endplate_concavity(crop, "inferior")

        # ── Endplate slope angles (FIX: linear regression, not ±orientation)
        inf_slope = _endplate_slope_angle(crop, "inferior")
        sup_slope = _endplate_slope_angle(crop, "superior")

        # ── Normalised area (FIX: divide by total mask pixels, not C2 bbox) -
        norm_area = area / total_mask_px if total_mask_px > 0 else 0.0

        # ── Store features (names must match feature_extraction.py exactly) ─
        features[f"{v} Area"]                   = area
        features[f"{v} Perimeter"]              = peri
        features[f"{v} Height"]                 = float(h)
        features[f"{v} Width"]                  = float(w)
        features[f"{v} Aspect Ratio"]           = float(ar)
        features[f"{v} Circularity"]            = float(circularity)
        features[f"{v} Eccentricity"]           = eccentricity
        features[f"{v} Solidity"]               = solidity
        features[f"{v} Extent"]                 = extent
        features[f"{v} Orientation"]            = angle_deg
        features[f"{v} Minor/Major Ratio"]      = float(minor_major)
        # Convex hull
        features[f"{v} Convex Hull Area"]       = hull_area
        features[f"{v} Hull Fill Ratio"]        = hull_fill
        features[f"{v} Hull Perimeter"]         = hull_perim   # FIX: was peri
        # Anterior / posterior
        features[f"{v} Anterior Height"]        = ant_h
        features[f"{v} Posterior Height"]       = post_h
        features[f"{v} Ant/Post Height Ratio"]  = ap_ratio
        # Endplate concavity  (FIX: was a single wrong 'Concavity' = 1-solidity)
        features[f"{v} Superior Concavity"]     = sup_concavity
        features[f"{v} Inferior Concavity"]     = inf_concavity
        # Endplate slope  (FIX: was ±orientation_degrees)
        features[f"{v} Inferior Slope Angle"]   = inf_slope
        features[f"{v} Superior Slope Angle"]   = sup_slope
        # Normalised area  (FIX: formula and C2 now included)
        features[f"{v} Norm Area"]              = norm_area

        row_data[v] = {
            "h":     h,
            "w":     w,
            "ar":    ar,
            "circ":  circularity,
            "solid": solidity,
            "area":  area,
            "bbox":  (minr, minc, maxr, maxc),
        }

        vert_info.append({
            "name":         v,
            "area":         round(area, 1),
            "aspect_ratio": round(ar, 3),
            "circularity":  round(circularity, 3),
            "solidity":     round(solidity, 3),
        })

    # ── Inter-vertebral ratio features (match feature_extraction.py names) ──

    # Area ratios  (FIX: these were missing from v7)
    features["C3/C2 Area Ratio"] = (row_data["C3"]["area"] / row_data["C2"]["area"]
                                     if row_data["C2"]["area"] > 0 else 0.0)
    features["C4/C2 Area Ratio"] = (row_data["C4"]["area"] / row_data["C2"]["area"]
                                     if row_data["C2"]["area"] > 0 else 0.0)
    features["C4/C3 Area Ratio"] = (row_data["C4"]["area"] / row_data["C3"]["area"]
                                     if row_data["C3"]["area"] > 0 else 0.0)

    # Height ratios
    features["C3/C2 Height Ratio"] = (row_data["C3"]["h"] / row_data["C2"]["h"]
                                       if row_data["C2"]["h"] > 0 else 0.0)
    features["C4/C3 Height Ratio"] = (row_data["C4"]["h"] / row_data["C3"]["h"]
                                       if row_data["C3"]["h"] > 0 else 0.0)
    features["C4/C2 Height Ratio"] = (row_data["C4"]["h"] / row_data["C2"]["h"]
                                       if row_data["C2"]["h"] > 0 else 0.0)

    # Solidity ratios
    features["C4/C2 Solidity Ratio"] = (row_data["C4"]["solid"] / row_data["C2"]["solid"]
                                         if row_data["C2"]["solid"] > 0 else 0.0)
    features["C4/C3 Solidity Ratio"] = (row_data["C4"]["solid"] / row_data["C3"]["solid"]
                                         if row_data["C3"]["solid"] > 0 else 0.0)

    # Disc gap heights: top-of-next minus bottom-of-current bounding box
    # (FIX: v7 used centroid distances and wrong names)
    gap_c2c3 = row_data["C3"]["bbox"][0] - row_data["C2"]["bbox"][2]   # top_C3 − bot_C2
    gap_c3c4 = row_data["C4"]["bbox"][0] - row_data["C3"]["bbox"][2]   # top_C4 − bot_C3
    features["C2-C3 Gap (px)"] = float(max(0, gap_c2c3))
    features["C3-C4 Gap (px)"] = float(max(0, gap_c3c4))
    features["Gap Ratio C3/C2"] = (
        features["C3-C4 Gap (px)"] / features["C2-C3 Gap (px)"]
        if features["C2-C3 Gap (px)"] > 0 else 0.0
    )

    # Solidity trend  (FIX: was missing from v7)
    features["Solidity Trend (C2\u2192C4)"] = (row_data["C4"]["solid"] - row_data["C2"]["solid"]) / 2

    # Circularity trend  (FIX: was ASCII '->' — must be Unicode '→')
    features["Circularity Trend (C2\u2192C4)"] = (row_data["C4"]["circ"] - row_data["C2"]["circ"]) / 2

    return features, vert_info


# =====================================================================
#  MAIN PIPELINE
# =====================================================================
_audit_done = False   # run audit only once per process lifetime


def predict_age(image_bytes: bytes) -> dict:
    global _audit_done

    # 1. Load & resize
    pil_img    = Image.open(io.BytesIO(image_bytes)).convert("L")
    img_256    = pil_img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    img_np_256 = np.array(img_256, dtype=np.float32) / 255.0

    # 2. U-Net segmentation
    tensor = torch.from_numpy(img_np_256).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits   = unet(tensor)
        prob_map = torch.sigmoid(logits)[0, 0].cpu().numpy()

    mask_256 = (prob_map > THRESHOLD).astype(np.uint8)
    print(f"Mask non-zero pixels (256×256): {mask_256.sum()}")

    # 3. Scale to training resolution
    #    547×693 matches the original LAST Dataset image dimensions.
    #    If your training images were a different fixed size, update this.
    mask_orig = cv2.resize(mask_256, (547, 693), interpolation=cv2.INTER_NEAREST)

    # 4. Morphological cleaning
    k3         = np.ones((3, 3), np.uint8)
    clean_mask = cv2.morphologyEx(mask_orig,  cv2.MORPH_CLOSE, k3, iterations=2)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN,  k3, iterations=1)
    print(f"Mask non-zero pixels (orig res): {clean_mask.sum()}")

    # 5. Detect vertebrae
    labeled = label(clean_mask)
    regions = [r for r in regionprops(labeled) if r.area > 100]

    if len(regions) < 3:
        raise ValueError(
            "Poor segmentation: fewer than 3 vertebrae detected. "
            "Please upload a clearer lateral cephalometric X-ray."
        )

    regions = sorted(regions, key=lambda r: r.centroid[0])[:3]

    # 6. Feature extraction
    all_feats, vert_info = _extract_all_features(clean_mask, regions, ["C2", "C3", "C4"])
    print(f"Extracted {len(all_feats)} features")

    # 7. One-time feature audit (raises KeyError with clear message if broken)
    if not _audit_done:
        _audit_features(all_feats)
        _audit_done = True

    # 8. Age prediction
    age_row  = pd.DataFrame([{f: all_feats[f] for f in age_feats}])
    pred_age = float(age_model.predict(age_row)[0])
    print(f"Predicted Age: {pred_age:.2f} yrs")

    # 9. Gender prediction
    gen_row       = pd.DataFrame([{f: all_feats[f] for f in gender_feats}])
    gen_pred_int  = int(gender_model.predict(gen_row)[0])
    gen_proba_arr = gender_model.predict_proba(gen_row)[0]
    gen_label     = "Female" if gen_pred_int == 1 else "Male"
    gen_conf      = float(gen_proba_arr[gen_pred_int])
    print(f"Predicted Gender: {gen_label}  (conf={gen_conf:.3f})")

    # 10. Display images
    orig_display = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
    mask_scaled  = cv2.resize(clean_mask, (IMG_SIZE, IMG_SIZE),
                               interpolation=cv2.INTER_NEAREST)
    mask_display = (mask_scaled * 255).astype(np.uint8)
    overlay_img  = _create_overlay(orig_display, mask_scaled)

    return {
        "predicted_age"    : round(pred_age, 2),
        "predicted_gender" : gen_label,
        "gender_confidence": round(gen_conf, 3),
        "vertebrae"        : vert_info,
        "images": {
            "original": _img_to_base64(orig_display),
            "mask"    : _img_to_base64(mask_display),
            "overlay" : _img_to_base64(overlay_img),
        },
    }


# =====================================================================
#  STANDALONE TEST
# =====================================================================
if __name__ == "__main__":
    IMAGE_PATH = r"LAST Dataset\RAWIMG\0001035.png"

    with open(IMAGE_PATH, "rb") as f:
        image_bytes = f.read()

    result = predict_age(image_bytes)

    print("\n" + "=" * 45)
    print("CVM PREDICTION RESULTS")
    print("=" * 45)
    print(f"  Predicted Age    : {result['predicted_age']} years")
    print(f"  Predicted Gender : {result['predicted_gender']}")
    print(f"  Gender Confidence: {result['gender_confidence'] * 100:.1f}%")
    print("=" * 45)

    import matplotlib.pyplot as plt
    import base64 as b64

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, title, key in zip(
        axes,
        ["Original", "Mask", "Overlay"],
        ["original", "mask", "overlay"],
    ):
        raw = result["images"][key].split(",")[1]
        img = Image.open(io.BytesIO(b64.b64decode(raw)))
        ax.imshow(img, cmap="gray" if title != "Overlay" else None)
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(
        f"Age: {result['predicted_age']} yrs  |  "
        f"Gender: {result['predicted_gender']}  ({result['gender_confidence'] * 100:.1f}%)",
        fontsize=13,
    )
    plt.tight_layout()
    plt.show()