"""
unknown_pred.py  –  CVM Full Inference Pipeline  v7  (FIXED)
Architecture reverse-engineered from the TRAINING script to match the saved
checkpoint exactly.

ConvBlock layout (7 sequential entries):
  .block.0  Conv2d  (bias=False)
  .block.1  BatchNorm2d
  .block.2  ReLU
  .block.3  Identity  (dropout_p=0 for enc/dec) | Dropout2d (bottleneck)
  .block.4  Conv2d  (bias=False)   ← was BN in v6 → size mismatch
  .block.5  BatchNorm2d            ← was ReLU in v6 → unexpected key
  .block.6  ReLU

AttentionGate Conv2d:  bias=False  (was bias=True in v6 → missing bias keys)

Decoder concat:  cat([skip, x])    (was cat([x, skip]) in v6 → wrong predictions)
"""

import cv2
import torch
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from skimage.measure import label, regionprops
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
            nn.Conv2d(g_ch,   inter_ch, kernel_size=1, bias=False),  # W_g.0
            nn.BatchNorm2d(inter_ch),                                  # W_g.1
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(x_ch,   inter_ch, kernel_size=1, bias=False),  # W_x.0
            nn.BatchNorm2d(inter_ch),                                  # W_x.1
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1,      kernel_size=1, bias=False),  # psi.0
            nn.BatchNorm2d(1),                                         # psi.1
            nn.Sigmoid(),                                              # psi.2
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g may be spatially different from x → align after projection
        g_up = F.interpolate(self.W_g(g), size=x.shape[2:],
                             mode="bilinear", align_corners=False)
        att  = self.relu(g_up + self.W_x(x))
        att  = self.psi(att)   # (B, 1, H, W)
        return x * att


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1,
                 features=(64, 128, 256, 512), bottleneck_dropout=0.4):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────
        self.encoders = nn.ModuleList()
        self.pools    = nn.ModuleList()          # separate pool per level
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
        # Encoder pass
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder pass — concat order must be [skip, x] as in training
        for up, att, dec, skip in zip(
            self.upconvs, self.att_gates, self.decoders, reversed(skips)
        ):
            x    = up(x)
            skip = att(g=x, x=skip)                # attend skip features
            if x.shape != skip.shape:              # safety spatial align
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
#  HELPERS
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
#  FULL 79-FEATURE EXTRACTION
# =====================================================================
def _extract_all_features(clean_mask, regions, vertebrae_labels):
    features  = {}
    vert_info = []
    row_data  = {}

    for v, r in zip(vertebrae_labels, regions):
        minr, minc, maxr, maxc = r.bbox
        h    = maxr - minr
        w    = maxc - minc
        area = r.area
        peri = r.perimeter if r.perimeter > 0 else 1e-6

        ar           = w / h if h > 0 else 0.0
        circularity  = 4 * np.pi * area / (peri ** 2)
        eccentricity = r.eccentricity
        solidity     = r.solidity
        extent       = r.extent
        orientation  = r.orientation

        convex_area  = r.convex_area if hasattr(r, "convex_area") else area
        concavity    = 1.0 - solidity
        hull_fill    = area / convex_area if convex_area > 0 else 1.0
        minor_major  = (r.minor_axis_length / r.major_axis_length
                        if r.major_axis_length > 0 else 0.0)
        angle_deg         = float(np.degrees(orientation))
        inferior_slope    = angle_deg
        superior_slope    = -angle_deg

        features[f"{v} Area"]                 = float(area)
        features[f"{v} Perimeter"]            = float(peri)
        features[f"{v} Height"]               = float(h)
        features[f"{v} Width"]                = float(w)
        features[f"{v} Aspect Ratio"]         = float(ar)
        features[f"{v} Circularity"]          = float(circularity)
        features[f"{v} Eccentricity"]         = float(eccentricity)
        features[f"{v} Solidity"]             = float(solidity)
        features[f"{v} Extent"]               = float(extent)
        features[f"{v} Concavity"]            = float(concavity)
        features[f"{v} Orientation"]          = float(angle_deg)
        features[f"{v} Minor/Major Ratio"]    = float(minor_major)
        features[f"{v} Hull Fill Ratio"]      = float(hull_fill)
        features[f"{v} Convex Hull Area"]     = float(convex_area)
        features[f"{v} Hull Perimeter"]       = float(peri)
        features[f"{v} Inferior Slope Angle"] = float(inferior_slope)
        features[f"{v} Superior Slope Angle"] = float(superior_slope)

        row_data[v] = {
            "h": h, "w": w, "ar": ar,
            "circ": circularity, "solid": solidity,
            "cy": float(r.centroid[0])
        }

        vert_info.append({
            "name":         v,
            "area":         round(float(area), 1),
            "aspect_ratio": round(float(ar), 3),
            "circularity":  round(float(circularity), 3),
            "solidity":     round(float(solidity), 3),
        })

    for va, vb in [("C3", "C2"), ("C4", "C3"), ("C4", "C2")]:
        ra, rb = row_data[va], row_data[vb]
        key = f"{va}/{vb}"
        features[f"{key} Height Ratio"]   = ra["h"]     / rb["h"]     if rb["h"]     > 0 else 0.0
        features[f"{key} Width Ratio"]    = ra["w"]     / rb["w"]     if rb["w"]     > 0 else 0.0
        features[f"{key} AR Ratio"]       = ra["ar"]    / rb["ar"]    if rb["ar"]    > 0 else 0.0
        features[f"{key} Solidity Ratio"] = ra["solid"] / rb["solid"] if rb["solid"] > 0 else 0.0

    features["Gap C2_C3"] = abs(row_data["C3"]["cy"] - row_data["C2"]["cy"])
    features["Gap C3_C4"] = abs(row_data["C4"]["cy"] - row_data["C3"]["cy"])

    for metric, lbl in [("h", "Height"), ("w", "Width")]:
        vals = [row_data[v][metric] for v in ("C2", "C3", "C4")]
        features[f"{lbl} Trend C2->C3"] = float(vals[1] - vals[0])
        features[f"{lbl} Trend C3->C4"] = float(vals[2] - vals[1])
        features[f"{lbl} Trend C2->C4"] = float(vals[2] - vals[0])

    c2_area = row_data["C2"]["h"] * row_data["C2"]["w"]
    for v in ("C3", "C4"):
        v_area = row_data[v]["h"] * row_data[v]["w"]
        features[f"{v} Norm Area"] = v_area / c2_area if c2_area > 0 else 0.0

    features["Circularity Trend (C2->C4)"] = (
        row_data["C4"]["circ"] - row_data["C2"]["circ"]
    )

    return features, vert_info


# =====================================================================
#  MAIN PIPELINE
# =====================================================================
def predict_age(image_bytes: bytes) -> dict:
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
    all_feats, vert_info = _extract_all_features(
        clean_mask, regions, ["C2", "C3", "C4"]
    )
    print(f"Extracted {len(all_feats)} features")

    # 7. Age prediction
    age_row  = pd.DataFrame([{f: all_feats.get(f, 0.0) for f in age_feats}])
    pred_age = float(age_model.predict(age_row)[0])
    print(f"Predicted Age: {pred_age:.2f} yrs")

    # 8. Gender prediction
    gen_row       = pd.DataFrame([{f: all_feats.get(f, 0.0) for f in gender_feats}])
    gen_pred_int  = int(gender_model.predict(gen_row)[0])
    gen_proba_arr = gender_model.predict_proba(gen_row)[0]
    gen_label     = "Female" if gen_pred_int == 1 else "Male"
    gen_conf      = float(gen_proba_arr[gen_pred_int])
    print(f"Predicted Gender: {gen_label}  (conf={gen_conf:.3f})")

    # 9. Display images
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
    IMAGE_PATH = r"LAST Dataset\RAWIMG\1744053.png"

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