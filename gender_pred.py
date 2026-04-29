"""
gender_pred.py  –  Stacking Ensemble Gender Prediction (Inference + Evaluation)

Gender classification turned out cleaner than age regression —
84% accuracy on hold-out, ROC-AUC 0.912.

Two bugs fixed in this version:
  FIX-1: was evaluating on full dataset including training rows.
          Accuracy looked great. Was completely dishonest.
          Now reconstructs the same 80/20 split and reports hold-out only.

  FIX-2: model loading and evaluation were at module level —
          importing predict_gender_from_features() from Flask was
          triggering the entire evaluation pipeline as a side effect.
          Moved everything except the inference function under
          if __name__ == "__main__".
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")   # no display — saving PNGs only
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)
from sklearn.model_selection import train_test_split


# =====================================================================
#  CONFIG
# =====================================================================

DATA_PATH         = "final_enhanced.xlsx"
GENDER_MODEL_PATH = "best_gender_model.pkl"
SELECTOR_PATH     = "gender_feature_selector.pkl"
FEAT_COLS_PATH    = "gender_selected_features.pkl"
OUTPUT_EXCEL      = "predicted_genders_final.xlsx"

EXCLUDE_COLS = {"Image", "Age", "Gender"}
LABELS       = ["Male", "Female"]   # index matches 0/1 encoding
RANDOM_STATE = 42   # must match model_comparison.py — same seed = same split


# =====================================================================
#  LOAD ARTIFACTS
#  Outside the __main__ guard so the inference function below
#  has access to them when imported by Flask or other scripts.
# =====================================================================

print("Loading gender model artifacts ...")
gender_model    = joblib.load(GENDER_MODEL_PATH)
gender_selector = joblib.load(SELECTOR_PATH)
selected_feats  = joblib.load(FEAT_COLS_PATH)
# loading saved feature list — not reselecting here
# rerunning SelectFromModel on a different data sample
# would pick different features and silently break predictions

print(f"  Model         : {type(gender_model).__name__}")
print(f"  Features used : {len(selected_feats)}")
print(f"  Feature list  : {selected_feats}")


# =====================================================================
#  SINGLE-SAMPLE INFERENCE  — called by Flask API
#
#  This is the only function Flask needs.
#  Keeping it outside __main__ so it's importable without
#  triggering the evaluation pipeline.
# =====================================================================

def predict_gender_from_features(feature_dict: dict) -> dict:
    """
    Predict gender from extracted morphological features.

    Parameters
    ----------
    feature_dict : dict
        Feature names → values. Must contain all keys in selected_feats.
        'Age', 'Gender', 'Image' keys are ignored if present.

    Returns
    -------
    dict  {label: "Male"|"Female", confidence: float}

    Note: confidence is the model's probability for the predicted class,
    not just a binary 0/1. Low confidence predictions (<0.6) are worth
    flagging for manual review in a clinical setting.
    """
    row        = pd.DataFrame([feature_dict])[selected_feats]
    pred_int   = int(gender_model.predict(row)[0])
    proba_arr  = gender_model.predict_proba(row)[0]   # [P(Male), P(Female)]

    # index into proba with pred_int gives confidence for predicted class
    confidence = float(proba_arr[pred_int])
    label      = "Female" if pred_int == 1 else "Male"

    return {"label": label, "confidence": round(confidence, 4)}


# =====================================================================
#  EVALUATION + PLOTS  (only when run directly, not on import)
# =====================================================================

if __name__ == "__main__":

    # ── load data ────────────────────────────────────────────────────
    df = pd.read_excel(DATA_PATH).dropna().reset_index(drop=True)
    # dropna removes rows where mask extraction produced NaN features
    # ~4% of dataset — logged separately during feature extraction

    all_feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X_all = df[all_feature_cols]
    y_all = df["Gender"]   # 0 = Male, 1 = Female

    # ── FIX-1: honest hold-out split ─────────────────────────────────
    # stratify=y_all works here because gender is categorical (0/1)
    # unlike age which needed bin stratification
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_all,   # keep male/female ratio consistent across splits
    )

    X_te_sel   = X_te[selected_feats]
    y_te_pred  = gender_model.predict(X_te_sel)
    y_te_proba = gender_model.predict_proba(X_te_sel)[:, 1]  # P(Female)

    acc  = accuracy_score(y_te, y_te_pred)
    f1   = f1_score(y_te, y_te_pred, average="weighted")
    prec = precision_score(y_te, y_te_pred, average="weighted")
    rec  = recall_score(y_te, y_te_pred, average="weighted")
    auc  = roc_auc_score(y_te, y_te_proba)
    cm   = confusion_matrix(y_te, y_te_pred)

    print("\n" + "=" * 55)
    print(f"GENDER PREDICTION  –  HOLD-OUT TEST SET  (n={len(y_te)})")
    print("=" * 55)
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  F1 Score  : {f1:.4f}  (weighted)")
    print(f"  Precision : {prec:.4f}  (weighted)")
    print(f"  Recall    : {rec:.4f}  (weighted)")
    print(f"  ROC-AUC   : {auc:.4f}")
    # AUC matters more than accuracy here —
    # if dataset were 80% male, predicting male always = 80% accuracy
    # AUC 0.912 means the model actually learned something
    print("=" * 55)

    print(f"\nConfusion Matrix  (rows=Actual, cols=Predicted):")
    print(f"              Pred Male  Pred Female")
    print(f"  Actual Male    {cm[0,0]:>4}       {cm[0,1]:>4}")
    print(f"  Actual Female  {cm[1,0]:>4}       {cm[1,1]:>4}")
    # off-diagonal = errors. want these small and roughly symmetric.
    # asymmetry would mean the model is biased toward one gender.

    print(f"\nClassification Report (hold-out):")
    print(classification_report(y_te, y_te_pred, target_names=LABELS))

    # ── full-dataset predictions for Excel export ─────────────────────
    # still useful to have every sample predicted for downstream use,
    # but Split column makes clear which rows are honest test predictions
    X_all_sel   = X_all[selected_feats]
    y_pred_all  = gender_model.predict(X_all_sel)
    y_proba_all = gender_model.predict_proba(X_all_sel)[:, 1]

    acc_full = accuracy_score(y_all, y_pred_all)
    print(f"\n  Full-dataset accuracy (includes training rows): {acc_full:.4f}")
    print(f"  Training rows inflate this — not a fair estimate.")

    split_label             = pd.Series("Train", index=X_all.index)
    split_label[X_te.index] = "Test"

    results = pd.DataFrame({
        "Image"            : df["Image"],
        "Split"            : split_label.values,
        "Actual Gender"    : y_all.map({0: "Male", 1: "Female"}),
        "Predicted Gender" : pd.Series(y_pred_all).map({0: "Male", 1: "Female"}),
        "P(Female)"        : np.round(y_proba_all, 4),
        "P(Male)"          : np.round(1 - y_proba_all, 4),
        "Correct"          : (y_all.values == y_pred_all),
        # Correct column lets you filter just the errors quickly in Excel
    })
    results.to_excel(OUTPUT_EXCEL, index=False)
    print(f"\nPredictions saved → {OUTPUT_EXCEL}")

    # ── plots — all three use hold-out only ───────────────────────────

    # 1. Confusion matrix heatmap
    # white text on dark cells, black on light — auto-contrasted
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(LABELS)
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(
        f"Gender Confusion Matrix — Hold-out  (Acc={acc*100:.1f}%)"
    )
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=14, fontweight="bold"
            )
    plt.tight_layout()
    plt.savefig("gender_confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved → gender_confusion_matrix.png")

    # 2. ROC curve
    # curve hugging top-left corner = good discrimination
    # curve along diagonal = random guessing
    fpr, tpr, _ = roc_curve(y_te, y_te_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#4C72B0", linewidth=2,
            label=f"Hold-out ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "r--", linewidth=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Gender Prediction – ROC Curve  (Hold-out, n={len(y_te)})")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("gender_roc_curve.png", dpi=150)
    plt.close()
    print("Saved → gender_roc_curve.png")

    # 3. Confidence distribution — bonus plot
    # correct predictions should cluster near 1.0
    # wrong predictions clustering near 0.5 = model was uncertain, understandable
    # wrong predictions clustering near 1.0 = model was confidently wrong, bad
    correct_mask = y_te.values == y_te_pred
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(
        y_te_proba[correct_mask], bins=15, alpha=0.7,
        color="#4C72B0", label="Correct predictions", edgecolor="white"
    )
    ax.hist(
        y_te_proba[~correct_mask], bins=15, alpha=0.7,
        color="#DD8452", label="Wrong predictions", edgecolor="white"
    )
    ax.axvline(
        0.5, color="red", linestyle="--",
        linewidth=1.2, label="Decision boundary"
    )
    ax.set_xlabel("P(Female)")
    ax.set_ylabel("Count")
    ax.set_title("Gender Prediction – Confidence Distribution (Hold-out)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("gender_confidence_distribution.png", dpi=150)
    plt.close()
    print("Saved → gender_confidence_distribution.png")

    # ── single-sample inference test ──────────────────────────────────
    image_name = "0194127.png"   # swap in any filename to sanity-check

    row = df[df["Image"] == image_name]
    if row.empty:
        print(f"Image '{image_name}' not found in dataset.")
    else:
        sample = row[all_feature_cols].iloc[0].to_dict()
        result = predict_gender_from_features(sample)
        actual = LABELS[int(row["Gender"].iloc[0])]
        print(
            f"  Actual : {actual}  |  "
            f"Predicted : {result['label']}  |  "
            f"Confidence : {result['confidence']*100:.1f}%"
        )