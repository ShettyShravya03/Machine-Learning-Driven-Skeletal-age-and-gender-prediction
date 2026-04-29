"""
gender_pred.py  –  Stacking Ensemble Gender Prediction (Inference + Evaluation)

Loads:  best_gender_model.pkl
        gender_feature_selector.pkl   (SelectFromModel – RF-based)
        gender_selected_features.pkl  (ordered list of selected feature names)
        final_enhanced.xlsx           (for evaluation only)

Outputs:
        predicted_genders_final.xlsx
        gender_confusion_matrix.png
        gender_roc_curve.png

FIX-1  Evaluation was run on the FULL dataset (training + test rows combined),
        so reported metrics were optimistically inflated by in-sample predictions.
        Fix: re-create the same 80/20 stratified split used in model_comparison.py
        (random_state=42) and report metrics on the HOLD-OUT test set only.
        Full-dataset predictions are still saved to Excel for completeness, but
        each row is tagged Train/Test so the distinction is clear.

FIX-2  Module-level code (model loading, data loading, evaluation, plotting)
        now runs only when the script is executed directly, not on import.
        This makes predict_gender_from_features() safe to import from Flask
        or unknown_pred.py without triggering side effects.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score, confusion_matrix,
                              classification_report, roc_curve)
from sklearn.model_selection import train_test_split

# =====================================================================
#  CONFIG
# =====================================================================
DATA_PATH           = "final_enhanced.xlsx"
GENDER_MODEL_PATH   = "best_gender_model.pkl"
SELECTOR_PATH       = "gender_feature_selector.pkl"
FEAT_COLS_PATH      = "gender_selected_features.pkl"
OUTPUT_EXCEL        = "predicted_genders_final.xlsx"

EXCLUDE_COLS        = {"Image", "Age", "Gender"}
LABELS              = ["Male", "Female"]
RANDOM_STATE        = 42   # must match model_comparison.py

# =====================================================================
#  LOAD ARTIFACTS  (always – needed for the inference function)
# =====================================================================
print("Loading gender model artifacts ...")
gender_model    = joblib.load(GENDER_MODEL_PATH)
gender_selector = joblib.load(SELECTOR_PATH)
selected_feats  = joblib.load(FEAT_COLS_PATH)

print(f"  Model          : {type(gender_model).__name__}")
print(f"  Features used  : {len(selected_feats)}")
print(f"  Feature list   : {selected_feats}")

# =====================================================================
#  STANDALONE SINGLE-SAMPLE INFERENCE FUNCTION
#  Safe to import: guarded by __name__ check so evaluation/plotting
#  only runs when executed directly, not on import.
# =====================================================================
def predict_gender_from_features(feature_dict: dict) -> dict:
    """
    Predict gender from a dict of morphological features.

    Parameters
    ----------
    feature_dict : dict
        Keys = feature names (morphological columns only).
        Do NOT include 'Age', 'Gender', or 'Image'.

    Returns
    -------
    dict
        {
          "label"      : "Male" | "Female",
          "confidence" : float   (0–1, probability of predicted class)
        }
    """
    row        = pd.DataFrame([feature_dict])[selected_feats]
    pred_int   = int(gender_model.predict(row)[0])
    proba_arr  = gender_model.predict_proba(row)[0]   # [P(Male), P(Female)]
    confidence = float(proba_arr[pred_int])
    label      = "Female" if pred_int == 1 else "Male"
    return {"label": label, "confidence": round(confidence, 4)}


# =====================================================================
#  EVALUATION + OUTPUT  (runs only when executed directly)
# =====================================================================
if __name__ == "__main__":

    # ── Load data ────────────────────────────────────────────────────
    df = pd.read_excel(DATA_PATH).dropna().reset_index(drop=True)

    all_feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X_all  = df[all_feature_cols]
    y_all  = df["Gender"]    # 0 = Male, 1 = Female

    # ── FIX-1: Honest hold-out evaluation ────────────────────────────
    # Re-create the exact 80/20 stratified split from model_comparison.py
    # so we report on samples the model has never seen.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_all,
    )

    X_te_sel = X_te[selected_feats]
    y_te_pred  = gender_model.predict(X_te_sel)
    y_te_proba = gender_model.predict_proba(X_te_sel)[:, 1]   # P(Female)

    acc  = accuracy_score(y_te, y_te_pred)
    f1   = f1_score(y_te, y_te_pred, average="weighted")
    prec = precision_score(y_te, y_te_pred, average="weighted")
    rec  = recall_score(y_te, y_te_pred, average="weighted")
    auc  = roc_auc_score(y_te, y_te_proba)
    cm   = confusion_matrix(y_te, y_te_pred)

    print("\n" + "=" * 55)
    print(f"GENDER PREDICTION  –  HOLD-OUT TEST SET  (n={len(y_te)})")
    print("=" * 55)
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)  ← primary metric")
    print(f"  F1 Score  : {f1:.4f}  (weighted)")
    print(f"  Precision : {prec:.4f}  (weighted)")
    print(f"  Recall    : {rec:.4f}  (weighted)")
    print(f"  ROC-AUC   : {auc:.4f}")
    print("=" * 55)

    print(f"\nConfusion Matrix  (rows=Actual, cols=Predicted):")
    print(f"              Pred Male  Pred Female")
    print(f"  Actual Male    {cm[0,0]:>4}       {cm[0,1]:>4}")
    print(f"  Actual Female  {cm[1,0]:>4}       {cm[1,1]:>4}")

    print(f"\nDetailed Classification Report (hold-out):")
    print(classification_report(y_te, y_te_pred, target_names=LABELS))

    # ── Full-dataset predictions (for Excel export) ───────────────────
    # Training rows will show artificially high accuracy — see Split column.
    X_all_sel  = X_all[selected_feats]
    y_pred_all = gender_model.predict(X_all_sel)
    y_proba_all = gender_model.predict_proba(X_all_sel)[:, 1]

    acc_full = accuracy_score(y_all, y_pred_all)
    print(f"\n  Full-dataset accuracy (includes training rows): {acc_full:.4f}")
    print(f"  ⚠  NOT a fair estimate — training rows inflate this number.")

    # Tag each row so readers know which errors are honest
    split_label = pd.Series("Train", index=X_all.index)
    split_label[X_te.index] = "Test"

    results = pd.DataFrame({
        "Image"            : df["Image"],
        "Split"            : split_label.values,         # NEW: Train/Test label
        "Actual Gender"    : y_all.map({0: "Male", 1: "Female"}),
        "Predicted Gender" : pd.Series(y_pred_all).map({0: "Male", 1: "Female"}),
        "P(Female)"        : np.round(y_proba_all, 4),
        "P(Male)"          : np.round(1 - y_proba_all, 4),
        "Correct"          : (y_all.values == y_pred_all),
    })
    results.to_excel(OUTPUT_EXCEL, index=False)
    print(f"\nPredictions saved → {OUTPUT_EXCEL}  (Split column marks Train/Test rows)")

    # ── Plots (hold-out subset only so visuals reflect honest perf) ───

    # 1. Confusion matrix heatmap
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(LABELS); ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Gender Confusion Matrix — Hold-out  (Acc={acc*100:.1f}%)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("gender_confusion_matrix.png", dpi=150)
    plt.close()
    print("Plot saved → gender_confusion_matrix.png")

    # 2. ROC curve (hold-out)
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
    print("Plot saved → gender_roc_curve.png")

    # 3. Prediction confidence distribution (hold-out) — bonus plot
    correct_mask = y_te.values == y_te_pred
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(y_te_proba[correct_mask],   bins=15, alpha=0.7,
            color="#4C72B0", label="Correct predictions", edgecolor="white")
    ax.hist(y_te_proba[~correct_mask],  bins=15, alpha=0.7,
            color="#DD8452", label="Wrong predictions",   edgecolor="white")
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1.2, label="Decision boundary")
    ax.set_xlabel("P(Female)")
    ax.set_ylabel("Count")
    ax.set_title("Gender Prediction – Confidence Distribution (Hold-out)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("gender_confidence_distribution.png", dpi=150)
    plt.close()
    print("Plot saved → gender_confidence_distribution.png")
    
    # ── Standalone inference test ─────────────────────────────────────
    image_name = "0194127.png"  # change this to your image filename

    row = df[df["Image"] == image_name]
    if row.empty:
        print(f"Image '{image_name}' not found in dataset.")
    else:
        sample = row[all_feature_cols].iloc[0].to_dict()
        result = predict_gender_from_features(sample)
        actual = LABELS[int(row["Gender"].iloc[0])]
        print(f"  Actual : {actual}  |  Predicted : {result['label']}"
          f"  |  Confidence : {result['confidence']*100:.1f}%")
        