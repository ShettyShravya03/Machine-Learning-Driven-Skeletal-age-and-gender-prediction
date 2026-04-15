"""
gender_pred.py  –  Stacking Ensemble Gender Prediction (Inference + Evaluation)

Loads:  best_gender_model.pkl
        gender_feature_selector.pkl   (SelectFromModel – RF-based)
        gender_selected_features.pkl  (ordered list of 15 selected feature names)
        final_enhanced.xlsx           (for evaluation; remove if running on new data)

Outputs:
        predicted_genders_final.xlsx
        gender_confusion_matrix.png
        gender_roc_curve.png
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

# =====================================================================
#  LOAD ARTIFACTS
# =====================================================================
print("Loading gender model artifacts ...")
gender_model    = joblib.load(GENDER_MODEL_PATH)
gender_selector = joblib.load(SELECTOR_PATH)
selected_feats  = joblib.load(FEAT_COLS_PATH)

print(f"  Model          : {type(gender_model).__name__}")
print(f"  Features used  : {len(selected_feats)}")
print(f"  Feature list   : {selected_feats}")

# =====================================================================
#  LOAD DATA
# =====================================================================
df = pd.read_excel(DATA_PATH).dropna().reset_index(drop=True)

all_feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
X_all   = df[all_feature_cols]
y_true  = df["Gender"]    # 0 = Male, 1 = Female

# =====================================================================
#  SELECT FEATURES  (same 15 as training)
# =====================================================================
X_sel = X_all[selected_feats]

# =====================================================================
#  PREDICT
# =====================================================================
y_pred  = gender_model.predict(X_sel)
y_proba = gender_model.predict_proba(X_sel)[:, 1]   # P(Female)

# =====================================================================
#  METRICS
# =====================================================================
acc  = accuracy_score(y_true, y_pred)
f1   = f1_score(y_true, y_pred, average="weighted")
prec = precision_score(y_true, y_pred, average="weighted")
rec  = recall_score(y_true, y_pred, average="weighted")
auc  = roc_auc_score(y_true, y_proba)
cm   = confusion_matrix(y_true, y_pred)

print("\n" + "=" * 50)
print("GENDER PREDICTION  –  EVALUATION")
print("=" * 50)
print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  F1 Score  : {f1:.4f}  (weighted)")
print(f"  Precision : {prec:.4f}  (weighted)")
print(f"  Recall    : {rec:.4f}  (weighted)")
print(f"  ROC-AUC   : {auc:.4f}")
print(f"  Samples   : {len(y_true)}")
print("=" * 50)

print(f"\nConfusion Matrix  (rows=Actual, cols=Predicted):")
print(f"              Pred Male  Pred Female")
print(f"  Actual Male    {cm[0,0]:>4}       {cm[0,1]:>4}")
print(f"  Actual Female  {cm[1,0]:>4}       {cm[1,1]:>4}")

print(f"\nDetailed Classification Report:")
print(classification_report(y_true, y_pred, target_names=LABELS))

# =====================================================================
#  SAVE PREDICTIONS
# =====================================================================
results = pd.DataFrame({
    "Image"             : df["Image"],
    "Actual Gender"     : y_true.map({0: "Male", 1: "Female"}),
    "Predicted Gender"  : pd.Series(y_pred).map({0: "Male", 1: "Female"}),
    "P(Female)"         : np.round(y_proba, 4),
    "P(Male)"           : np.round(1 - y_proba, 4),
    "Correct"           : (y_true.values == y_pred),
})
results.to_excel(OUTPUT_EXCEL, index=False)
print(f"\nPredictions saved -> {OUTPUT_EXCEL}")

# =====================================================================
#  PLOTS
# =====================================================================

# 1. Confusion matrix heatmap
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
plt.colorbar(im, ax=ax)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(LABELS); ax.set_yticklabels(LABELS)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"Gender Confusion Matrix  (Acc={acc*100:.1f}%)")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("gender_confusion_matrix.png", dpi=150)
plt.close()
print("Plot saved -> gender_confusion_matrix.png")

# 2. ROC curve
fpr, tpr, _ = roc_curve(y_true, y_proba)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color="#4C72B0", linewidth=2, label=f"ROC (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], "r--", linewidth=1, label="Random classifier")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Gender Prediction – ROC Curve")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("gender_roc_curve.png", dpi=150)
plt.close()
print("Plot saved -> gender_roc_curve.png")

# =====================================================================
#  STANDALONE SINGLE-SAMPLE INFERENCE FUNCTION
#  (imported by unknown_pred.py)
# =====================================================================
def predict_gender_from_features(feature_dict: dict) -> dict:
    """
    Predict gender from a dict of morphological features.

    Parameters
    ----------
    feature_dict : dict
        Keys = feature names (all 79 morphological columns).
        Do NOT include 'Age', 'Gender', or 'Image'.

    Returns
    -------
    dict
        {
          "label"      : "Male" | "Female",
          "confidence" : float   (0–1, probability of predicted class)
        }
    """
    row         = pd.DataFrame([feature_dict])[selected_feats]
    pred_int    = int(gender_model.predict(row)[0])
    proba_arr   = gender_model.predict_proba(row)[0]   # [P(Male), P(Female)]
    confidence  = float(proba_arr[pred_int])
    label       = "Female" if pred_int == 1 else "Male"
    return {"label": label, "confidence": round(confidence, 4)}


if __name__ == "__main__":
    print("\nStandalone inference test (first sample):")
    sample = df[all_feature_cols].iloc[0].to_dict()
    result = predict_gender_from_features(sample)
    actual = LABELS[int(y_true.iloc[0])]
    print(f"  Actual : {actual}  |  Predicted : {result['label']}"
          f"  |  Confidence : {result['confidence']*100:.1f}%")