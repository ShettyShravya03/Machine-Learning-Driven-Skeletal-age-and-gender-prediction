"""
age_pred.py  –  XGBoost Age Prediction (Inference + Evaluation)

Loads:  best_age_model.pkl
        age_feature_selector.pkl   (SelectFromModel – RF-based)
        age_selected_features.pkl  (ordered list of 23 selected feature names)
        final_enhanced.xlsx        (for evaluation; remove if running on new data)

Outputs:
        predicted_ages_final.xlsx
        age_residual_plot.png
        age_actual_vs_predicted.png
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =====================================================================
#  CONFIG
# =====================================================================
DATA_PATH        = "final_enhanced.xlsx"
AGE_MODEL_PATH   = "best_age_model.pkl"
SELECTOR_PATH    = "age_feature_selector.pkl"
FEAT_COLS_PATH   = "age_selected_features.pkl"
OUTPUT_EXCEL     = "predicted_ages_final.xlsx"

EXCLUDE_COLS     = {"Image", "Age", "Gender"}

# =====================================================================
#  LOAD ARTIFACTS
# =====================================================================
print("Loading age model artifacts ...")
age_model       = joblib.load(AGE_MODEL_PATH)
age_selector    = joblib.load(SELECTOR_PATH)
selected_feats  = joblib.load(FEAT_COLS_PATH)

print(f"  Model          : {type(age_model).__name__}")
print(f"  Features used  : {len(selected_feats)}")
print(f"  Feature list   : {selected_feats}")

# =====================================================================
#  LOAD DATA
# =====================================================================
df = pd.read_excel(DATA_PATH).dropna().reset_index(drop=True)

all_feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
X_all   = df[all_feature_cols]
y_true  = df["Age"]

# =====================================================================
#  SELECT FEATURES  (same 23 as training)
# =====================================================================
X_sel = X_all[selected_feats]

# =====================================================================
#  PREDICT
# =====================================================================
y_pred = age_model.predict(X_sel)

# =====================================================================
#  METRICS
# =====================================================================
mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2   = r2_score(y_true, y_pred)

print("\n" + "=" * 50)
print("AGE PREDICTION  –  EVALUATION")
print("=" * 50)
print(f"  MAE   : {mae:.3f} years")
print(f"  RMSE  : {rmse:.3f} years")
print(f"  R²    : {r2:.4f}")
print(f"  Samples: {len(y_true)}")
print("=" * 50)

# Within-N-years accuracy
for n in [2, 3, 5]:
    within = (np.abs(y_true - y_pred) <= n).mean() * 100
    print(f"  Within {n} yrs  : {within:.1f}%")

# =====================================================================
#  SAVE PREDICTIONS
# =====================================================================
results = pd.DataFrame({
    "Image"          : df["Image"],
    "Actual Age"     : y_true,
    "Predicted Age"  : np.round(y_pred, 2),
    "Absolute Error" : np.round(np.abs(y_true - y_pred), 2),
    "Signed Error"   : np.round(y_pred - y_true, 2),   # + = overestimate
})
results.to_excel(OUTPUT_EXCEL, index=False)
print(f"\nPredictions saved -> {OUTPUT_EXCEL}")

# =====================================================================
#  PLOTS
# =====================================================================

# 1. Actual vs Predicted
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_true, y_pred, alpha=0.45, s=20, color="#4C72B0", edgecolors="none")
lims = [min(y_true.min(), y_pred.min()) - 1,
        max(y_true.max(), y_pred.max()) + 1]
ax.plot(lims, lims, "r--", linewidth=1.2, label="Perfect fit")
ax.set_xlabel("Actual Age (years)")
ax.set_ylabel("Predicted Age (years)")
ax.set_title(f"Age: Actual vs Predicted  (MAE={mae:.2f} yrs, R²={r2:.3f})")
ax.legend()
plt.tight_layout()
plt.savefig("age_actual_vs_predicted.png", dpi=150)
plt.close()
print("Plot saved -> age_actual_vs_predicted.png")

# 2. Residual plot
residuals = y_pred - y_true
fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(y_pred, residuals, alpha=0.4, s=20, color="#DD8452", edgecolors="none")
ax.axhline(0, color="red", linewidth=1.2, linestyle="--")
ax.set_xlabel("Predicted Age (years)")
ax.set_ylabel("Residual (Predicted - Actual)")
ax.set_title("Age Prediction – Residual Plot")
plt.tight_layout()
plt.savefig("age_residual_plot.png", dpi=150)
plt.close()
print("Plot saved -> age_residual_plot.png")

# =====================================================================
#  STANDALONE SINGLE-SAMPLE INFERENCE FUNCTION
#  (imported by unknown_pred.py)
# =====================================================================
def predict_age_from_features(feature_dict: dict) -> float:
    """
    Predict skeletal age from a dict of morphological features.

    Parameters
    ----------
    feature_dict : dict
        Keys = feature names (all 79 morphological columns).
        Do NOT include 'Age', 'Gender', or 'Image'.

    Returns
    -------
    float  –  predicted age in years
    """
    row = pd.DataFrame([feature_dict])[selected_feats]
    return float(age_model.predict(row)[0])


if __name__ == "__main__":
    print("\nStandalone inference test (first sample):")
    sample = df[all_feature_cols].iloc[0].to_dict()
    pred   = predict_age_from_features(sample)
    actual = y_true.iloc[0]
    print(f"  Actual : {actual}  |  Predicted : {pred:.2f}  |  Error : {abs(actual-pred):.2f} yrs")