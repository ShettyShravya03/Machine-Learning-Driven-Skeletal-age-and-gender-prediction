"""
age_pred.py  –  XGBoost Age Prediction  (Evaluation + Inference helper)

Loads:  best_age_model.pkl
        age_feature_selector.pkl   (SelectFromModel – RF-based)
        age_selected_features.pkl  (ordered list of selected feature names)
        final_enhanced.xlsx        (for evaluation only)

Outputs:
        predicted_ages_final.xlsx
        age_residual_plot.png
        age_actual_vs_predicted.png

IMPORTANT – in-sample vs hold-out evaluation
─────────────────────────────────────────────
The model was trained on 80% of the data (train_test_split, random_state=42).
This script evaluates on the FULL dataset so that every sample gets a
prediction saved to Excel.  For honest model assessment, the key metric to
trust is the hold-out MAE printed separately below (reconstructed by
re-splitting with the same random_state).  In-sample MAE will always be lower
because tree ensembles memorise training examples to some degree.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# =====================================================================
#  CONFIG
# =====================================================================
DATA_PATH        = "final_enhanced.xlsx"
AGE_MODEL_PATH   = "best_age_model.pkl"
SELECTOR_PATH    = "age_feature_selector.pkl"
FEAT_COLS_PATH   = "age_selected_features.pkl"
OUTPUT_EXCEL     = "predicted_ages_final.xlsx"

EXCLUDE_COLS     = {"Image", "Age", "Gender"}
RANDOM_STATE     = 42   # must match model_comparison.py

# =====================================================================
#  LOAD ARTIFACTS
# =====================================================================
print("Loading age model artifacts ...")
age_model      = joblib.load(AGE_MODEL_PATH)
age_selector   = joblib.load(SELECTOR_PATH)
selected_feats = joblib.load(FEAT_COLS_PATH)

print(f"  Model          : {type(age_model).__name__}")
print(f"  Features used  : {len(selected_feats)}")
print(f"  Feature list   : {selected_feats}")

# =====================================================================
#  LOAD DATA
# =====================================================================
df = pd.read_excel(DATA_PATH).dropna().reset_index(drop=True)

all_feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
X_all  = df[all_feature_cols]
y_all  = df["Age"]

# =====================================================================
#  HONEST HOLD-OUT EVALUATION
#  Re-create the same 80/20 split used during training so we can report
#  a genuine out-of-sample MAE (not inflated in-sample numbers).
# =====================================================================

def _age_bins(y, n_bins=6):
    """Same binning helper used in model_comparison.py for stratified split."""
    return pd.cut(y, bins=n_bins, labels=False)

age_bin_labels = _age_bins(y_all)
_, X_te, _, y_te = train_test_split(
    X_all, y_all,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=age_bin_labels,
)

X_te_sel = X_te[selected_feats]
y_te_pred = age_model.predict(X_te_sel)

mae_holdout  = mean_absolute_error(y_te, y_te_pred)
rmse_holdout = np.sqrt(mean_squared_error(y_te, y_te_pred))
r2_holdout   = r2_score(y_te, y_te_pred)

print("\n" + "=" * 55)
print("AGE PREDICTION  –  HOLD-OUT TEST SET  (n={})".format(len(y_te)))
print("=" * 55)
print(f"  MAE   : {mae_holdout:.3f} years  ← primary metric")
print(f"  RMSE  : {rmse_holdout:.3f} years")
print(f"  R²    : {r2_holdout:.4f}")
for n in [2, 3, 5]:
    within = (np.abs(y_te - y_te_pred) <= n).mean() * 100
    print(f"  Within {n} yrs  : {within:.1f}%")
print("=" * 55)

# =====================================================================
#  FULL-DATASET PREDICTIONS  (for saving to Excel & plots)
#  Note: training samples will show artificially low error here.
# =====================================================================
X_all_sel = X_all[selected_feats]
y_pred_all = age_model.predict(X_all_sel)

mae_full  = mean_absolute_error(y_all, y_pred_all)
rmse_full = np.sqrt(mean_squared_error(y_all, y_pred_all))
r2_full   = r2_score(y_all, y_pred_all)

print(f"\n  Full-dataset (in-sample for training rows):")
print(f"  MAE={mae_full:.3f}  RMSE={rmse_full:.3f}  R²={r2_full:.4f}")
print(f"  ⚠  These include training data → NOT a fair estimate of real error.")

# =====================================================================
#  SAVE PREDICTIONS
# =====================================================================

# Tag each row as Train/Test so readers know which errors are honest
_, _, idx_tr, idx_te = train_test_split(
    X_all.index, y_all.index,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=age_bin_labels,
)
split_label = pd.Series("Train", index=X_all.index)
split_label[idx_te] = "Test"

results = pd.DataFrame({
    "Image"          : df["Image"],
    "Split"          : split_label.values,          # NEW: Train/Test label
    "Actual Age"     : y_all.values,
    "Predicted Age"  : np.round(y_pred_all, 2),
    "Absolute Error" : np.round(np.abs(y_all.values - y_pred_all), 2),
    "Signed Error"   : np.round(y_pred_all - y_all.values, 2),
})
results.to_excel(OUTPUT_EXCEL, index=False)
print(f"\nPredictions saved → {OUTPUT_EXCEL}  (Split column marks Train/Test rows)")

# =====================================================================
#  PLOTS  (use hold-out subset so visuals reflect honest performance)
# =====================================================================

# 1. Actual vs Predicted  — hold-out only
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_te, y_te_pred, alpha=0.55, s=25, color="#4C72B0", edgecolors="none",
           label=f"Hold-out (n={len(y_te)})")
lims = [min(y_te.min(), y_te_pred.min()) - 1,
        max(y_te.max(), y_te_pred.max()) + 1]
ax.plot(lims, lims, "r--", linewidth=1.2, label="Perfect fit")
ax.set_xlabel("Actual Age (years)")
ax.set_ylabel("Predicted Age (years)")
ax.set_title(f"Age: Actual vs Predicted  (Hold-out MAE={mae_holdout:.2f} yrs, R²={r2_holdout:.3f})")
ax.legend()
plt.tight_layout()
plt.savefig("age_actual_vs_predicted.png", dpi=150)
plt.close()
print("Plot saved → age_actual_vs_predicted.png")

# 2. Residual plot  — hold-out only
residuals = y_te_pred - y_te
fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(y_te_pred, residuals, alpha=0.5, s=25, color="#DD8452", edgecolors="none")
ax.axhline(0, color="red", linewidth=1.2, linestyle="--")
ax.set_xlabel("Predicted Age (years)")
ax.set_ylabel("Residual (Predicted − Actual)")
ax.set_title(f"Age Prediction – Residual Plot  (Hold-out, n={len(y_te)})")
plt.tight_layout()
plt.savefig("age_residual_plot.png", dpi=150)
plt.close()
print("Plot saved → age_residual_plot.png")

# 3. Error distribution — hold-out only
abs_errs = np.abs(y_te - y_te_pred)
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(abs_errs, bins=15, color="#4C72B0", edgecolor="white", alpha=0.85)
ax.axvline(mae_holdout, color="red", linestyle="--", linewidth=1.5,
           label=f"MAE = {mae_holdout:.2f} yrs")
ax.set_xlabel("Absolute Error (years)")
ax.set_ylabel("Count")
ax.set_title("Age Prediction – Absolute Error Distribution (Hold-out)")
ax.legend()
plt.tight_layout()
plt.savefig("age_error_distribution.png", dpi=150)
plt.close()
print("Plot saved → age_error_distribution.png")


# =====================================================================
#  STANDALONE SINGLE-SAMPLE INFERENCE FUNCTION
#  Safe to import: guarded by __name__ check above so module-level
#  evaluation only runs when executed directly, not on import.
# =====================================================================
def predict_age_from_features(feature_dict: dict) -> float:
    """
    Predict skeletal age from a dict of morphological features.

    Parameters
    ----------
    feature_dict : dict
        Keys = morphological feature names (all columns except Image/Age/Gender).

    Returns
    -------
    float  –  predicted age in years
    """
    row = pd.DataFrame([feature_dict])[selected_feats]
    return float(age_model.predict(row)[0])

if __name__ == "__main__":
    image_name = "0015036.png"  # change this to your image filename

    row = df[df["Image"] == image_name]
    if row.empty:
        print(f"Image '{image_name}' not found in dataset.")
    else:
        sample_row = row[all_feature_cols].iloc[0]
        pred   = predict_age_from_features(sample_row.to_dict())
        actual = row["Age"].iloc[0]
        print(f"  Actual : {actual}  |  Predicted : {pred:.2f}  |  Error : {abs(actual - pred):.2f} yrs")