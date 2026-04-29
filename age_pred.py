"""
age_pred.py  –  XGBoost Age Prediction  (Evaluation + Inference)

Loads the trained model and runs evaluation on hold-out data.
Also has a predict_age_from_features() function the Flask API calls.

One thing that burned me early: evaluating on the full dataset and
reporting that MAE as the model's performance. Looks great on paper,
terrible in reality — training rows have near-zero error because
XGBoost memorises them. Added the honest hold-out block after that.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")   # no display needed — just saving PNGs
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# =====================================================================
#  CONFIG
# =====================================================================

DATA_PATH      = "final_enhanced.xlsx"
AGE_MODEL_PATH = "best_age_model.pkl"
SELECTOR_PATH  = "age_feature_selector.pkl"
FEAT_COLS_PATH = "age_selected_features.pkl"
OUTPUT_EXCEL   = "predicted_ages_final.xlsx"

EXCLUDE_COLS = {"Image", "Age", "Gender"}
RANDOM_STATE = 42   # must match whatever was used in training
                    # changing this would give a different split
                    # and incomparable results


# =====================================================================
#  LOAD ARTIFACTS
# =====================================================================

print("Loading age model artifacts ...")
age_model      = joblib.load(AGE_MODEL_PATH)
age_selector   = joblib.load(SELECTOR_PATH)
selected_feats = joblib.load(FEAT_COLS_PATH)
# selected_feats is a saved list — not recomputed here
# recomputing SHAP selection on a different subset would give
# different features and break the model

print(f"  Model         : {type(age_model).__name__}")
print(f"  Features used : {len(selected_feats)}")
print(f"  Feature list  : {selected_feats}")


# =====================================================================
#  LOAD DATA
# =====================================================================

df = pd.read_excel(DATA_PATH).dropna().reset_index(drop=True)
# dropna removes the ~4% of samples where mask extraction failed
# and feature values came through as NaN

all_feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
X_all = df[all_feature_cols]
y_all = df["Age"]


# =====================================================================
#  HONEST HOLD-OUT EVALUATION
#
#  Recreating the exact same 80/20 split from training.
#  Same random_state + same stratification = identical split.
#  This is the only MAE number worth reporting.
#
#  Stratifying by age bins because raw stratify=y doesn't work
#  for continuous targets — binning makes the age distribution
#  consistent across train/test folds.
# =====================================================================

def _age_bins(y, n_bins=6):
    # same binning used in model_comparison.py
    # 6 bins over the age range gave balanced splits
    # fewer bins → some folds had no young patients
    return pd.cut(y, bins=n_bins, labels=False)

age_bin_labels = _age_bins(y_all)

_, X_te, _, y_te = train_test_split(
    X_all, y_all,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=age_bin_labels,
)

X_te_sel  = X_te[selected_feats]
y_te_pred = age_model.predict(X_te_sel)

mae_holdout  = mean_absolute_error(y_te, y_te_pred)
rmse_holdout = np.sqrt(mean_squared_error(y_te, y_te_pred))
r2_holdout   = r2_score(y_te, y_te_pred)

print("\n" + "=" * 55)
print(f"AGE PREDICTION  –  HOLD-OUT TEST SET  (n={len(y_te)})")
print("=" * 55)
print(f"  MAE   : {mae_holdout:.3f} years  ← primary metric")
print(f"  RMSE  : {rmse_holdout:.3f} years")
print(f"  R²    : {r2_holdout:.4f}")

# within-N-years accuracy — more clinically interpretable than MAE alone
# an orthodontist cares whether the estimate is within 2 years, not the
# exact average error
for n in [2, 3, 5]:
    within = (np.abs(y_te - y_te_pred) <= n).mean() * 100
    print(f"  Within {n} yrs  : {within:.1f}%")
print("=" * 55)


# =====================================================================
#  FULL-DATASET PREDICTIONS  (for Excel output and downstream use)
#
#  Running on all samples so every patient gets a saved prediction.
#  Training rows will show artificially low error — XGBoost memorises
#  them to some degree. The Split column in the Excel output marks
#  which rows are honest test predictions vs training predictions.
# =====================================================================

X_all_sel  = X_all[selected_feats]
y_pred_all = age_model.predict(X_all_sel)

mae_full  = mean_absolute_error(y_all, y_pred_all)
rmse_full = np.sqrt(mean_squared_error(y_all, y_pred_all))
r2_full   = r2_score(y_all, y_pred_all)

print(f"\nFull-dataset MAE={mae_full:.3f}  RMSE={rmse_full:.3f}  R²={r2_full:.4f}")
print(f"  These include training data — not a fair estimate of real error.")


# =====================================================================
#  SAVE PREDICTIONS TO EXCEL
# =====================================================================

# reconstruct index split to tag each row
_, _, idx_tr, idx_te = train_test_split(
    X_all.index, y_all.index,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=age_bin_labels,
)

split_label          = pd.Series("Train", index=X_all.index)
split_label[idx_te]  = "Test"
# Test rows = honest predictions. Train rows = optimistic.
# Added Split column after realising the Excel was being used
# to report results without this distinction — misleading.

results = pd.DataFrame({
    "Image"         : df["Image"],
    "Split"         : split_label.values,
    "Actual Age"    : y_all.values,
    "Predicted Age" : np.round(y_pred_all, 2),
    "Absolute Error": np.round(np.abs(y_all.values - y_pred_all), 2),
    "Signed Error"  : np.round(y_pred_all - y_all.values, 2),
    # signed error shows bias direction —
    # positive = model overestimates age
    # negative = model underestimates
})
results.to_excel(OUTPUT_EXCEL, index=False)
print(f"\nPredictions saved → {OUTPUT_EXCEL}")


# =====================================================================
#  PLOTS  — all three use hold-out only
#  plotting full dataset looked better but was dishonest
# =====================================================================

# 1. Actual vs Predicted
# points on the red diagonal = perfect prediction
# consistent offset below diagonal = systematic underestimation
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(
    y_te, y_te_pred,
    alpha=0.55, s=25,
    color="#4C72B0", edgecolors="none",
    label=f"Hold-out (n={len(y_te)})"
)
lims = [
    min(y_te.min(), y_te_pred.min()) - 1,
    max(y_te.max(), y_te_pred.max()) + 1,
]
ax.plot(lims, lims, "r--", linewidth=1.2, label="Perfect fit")
ax.set_xlabel("Actual Age (years)")
ax.set_ylabel("Predicted Age (years)")
ax.set_title(
    f"Age: Actual vs Predicted  "
    f"(Hold-out MAE={mae_holdout:.2f} yrs, R²={r2_holdout:.3f})"
)
ax.legend()
plt.tight_layout()
plt.savefig("age_actual_vs_predicted.png", dpi=150)
plt.close()
print("Saved → age_actual_vs_predicted.png")

# 2. Residual plot
# random scatter around 0 = no systematic bias
# pattern (curve, funnel) = model has a structural problem
residuals = y_te_pred - y_te
fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(
    y_te_pred, residuals,
    alpha=0.5, s=25,
    color="#DD8452", edgecolors="none"
)
ax.axhline(0, color="red", linewidth=1.2, linestyle="--")
ax.set_xlabel("Predicted Age (years)")
ax.set_ylabel("Residual (Predicted − Actual)")
ax.set_title(f"Age Prediction – Residual Plot  (Hold-out, n={len(y_te)})")
plt.tight_layout()
plt.savefig("age_residual_plot.png", dpi=150)
plt.close()
print("Saved → age_residual_plot.png")

# 3. Absolute error distribution
# want this skewed left — most errors small, few large
# a symmetric bell means the model is consistently uncertain
abs_errs = np.abs(y_te - y_te_pred)
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(abs_errs, bins=15,
        color="#4C72B0", edgecolor="white", alpha=0.85)
ax.axvline(
    mae_holdout, color="red",
    linestyle="--", linewidth=1.5,
    label=f"MAE = {mae_holdout:.2f} yrs"
)
ax.set_xlabel("Absolute Error (years)")
ax.set_ylabel("Count")
ax.set_title("Age Prediction – Error Distribution (Hold-out)")
ax.legend()
plt.tight_layout()
plt.savefig("age_error_distribution.png", dpi=150)
plt.close()
print("Saved → age_error_distribution.png")


# =====================================================================
#  SINGLE-SAMPLE INFERENCE  — called by Flask API
#
#  Kept separate from the evaluation block above so the API can
#  import just this function without triggering the full evaluation.
#  The if __name__ == "__main__" guard below does the same for the
#  test call at the bottom.
# =====================================================================

def predict_age_from_features(feature_dict: dict) -> float:
    """
    Predict skeletal age from extracted morphological features.

    Parameters
    ----------
    feature_dict : dict
        Feature names → values. Must contain all keys in selected_feats.
        Extra keys are ignored — only selected_feats columns are used.

    Returns
    -------
    float  –  predicted skeletal age in years
    """
    # subset to selected features in the exact order the model expects
    # wrong column order would silently give wrong predictions with XGBoost
    row = pd.DataFrame([feature_dict])[selected_feats]
    return float(age_model.predict(row)[0])


# =====================================================================
#  QUICK SINGLE-SAMPLE TEST
# =====================================================================

if __name__ == "__main__":
    image_name = "0015036.png"

    row = df[df["Image"] == image_name]
    if row.empty:
        print(f"Image '{image_name}' not found in dataset.")
    else:
        sample_features = row[all_feature_cols].iloc[0].to_dict()
        pred   = predict_age_from_features(sample_features)
        actual = row["Age"].iloc[0]
        error  = abs(actual - pred)
        print(f"  Actual : {actual}  |  Predicted : {pred:.2f}  |  Error : {error:.2f} yrs")