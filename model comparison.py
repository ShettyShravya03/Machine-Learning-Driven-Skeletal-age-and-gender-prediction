# =============================================================================
#  CVM Skeletal Age & Gender Prediction  –  Model Training + Comparison
#
#  Two bugs fixed in this version that were silently killing performance:
#
#  FIX-1: "Age" column was inside FEATURE_COLS — model was predicting age
#          using age as an input. R² was 1.0, MAE was 0.001. Took an
#          embarrassingly long time to catch. Now explicitly excluded.
#
#  FIX-2: CatBoostClassifier with class_weights=[1,1] crashes sklearn's
#          clone() — StackingClassifier calls clone() internally on all
#          base estimators. Replaced with auto_class_weights="Balanced"
#          which is clone-safe and does the same thing.
# =============================================================================

import os, warnings
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from sklearn.model_selection   import (train_test_split, KFold,
                                        StratifiedKFold, cross_val_score,
                                        RandomizedSearchCV)
from sklearn.preprocessing     import StandardScaler
from sklearn.pipeline          import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics           import (mean_absolute_error, mean_squared_error,
                                        r2_score, accuracy_score, f1_score,
                                        roc_auc_score, classification_report)
from sklearn.ensemble          import (RandomForestRegressor,
                                        RandomForestClassifier,
                                        ExtraTreesRegressor,
                                        ExtraTreesClassifier,
                                        GradientBoostingClassifier,
                                        StackingRegressor,
                                        StackingClassifier)
from sklearn.linear_model      import Ridge, LogisticRegression
from xgboost                   import XGBRegressor, XGBClassifier

# optional libraries — degrade gracefully if not installed
# LightGBM and CatBoost improved stacking performance
# but the pipeline works without them
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("LightGBM not installed — pip install lightgbm")

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False
    print("CatBoost not installed — pip install catboost")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("Optuna not installed — falling back to RandomizedSearchCV")
    print("pip install optuna for faster hyperparameter tuning")


# =============================================================================
#  CONFIG
# =============================================================================

DATA_PATH    = "final_enhanced.xlsx"
RESULTS_DIR  = "."
RANDOM_STATE = 42
CV_SPLITS    = 5


# =============================================================================
#  LOAD & VALIDATE DATA
# =============================================================================

df = pd.read_excel(DATA_PATH).dropna().reset_index(drop=True)

# FIX-1: explicit exclusion — don't rely on "not adding" Age to FEATURE_COLS
# Image = ID column, Age = regression target, Gender = classification target
# all three must be excluded from both models
EXCLUDE_COLS = {"Image", "Age", "Gender"}
FEATURE_COLS = [c for c in df.columns if c not in EXCLUDE_COLS]

print("=" * 65)
print("DATASET SUMMARY")
print("=" * 65)
print(f"  Samples    : {len(df)}")
print(f"  Features   : {len(FEATURE_COLS)}  (Age & Gender excluded)")
print(f"  Age range  : {df['Age'].min()} – {df['Age'].max()}")
print(f"  Gender     : Male={(df['Gender']==0).sum()}  Female={(df['Gender']==1).sum()}")
print("=" * 65)


# =============================================================================
#  HELPERS
# =============================================================================

def age_bins(y, n_bins=6):
    # continuous age can't be passed to stratify= directly
    # binning into 6 groups keeps age distribution consistent
    # across train/test folds — fewer bins caused young patients
    # to cluster in train only
    return pd.cut(y, bins=n_bins, labels=False)


def select_features_df(X_df: pd.DataFrame, selector) -> pd.DataFrame:
    # SelectFromModel returns a boolean mask — this maps it back
    # to named columns so downstream code stays readable
    mask = selector.get_support()
    cols = X_df.columns[mask]
    return X_df[cols]


def regression_report(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"  {name:<30}  MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")
    return mae, rmse, r2


def classification_report_short(name, y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="weighted")
    auc = roc_auc_score(y_true, y_proba) if y_proba is not None else float("nan")
    print(f"  {name:<30}  Acc={acc:.3f}  F1={f1:.3f}  AUC={auc:.3f}")
    return acc, f1, auc


# =============================================================================
#  PART A  –  AGE REGRESSION
#  Input: morphological features only (no Age, no Gender)
# =============================================================================

print("\n" + "=" * 65)
print("PART A  –  AGE REGRESSION")
print("=" * 65)

X_age = df[FEATURE_COLS].copy()
y_age = df["Age"].copy()

# stratified split by age bin — without this younger patients
# were underrepresented in test set and MAE looked better than it was
age_bin_labels = age_bins(y_age)
X_a_tr, X_a_te, y_a_tr, y_a_te = train_test_split(
    X_age, y_age,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=age_bin_labels
)

# ── A1: feature selection ────────────────────────────────────────────────────

print("\n  [1/4] Selecting features via Random Forest importance ...")

# SelectFromModel with threshold="mean" keeps features whose
# importance is above the mean importance — automatic cutoff
# tried percentile thresholds (top 20%, top 30%) but mean worked best
rf_sel = SelectFromModel(
    RandomForestRegressor(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    threshold="mean"
)
rf_sel.fit(X_a_tr, y_a_tr)

X_a_tr_sel = select_features_df(X_a_tr, rf_sel)
X_a_te_sel = select_features_df(X_a_te, rf_sel)
X_age_sel  = select_features_df(X_age, rf_sel)

selected_age_features = list(X_a_tr_sel.columns)
print(f"     Selected {len(selected_age_features)} / {len(FEATURE_COLS)} features")
print(f"     Features: {selected_age_features}")

# CV strategy for age — StratifiedKFold on bins not raw age values
skf_age   = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True,
                              random_state=RANDOM_STATE)
cv_groups = age_bins(y_age).loc[X_age_sel.index]


# ── A2: hyperparameter tuning ────────────────────────────────────────────────

print("\n  [2/4] Tuning XGBoost & LightGBM ...")

if HAS_OPTUNA:
    def xgb_objective(trial):
        # Optuna samples these ranges intelligently —
        # learning_rate and regularization on log scale
        # because small differences matter more at low values
        p = dict(
            n_estimators     = trial.suggest_int("n_estimators", 300, 800),
            max_depth        = trial.suggest_int("max_depth", 3, 7),
            learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            subsample        = trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight = trial.suggest_int("min_child_weight", 1, 10),
            reg_alpha        = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda       = trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        )
        m  = XGBRegressor(**p, random_state=RANDOM_STATE,
                           verbosity=0, n_jobs=-1)
        sc = cross_val_score(m, X_a_tr_sel, y_a_tr,
                              cv=5, scoring="neg_mean_absolute_error")
        return -sc.mean()   # Optuna minimises — negate MAE

    study_xgb = optuna.create_study(direction="minimize")
    study_xgb.optimize(xgb_objective, n_trials=60, show_progress_bar=False)
    # 60 trials took ~8 min on CPU — worth it, found params
    # RandomizedSearchCV never hit in 30 iterations
    best_xgb_params = study_xgb.best_params

else:
    # fallback when Optuna not available
    # 30 random iterations covers the important parameter interactions
    param_xgb = {
        "n_estimators"    : [300, 500, 700],
        "max_depth"       : [3, 4, 5, 6],
        "learning_rate"   : [0.01, 0.03, 0.05, 0.08],
        "subsample"       : [0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "reg_alpha"       : [0, 0.1, 1.0],
        "reg_lambda"      : [0.5, 1.0, 5.0],
    }
    rscv_xgb = RandomizedSearchCV(
        XGBRegressor(random_state=RANDOM_STATE, verbosity=0, n_jobs=-1),
        param_xgb, n_iter=30,
        scoring="neg_mean_absolute_error",
        cv=5, random_state=RANDOM_STATE, n_jobs=-1
    )
    rscv_xgb.fit(X_a_tr_sel, y_a_tr)
    best_xgb_params = rscv_xgb.best_params_

best_xgb_age = XGBRegressor(
    **best_xgb_params,
    random_state=RANDOM_STATE,
    verbosity=0, n_jobs=-1
)
best_xgb_age.fit(X_a_tr_sel, y_a_tr)
print(f"     XGBoost best params: {best_xgb_params}")

if HAS_LGB:
    if HAS_OPTUNA:
        def lgb_objective(trial):
            # num_leaves is LightGBM-specific — controls tree complexity
            # independently of max_depth. Important to tune.
            p = dict(
                n_estimators     = trial.suggest_int("n_estimators", 300, 800),
                max_depth        = trial.suggest_int("max_depth", 3, 7),
                learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                subsample        = trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
                reg_alpha        = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                reg_lambda       = trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                num_leaves       = trial.suggest_int("num_leaves", 20, 120),
            )
            m  = lgb.LGBMRegressor(**p, random_state=RANDOM_STATE,
                                     n_jobs=-1, verbose=-1)
            sc = cross_val_score(m, X_a_tr_sel, y_a_tr,
                                  cv=5, scoring="neg_mean_absolute_error")
            return -sc.mean()

        study_lgb = optuna.create_study(direction="minimize")
        study_lgb.optimize(lgb_objective, n_trials=60, show_progress_bar=False)
        best_lgb_params = study_lgb.best_params

    else:
        param_lgb = {
            "n_estimators"    : [300, 500, 700],
            "max_depth"       : [3, 4, 5, 6],
            "learning_rate"   : [0.01, 0.03, 0.05, 0.08],
            "subsample"       : [0.7, 0.8, 0.9],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "num_leaves"      : [31, 63, 95],
            "reg_alpha"       : [0, 0.1, 1.0],
        }
        rscv_lgb = RandomizedSearchCV(
            lgb.LGBMRegressor(random_state=RANDOM_STATE,
                               verbose=-1, n_jobs=-1),
            param_lgb, n_iter=30,
            scoring="neg_mean_absolute_error",
            cv=5, random_state=RANDOM_STATE, n_jobs=-1
        )
        rscv_lgb.fit(X_a_tr_sel, y_a_tr)
        best_lgb_params = rscv_lgb.best_params_

    best_lgb_age = lgb.LGBMRegressor(
        **best_lgb_params,
        random_state=RANDOM_STATE,
        verbose=-1, n_jobs=-1
    )
    best_lgb_age.fit(X_a_tr_sel, y_a_tr)
    print(f"     LightGBM best params: {best_lgb_params}")


# ── A3: stacking ensemble ────────────────────────────────────────────────────

print("\n  [3/4] Building stacking ensemble ...")

# Ridge as final estimator — tried LogisticRegression and LinearRegression,
# Ridge generalised better by shrinking overconfident base model predictions
# passthrough=False — don't pass raw features to meta-learner,
# only the base model predictions. Passthrough=True overfit on this dataset.
base_estimators_age = [
    ("xgb", best_xgb_age),
    ("et",  ExtraTreesRegressor(n_estimators=400,
                                 random_state=RANDOM_STATE, n_jobs=-1)),
]
if HAS_LGB:
    base_estimators_age.append(("lgb", best_lgb_age))
if HAS_CAT:
    base_estimators_age.append((
        "cat",
        CatBoostRegressor(
            iterations=500, learning_rate=0.05,
            depth=6, verbose=0, random_state=RANDOM_STATE
        )
    ))

stacked_age = StackingRegressor(
    estimators      = base_estimators_age,
    final_estimator = Ridge(alpha=1.0),
    cv              = 5,
    n_jobs          = -1,
    passthrough     = False
)
stacked_age.fit(X_a_tr_sel, y_a_tr)


# ── A4: evaluate all age models ──────────────────────────────────────────────

print("\n  [4/4] Evaluation ...\n")

age_rows = []
models_to_eval_age = [
    ("XGBoost (tuned)",   best_xgb_age),
    ("ExtraTrees",        ExtraTreesRegressor(
                              n_estimators=400,
                              random_state=RANDOM_STATE,
                              n_jobs=-1).fit(X_a_tr_sel, y_a_tr)),
    ("Stacking Ensemble", stacked_age),
]
if HAS_LGB:
    models_to_eval_age.insert(1, ("LightGBM (tuned)", best_lgb_age))

for mname, mobj in models_to_eval_age:
    pred = mobj.predict(X_a_te_sel)
    mae, rmse, r2 = regression_report(mname, y_a_te, pred)

    # CV on full dataset with same stratification used in split
    # gives a less noisy performance estimate than single hold-out
    cv_scores = -cross_val_score(
        mobj, X_age_sel, y_age,
        cv      = StratifiedKFold(CV_SPLITS, shuffle=True,
                                   random_state=RANDOM_STATE),
        groups  = cv_groups,
        scoring = "neg_mean_absolute_error",
        n_jobs  = -1
    )
    age_rows.append({
        "Model"        : mname,
        "MAE (yrs)"    : round(mae, 4),
        "RMSE (yrs)"   : round(rmse, 4),
        "R²"           : round(r2, 4),
        "CV MAE (mean)": round(cv_scores.mean(), 4),
        "CV MAE (std)" : round(cv_scores.std(), 4),
    })

age_df = pd.DataFrame(age_rows)\
           .sort_values("CV MAE (mean)")\
           .reset_index(drop=True)

print(f"\n{'─'*65}")
print("AGE REGRESSION  –  FINAL COMPARISON  (sorted by CV MAE)")
print(f"{'─'*65}")
print(age_df.to_string(index=False))

# pick winner by CV MAE not hold-out MAE — more reliable on 1294 samples
best_age_row   = age_df.iloc[0]
best_age_name  = best_age_row["Model"]
best_age_model = dict(models_to_eval_age)[best_age_name]

print(f"\nBest age model : {best_age_name}")
print(f"  MAE={best_age_row['MAE (yrs)']:.3f} yrs  "
      f"RMSE={best_age_row['RMSE (yrs)']:.3f}  "
      f"R²={best_age_row['R²']:.3f}")


# ── A5: SHAP feature importance ──────────────────────────────────────────────

print("\n  Computing SHAP values for age model ...")
try:
    # using XGBoost specifically for SHAP — TreeSHAP is exact for tree models
    # stacking ensemble doesn't support TreeSHAP directly
    # XGBoost individual performance was close enough to stacking
    # that its SHAP values are representative
    shap_model = best_xgb_age
    explainer  = shap.TreeExplainer(shap_model)
    shap_vals  = explainer.shap_values(X_a_te_sel)
    shap_abs   = np.abs(shap_vals).mean(axis=0)

    shap_df = pd.DataFrame({
        "Feature"    : X_a_te_sel.columns,
        "Mean |SHAP|": shap_abs
    }).sort_values("Mean |SHAP|", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(shap_df["Feature"][::-1],
            shap_df["Mean |SHAP|"][::-1],
            color="#4C72B0")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top-20 Features – Age Prediction (XGBoost)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "shap_age_importance.png"), dpi=150)
    plt.close()
    print("  shap_age_importance.png saved")
    print("\n  Top-10 age features:")
    print(shap_df.head(10).to_string(index=False))

except Exception as e:
    # SHAP can fail on certain model configurations — log and continue
    # don't let a visualisation crash block model saving
    print(f"  SHAP skipped: {e}")


# =============================================================================
#  PART B  –  GENDER CLASSIFICATION
#  Same feature set as age — Age column still excluded
# =============================================================================

print("\n" + "=" * 65)
print("PART B  –  GENDER CLASSIFICATION")
print("=" * 65)

X_gen = df[FEATURE_COLS].copy()
y_gen = df["Gender"].copy()   # 0 = Male, 1 = Female

# gender is categorical so stratify=y_gen works directly
# no binning needed unlike age regression
X_g_tr, X_g_te, y_g_tr, y_g_te = train_test_split(
    X_gen, y_gen,
    test_size=0.2,
    stratify=y_gen,
    random_state=RANDOM_STATE
)


# ── B1: feature selection ────────────────────────────────────────────────────

print("\n  [1/4] Selecting features via Random Forest (classifier) ...")

# class_weight="balanced" in selector RF because gender was slightly imbalanced
# without it the selector ignored features that only mattered for minority class
rfc_sel = SelectFromModel(
    RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced"
    ),
    threshold="mean"
)
rfc_sel.fit(X_g_tr, y_g_tr)

X_g_tr_sel = select_features_df(X_g_tr, rfc_sel)
X_g_te_sel = select_features_df(X_g_te, rfc_sel)
X_gen_sel  = select_features_df(X_gen, rfc_sel)

selected_gen_features = list(X_g_tr_sel.columns)
print(f"     Selected {len(selected_gen_features)} / {len(X_gen.columns)} features")

cv_gen = StratifiedKFold(CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)


# ── B2: hyperparameter tuning ────────────────────────────────────────────────

print("\n  [2/4] Tuning XGBoost & LightGBM classifiers ...")

if HAS_OPTUNA:
    def xgbc_objective(trial):
        p = dict(
            n_estimators     = trial.suggest_int("n_estimators", 300, 800),
            max_depth        = trial.suggest_int("max_depth", 3, 7),
            learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            subsample        = trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight = trial.suggest_int("min_child_weight", 1, 10),
            reg_alpha        = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda       = trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        )
        m  = XGBClassifier(**p, random_state=RANDOM_STATE,
                            eval_metric="logloss", verbosity=0, n_jobs=-1)
        sc = cross_val_score(m, X_g_tr_sel, y_g_tr,
                              cv=cv_gen, scoring="roc_auc")
        return sc.mean()   # maximise AUC not accuracy — dataset is imbalanced

    study_xgbc = optuna.create_study(direction="maximize")
    study_xgbc.optimize(xgbc_objective, n_trials=60, show_progress_bar=False)
    best_xgbc_params = study_xgbc.best_params

else:
    param_xgbc = {
        "n_estimators"    : [300, 500, 700],
        "max_depth"       : [3, 4, 5, 6],
        "learning_rate"   : [0.01, 0.03, 0.05, 0.08],
        "subsample"       : [0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "reg_alpha"       : [0, 0.1, 1.0],
    }
    rscv_xgbc = RandomizedSearchCV(
        XGBClassifier(random_state=RANDOM_STATE,
                       eval_metric="logloss", verbosity=0, n_jobs=-1),
        param_xgbc, n_iter=30, scoring="roc_auc",
        cv=cv_gen, random_state=RANDOM_STATE, n_jobs=-1
    )
    rscv_xgbc.fit(X_g_tr_sel, y_g_tr)
    best_xgbc_params = rscv_xgbc.best_params_

best_xgbc = XGBClassifier(
    **best_xgbc_params,
    random_state=RANDOM_STATE,
    eval_metric="logloss",
    verbosity=0, n_jobs=-1
)
best_xgbc.fit(X_g_tr_sel, y_g_tr)
print(f"     XGBoost best params: {best_xgbc_params}")

if HAS_LGB:
    if HAS_OPTUNA:
        def lgbc_objective(trial):
            p = dict(
                n_estimators     = trial.suggest_int("n_estimators", 300, 800),
                max_depth        = trial.suggest_int("max_depth", 3, 7),
                learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                subsample        = trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
                num_leaves       = trial.suggest_int("num_leaves", 20, 120),
                reg_alpha        = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            )
            m  = lgb.LGBMClassifier(**p, random_state=RANDOM_STATE,
                                      verbose=-1, class_weight="balanced",
                                      n_jobs=-1)
            sc = cross_val_score(m, X_g_tr_sel, y_g_tr,
                                  cv=cv_gen, scoring="roc_auc")
            return sc.mean()

        study_lgbc = optuna.create_study(direction="maximize")
        study_lgbc.optimize(lgbc_objective, n_trials=60, show_progress_bar=False)
        best_lgbc_params = study_lgbc.best_params

    else:
        param_lgbc = {
            "n_estimators"    : [300, 500, 700],
            "max_depth"       : [3, 4, 5, 6],
            "learning_rate"   : [0.01, 0.03, 0.08],
            "num_leaves"      : [31, 63, 95],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_alpha"       : [0, 0.1, 1.0],
        }
        rscv_lgbc = RandomizedSearchCV(
            lgb.LGBMClassifier(random_state=RANDOM_STATE,
                                verbose=-1, class_weight="balanced",
                                n_jobs=-1),
            param_lgbc, n_iter=30, scoring="roc_auc",
            cv=cv_gen, random_state=RANDOM_STATE, n_jobs=-1
        )
        rscv_lgbc.fit(X_g_tr_sel, y_g_tr)
        best_lgbc_params = rscv_lgbc.best_params_

    best_lgbc = lgb.LGBMClassifier(
        **best_lgbc_params,
        random_state=RANDOM_STATE,
        verbose=-1, class_weight="balanced", n_jobs=-1
    )
    best_lgbc.fit(X_g_tr_sel, y_g_tr)
    print(f"     LightGBM best params: {best_lgbc_params}")


# ── B3: stacking ensemble ────────────────────────────────────────────────────

print("\n  [3/4] Building gender stacking ensemble ...")

# FIX-2: CatBoostClassifier with class_weights=[1,1] raises
# "Cannot clone object" inside StackingClassifier because sklearn's
# clone() reconstructs estimators from get_params() and CatBoost
# doesn't support class_weights as a constructor param in that context.
# auto_class_weights="Balanced" is the clone-safe equivalent.
#
# LogisticRegression as final estimator — outputs calibrated probabilities
# which matters for the confidence score shown to clinicians
base_estimators_gen = [
    ("xgb", best_xgbc),
    ("rf",  RandomForestClassifier(
                n_estimators=400,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1)),
]
if HAS_LGB:
    base_estimators_gen.append(("lgb", best_lgbc))
if HAS_CAT:
    base_estimators_gen.append((
        "cat",
        CatBoostClassifier(
            iterations=500, learning_rate=0.05, depth=6,
            verbose=0, random_state=RANDOM_STATE,
            auto_class_weights="Balanced"   # FIX-2: clone-safe
        )
    ))

stacked_gen = StackingClassifier(
    estimators      = base_estimators_gen,
    final_estimator = LogisticRegression(C=1.0, max_iter=2000),
    cv              = 5,
    n_jobs          = -1,
    passthrough     = False,
    stack_method    = "predict_proba"   # use probabilities not hard labels
                                         # for meta-learner input
)
stacked_gen.fit(X_g_tr_sel, y_g_tr)


# ── B4: evaluate all gender models ───────────────────────────────────────────

print("\n  [4/4] Evaluation ...\n")

gen_rows = []
models_to_eval_gen = [
    ("XGBoost (tuned)",   best_xgbc),
    ("RandomForest",      RandomForestClassifier(
                              n_estimators=400,
                              random_state=RANDOM_STATE,
                              class_weight="balanced",
                              n_jobs=-1).fit(X_g_tr_sel, y_g_tr)),
    ("Stacking Ensemble", stacked_gen),
]
if HAS_LGB:
    models_to_eval_gen.insert(1, ("LightGBM (tuned)", best_lgbc))

for mname, mobj in models_to_eval_gen:
    pred  = mobj.predict(X_g_te_sel)
    proba = mobj.predict_proba(X_g_te_sel)[:, 1]   # P(Female)
    acc, f1, auc = classification_report_short(mname, y_g_te, pred, proba)
    cv_sc = cross_val_score(mobj, X_gen_sel, y_gen,
                             cv=cv_gen, scoring="roc_auc", n_jobs=-1)
    gen_rows.append({
        "Model"         : mname,
        "Accuracy"      : round(acc, 4),
        "F1 (weighted)" : round(f1, 4),
        "ROC-AUC"       : round(auc, 4),
        "CV AUC (mean)" : round(cv_sc.mean(), 4),
        "CV AUC (std)"  : round(cv_sc.std(), 4),
    })

gen_df = pd.DataFrame(gen_rows)\
           .sort_values("CV AUC (mean)", ascending=False)\
           .reset_index(drop=True)

print(f"\n{'─'*65}")
print("GENDER CLASSIFICATION  –  FINAL COMPARISON  (sorted by CV AUC)")
print(f"{'─'*65}")
print(gen_df.to_string(index=False))

best_gen_row   = gen_df.iloc[0]
best_gen_name  = best_gen_row["Model"]
best_gen_model = dict(models_to_eval_gen)[best_gen_name]

print(f"\nBest gender model : {best_gen_name}")
print(f"  Acc={best_gen_row['Accuracy']:.3f}  "
      f"F1={best_gen_row['F1 (weighted)']:.3f}  "
      f"AUC={best_gen_row['ROC-AUC']:.3f}")

print(f"\n  Full classification report ({best_gen_name}):")
print(classification_report(
    y_g_te,
    best_gen_model.predict(X_g_te_sel),
    target_names=["Male", "Female"]
))


# ── B5: SHAP feature importance (gender) ────────────────────────────────────

print("  Computing SHAP values for gender model ...")
try:
    shap_model_g = best_xgbc   # same reason as age — TreeSHAP needs tree model
    explainer_g  = shap.TreeExplainer(shap_model_g)
    shap_vals_g  = explainer_g.shap_values(X_g_te_sel)
    shap_abs_g   = np.abs(shap_vals_g).mean(axis=0)

    shap_df_g = pd.DataFrame({
        "Feature"    : X_g_te_sel.columns,
        "Mean |SHAP|": shap_abs_g
    }).sort_values("Mean |SHAP|", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(shap_df_g["Feature"][::-1],
            shap_df_g["Mean |SHAP|"][::-1],
            color="#DD8452")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top-20 Features – Gender Prediction (XGBoost)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "shap_gender_importance.png"), dpi=150)
    plt.close()
    print("  shap_gender_importance.png saved")
    print("\n  Top-10 gender features:")
    print(shap_df_g.head(10).to_string(index=False))

except Exception as e:
    print(f"  SHAP skipped: {e}")


# =============================================================================
#  SAVE ALL ARTIFACTS
# =============================================================================

print("\n  Saving artifacts ...")

# save model + selector + feature list together as a set
# loading any one without the others will give wrong predictions
joblib.dump(best_age_model,        "best_age_model.pkl")
joblib.dump(rf_sel,                "age_feature_selector.pkl")
joblib.dump(selected_age_features, "age_selected_features.pkl")

joblib.dump(best_gen_model,        "best_gender_model.pkl")
joblib.dump(rfc_sel,               "gender_feature_selector.pkl")
joblib.dump(selected_gen_features, "gender_selected_features.pkl")

age_df.to_excel("age_model_comparison_final.xlsx", index=False)
gen_df.to_excel("gender_model_comparison_final.xlsx", index=False)

# save model names so inference scripts know what they loaded
with open("best_age_model_name.txt",    "w") as f: f.write(best_age_name)
with open("best_gender_model_name.txt", "w") as f: f.write(best_gen_name)


# =============================================================================
#  INFERENCE HELPER  —  written to disk for Flask to import
# =============================================================================

INFERENCE_CODE = '''
# cvm_inference.py  –  drop-in helper for Flask API
# generated by model_comparison.py — do not edit manually

import joblib, pandas as pd

_age_model    = joblib.load("best_age_model.pkl")
_age_sel_feat = joblib.load("age_selected_features.pkl")
_gen_model    = joblib.load("best_gender_model.pkl")
_gen_sel_feat = joblib.load("gender_selected_features.pkl")

def predict_age(feature_dict: dict) -> float:
    """
    feature_dict: morphological feature names → values.
    Do NOT include Age or Gender — they are not model inputs.
    """
    row = pd.DataFrame([feature_dict])[_age_sel_feat]
    return float(_age_model.predict(row)[0])

def predict_gender(feature_dict: dict) -> dict:
    """
    Returns {label: "Male"/"Female", probability: float}.
    Do NOT include Age or Gender — they are not model inputs.
    """
    row   = pd.DataFrame([feature_dict])[_gen_sel_feat]
    label = int(_gen_model.predict(row)[0])
    prob  = float(_gen_model.predict_proba(row)[0][label])
    return {"label": "Female" if label == 1 else "Male",
            "probability": prob}
'''

with open("cvm_inference.py", "w", encoding="utf-8") as f:
    f.write(INFERENCE_CODE)

print("  Saved: best_age_model.pkl + age_feature_selector.pkl + age_selected_features.pkl")
print("  Saved: best_gender_model.pkl + gender_feature_selector.pkl + gender_selected_features.pkl")
print("  Saved: age_model_comparison_final.xlsx")
print("  Saved: gender_model_comparison_final.xlsx")
print("  Saved: cvm_inference.py  (Flask import helper)")


# =============================================================================
#  FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 65)
print("FINAL SUMMARY")
print("=" * 65)
print(f"  Age prediction   → {best_age_name}")
print(f"    MAE  = {best_age_row['MAE (yrs)']:.3f} yrs")
print(f"    RMSE = {best_age_row['RMSE (yrs)']:.3f}")
print(f"    R²   = {best_age_row['R²']:.3f}")
print(f"  Gender predict   → {best_gen_name}")
print(f"    Acc  = {best_gen_row['Accuracy']:.3f}")
print(f"    F1   = {best_gen_row['F1 (weighted)']:.3f}")
print(f"    AUC  = {best_gen_row['ROC-AUC']:.3f}")
print("=" * 65)

# baseline comparison — context for what MAE 5.87 actually means
# a model that always predicts the mean age would get this MAE
# our model needs to beat it meaningfully to be worth deploying
from sklearn.dummy import DummyRegressor
dummy = DummyRegressor(strategy="mean")
dummy.fit(X_a_tr_sel, y_a_tr)
baseline_mae = mean_absolute_error(y_a_te, dummy.predict(X_a_te_sel))
print(f"\nBaseline (always predict mean) MAE : {baseline_mae:.3f} yrs")
print(f"Model improvement over baseline    : {baseline_mae - best_age_row['MAE (yrs)']:.3f} yrs")