# =============================================================================
#  CVM SKELETAL AGE & GENDER PREDICTION  –  FIXED PIPELINE
#  Fixes:
#    1. Age leakage: "Age" was in FEATURE_COLS — now explicitly excluded
#    2. CatBoost clone error: removed class_weights from CatBoostClassifier
#       (use auto_class_weights instead, which is clone-safe)
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

from sklearn.model_selection     import (train_test_split, KFold,
                                          StratifiedKFold, cross_val_score,
                                          RandomizedSearchCV)
from sklearn.preprocessing       import StandardScaler
from sklearn.pipeline            import Pipeline
from sklearn.feature_selection   import SelectFromModel
from sklearn.metrics             import (mean_absolute_error, mean_squared_error,
                                          r2_score, accuracy_score, f1_score,
                                          roc_auc_score, classification_report)
from sklearn.ensemble            import (RandomForestRegressor,
                                          RandomForestClassifier,
                                          ExtraTreesRegressor,
                                          ExtraTreesClassifier,
                                          GradientBoostingClassifier,
                                          StackingRegressor,
                                          StackingClassifier)
from sklearn.linear_model        import Ridge, LogisticRegression
from xgboost                     import XGBRegressor, XGBClassifier

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("⚠  LightGBM not installed – pip install lightgbm")

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False
    print("⚠  CatBoost not installed – pip install catboost")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("ℹ  Optuna not installed – using RandomizedSearchCV  (pip install optuna for faster tuning)")

# =============================================================================
#  CONFIG
# =============================================================================
DATA_PATH     = "final_enhanced.xlsx"
RESULTS_DIR   = "."
RANDOM_STATE  = 42
CV_SPLITS     = 5

# =============================================================================
#  LOAD & VALIDATE DATA
# =============================================================================
df = pd.read_excel(DATA_PATH).dropna().reset_index(drop=True)

# ── FIX 1: Explicitly exclude "Age" and "Gender" from feature columns ─────────
# Previously "Age" was left in FEATURE_COLS, causing perfect leakage (R²=1.0).
# "Image" is also excluded as it's an ID column.
EXCLUDE_COLS = {"Image", "Age", "Gender"}
FEATURE_COLS = [c for c in df.columns if c not in EXCLUDE_COLS]

print("=" * 65)
print("DATASET SUMMARY")
print("=" * 65)
print(f"  Samples        : {len(df)}")
print(f"  Features       : {len(FEATURE_COLS)}  (Age & Gender excluded from predictors)")
print(f"  Age range      : {df['Age'].min()} – {df['Age'].max()}")
print(f"  Gender dist    : Male={(df['Gender']==0).sum()}  Female={(df['Gender']==1).sum()}")
print("=" * 65)

# =============================================================================
#  HELPER – age-bin stratification for CV
# =============================================================================
def age_bins(y, n_bins=6):
    return pd.cut(y, bins=n_bins, labels=False)

# =============================================================================
#  HELPER – keep feature names through SelectFromModel
# =============================================================================
def select_features_df(X_df: pd.DataFrame, selector) -> pd.DataFrame:
    mask = selector.get_support()
    cols = X_df.columns[mask]
    return X_df[cols]

# =============================================================================
#  HELPER – print & return metrics
# =============================================================================
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
#  ██████  PART A – AGE REGRESSION
#  Features: all morphological columns (Age and Gender excluded)
#  Note: Gender is also excluded here to keep age prediction unbiased;
#        you can add it back by including "Gender" in X_age below if desired.
# =============================================================================
print("\n" + "=" * 65)
print("PART A  –  AGE REGRESSION")
print("=" * 65)

X_age = df[FEATURE_COLS].copy()   # pure morphological features only
y_age = df["Age"].copy()

# ── Stratified split (by age bin) ────────────────────────────────────────────
age_bin_labels = age_bins(y_age)
X_a_tr, X_a_te, y_a_tr, y_a_te = train_test_split(
    X_age, y_age, test_size=0.2, random_state=RANDOM_STATE,
    stratify=age_bin_labels
)

# ── Feature selection (RF-based) ──────────────────────────────────────────────
print("\n  [1/4] Selecting features via Random Forest importance …")
rf_sel = SelectFromModel(
    RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    threshold="mean"
)
rf_sel.fit(X_a_tr, y_a_tr)

X_a_tr_sel = select_features_df(X_a_tr, rf_sel)
X_a_te_sel = select_features_df(X_a_te, rf_sel)
X_age_sel  = select_features_df(X_age,  rf_sel)

selected_age_features = list(X_a_tr_sel.columns)
n_sel = len(selected_age_features)
print(f"     Selected {n_sel} / {len(FEATURE_COLS)} features")
print(f"     Features: {selected_age_features}")

# ── Cross-validation strategy (stratified by age bin) ────────────────────────
skf_age   = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
cv_groups = age_bins(y_age).loc[X_age_sel.index]

# =============================================================================
#  A1 – Hyperparameter tuning
# =============================================================================
print("\n  [2/4] Tuning XGBoost & LightGBM …")

if HAS_OPTUNA:
    def xgb_objective(trial):
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
        m  = XGBRegressor(**p, random_state=RANDOM_STATE, verbosity=0, n_jobs=-1)
        sc = cross_val_score(m, X_a_tr_sel, y_a_tr, cv=5,
                             scoring="neg_mean_absolute_error")
        return -sc.mean()

    study_xgb = optuna.create_study(direction="minimize")
    study_xgb.optimize(xgb_objective, n_trials=60, show_progress_bar=False)
    best_xgb_params = study_xgb.best_params
else:
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
        param_xgb, n_iter=30, scoring="neg_mean_absolute_error",
        cv=5, random_state=RANDOM_STATE, n_jobs=-1
    )
    rscv_xgb.fit(X_a_tr_sel, y_a_tr)
    best_xgb_params = rscv_xgb.best_params_

best_xgb_age = XGBRegressor(
    **best_xgb_params, random_state=RANDOM_STATE, verbosity=0, n_jobs=-1
)
best_xgb_age.fit(X_a_tr_sel, y_a_tr)
print(f"     XGBoost best params: {best_xgb_params}")

if HAS_LGB:
    if HAS_OPTUNA:
        def lgb_objective(trial):
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
            m  = lgb.LGBMRegressor(**p, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
            sc = cross_val_score(m, X_a_tr_sel, y_a_tr, cv=5,
                                 scoring="neg_mean_absolute_error")
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
            lgb.LGBMRegressor(random_state=RANDOM_STATE, verbose=-1, n_jobs=-1),
            param_lgb, n_iter=30, scoring="neg_mean_absolute_error",
            cv=5, random_state=RANDOM_STATE, n_jobs=-1
        )
        rscv_lgb.fit(X_a_tr_sel, y_a_tr)
        best_lgb_params = rscv_lgb.best_params_

    best_lgb_age = lgb.LGBMRegressor(
        **best_lgb_params, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1
    )
    best_lgb_age.fit(X_a_tr_sel, y_a_tr)
    print(f"     LightGBM best params: {best_lgb_params}")

# =============================================================================
#  A2 – Stacking Ensemble
#  FIX 2 (applied here too): CatBoostRegressor is clone-safe by default
#  (no class_weights parameter), so it is fine to include here.
# =============================================================================
print("\n  [3/4] Building stacking ensemble …")

base_estimators_age = [
    ("xgb", best_xgb_age),
    ("et",  ExtraTreesRegressor(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1)),
]
if HAS_LGB:
    base_estimators_age.append(("lgb", best_lgb_age))
if HAS_CAT:
    base_estimators_age.append((
        "cat",
        CatBoostRegressor(iterations=500, learning_rate=0.05,
                           depth=6, verbose=0, random_state=RANDOM_STATE)
    ))

stacked_age = StackingRegressor(
    estimators      = base_estimators_age,
    final_estimator = Ridge(alpha=1.0),
    cv              = 5,
    n_jobs          = -1,
    passthrough     = False
)
stacked_age.fit(X_a_tr_sel, y_a_tr)

# =============================================================================
#  A3 – Evaluate all age models
# =============================================================================
print("\n  [4/4] Evaluation …\n")

age_rows = []
models_to_eval_age = [
    ("XGBoost (tuned)",   best_xgb_age),
    ("ExtraTrees",        ExtraTreesRegressor(n_estimators=400,
                              random_state=RANDOM_STATE, n_jobs=-1).fit(X_a_tr_sel, y_a_tr)),
    ("Stacking Ensemble", stacked_age),
]
if HAS_LGB:
    models_to_eval_age.insert(1, ("LightGBM (tuned)", best_lgb_age))

for mname, mobj in models_to_eval_age:
    pred = mobj.predict(X_a_te_sel)
    mae, rmse, r2 = regression_report(mname, y_a_te, pred)
    cv_scores = -cross_val_score(
        mobj, X_age_sel, y_age,
        cv      = StratifiedKFold(CV_SPLITS, shuffle=True, random_state=RANDOM_STATE),
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

age_df = pd.DataFrame(age_rows).sort_values("CV MAE (mean)").reset_index(drop=True)
print(f"\n{'─'*65}")
print("AGE REGRESSION  –  FINAL COMPARISON  (sorted by CV MAE)")
print(f"{'─'*65}")
print(age_df.to_string(index=False))

best_age_row   = age_df.iloc[0]
best_age_name  = best_age_row["Model"]
best_age_model = dict(models_to_eval_age)[best_age_name]
print(f"\n🏆  Best age model : {best_age_name}")
print(f"    MAE={best_age_row['MAE (yrs)']:.3f} yrs  "
      f"RMSE={best_age_row['RMSE (yrs)']:.3f}  R²={best_age_row['R²']:.3f}")

# =============================================================================
#  A4 – SHAP feature importance (age)
# =============================================================================
print("\n  Computing SHAP values for age model …")
try:
    shap_model = best_xgb_age
    explainer  = shap.TreeExplainer(shap_model)
    shap_vals  = explainer.shap_values(X_a_te_sel)
    shap_abs   = np.abs(shap_vals).mean(axis=0)
    shap_df    = pd.DataFrame({
        "Feature"    : X_a_te_sel.columns,
        "Mean |SHAP|": shap_abs
    }).sort_values("Mean |SHAP|", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(shap_df["Feature"][::-1], shap_df["Mean |SHAP|"][::-1], color="#4C72B0")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top-20 Features – Age Prediction (XGBoost)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "shap_age_importance.png"), dpi=150)
    plt.close()
    print("  ✅  shap_age_importance.png saved")
    print("\n  Top-10 age features:")
    print(shap_df.head(10).to_string(index=False))
except Exception as e:
    print(f"  ⚠  SHAP skipped: {e}")

# =============================================================================
#  ██████  PART B – GENDER CLASSIFICATION
# =============================================================================
print("\n" + "=" * 65)
print("PART B  –  GENDER CLASSIFICATION")
print("=" * 65)

X_gen = df[FEATURE_COLS].copy()   # FEATURE_COLS already excludes Age & Gender
y_gen = df["Gender"].copy()

X_g_tr, X_g_te, y_g_tr, y_g_te = train_test_split(
    X_gen, y_gen, test_size=0.2,
    stratify=y_gen, random_state=RANDOM_STATE
)

print("\n  [1/4] Selecting features via Random Forest (classifier) …")
rfc_sel = SelectFromModel(
    RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE,
                            n_jobs=-1, class_weight="balanced"),
    threshold="mean"
)
rfc_sel.fit(X_g_tr, y_g_tr)

X_g_tr_sel = select_features_df(X_g_tr, rfc_sel)
X_g_te_sel = select_features_df(X_g_te, rfc_sel)
X_gen_sel  = select_features_df(X_gen, rfc_sel)

selected_gen_features = list(X_g_tr_sel.columns)
print(f"     Selected {len(selected_gen_features)} / {len(X_gen.columns)} features")

cv_gen = StratifiedKFold(CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# =============================================================================
#  B1 – Tune XGBoost & LightGBM classifiers
# =============================================================================
print("\n  [2/4] Tuning XGBoost & LightGBM classifiers …")

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
        m  = XGBClassifier(**p, random_state=RANDOM_STATE, verbosity=0,
                            eval_metric="logloss", n_jobs=-1)
        sc = cross_val_score(m, X_g_tr_sel, y_g_tr, cv=cv_gen, scoring="roc_auc")
        return sc.mean()

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
        XGBClassifier(random_state=RANDOM_STATE, verbosity=0,
                       eval_metric="logloss", n_jobs=-1),
        param_xgbc, n_iter=30, scoring="roc_auc",
        cv=cv_gen, random_state=RANDOM_STATE, n_jobs=-1
    )
    rscv_xgbc.fit(X_g_tr_sel, y_g_tr)
    best_xgbc_params = rscv_xgbc.best_params_

best_xgbc = XGBClassifier(
    **best_xgbc_params, random_state=RANDOM_STATE,
    eval_metric="logloss", verbosity=0, n_jobs=-1
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
            m  = lgb.LGBMClassifier(**p, random_state=RANDOM_STATE, verbose=-1,
                                      class_weight="balanced", n_jobs=-1)
            sc = cross_val_score(m, X_g_tr_sel, y_g_tr, cv=cv_gen, scoring="roc_auc")
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
            lgb.LGBMClassifier(random_state=RANDOM_STATE, verbose=-1,
                                class_weight="balanced", n_jobs=-1),
            param_lgbc, n_iter=30, scoring="roc_auc",
            cv=cv_gen, random_state=RANDOM_STATE, n_jobs=-1
        )
        rscv_lgbc.fit(X_g_tr_sel, y_g_tr)
        best_lgbc_params = rscv_lgbc.best_params_

    best_lgbc = lgb.LGBMClassifier(
        **best_lgbc_params, random_state=RANDOM_STATE,
        verbose=-1, class_weight="balanced", n_jobs=-1
    )
    best_lgbc.fit(X_g_tr_sel, y_g_tr)
    print(f"     LightGBM best params: {best_lgbc_params}")

# =============================================================================
#  B2 – Stacking Ensemble
#  FIX 2: CatBoostClassifier with class_weights=[1,1] cannot be cloned by
#  sklearn. Two safe alternatives:
#    a) Use auto_class_weights="Balanced"  (clone-safe CatBoost param)
#    b) Simply omit class_weights (balanced already via LGB/RF in ensemble)
#  We use option (a) so class balance is still handled.
# =============================================================================
print("\n  [3/4] Building gender stacking ensemble …")

base_estimators_gen = [
    ("xgb", best_xgbc),
    ("rf",  RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE,
                                    class_weight="balanced", n_jobs=-1)),
]
if HAS_LGB:
    base_estimators_gen.append(("lgb", best_lgbc))
if HAS_CAT:
    # ── FIX: use auto_class_weights instead of class_weights list ─────────────
    base_estimators_gen.append((
        "cat",
        CatBoostClassifier(
            iterations=500, learning_rate=0.05, depth=6,
            verbose=0, random_state=RANDOM_STATE,
            auto_class_weights="Balanced"   # clone-safe; replaces class_weights=[1,1]
        )
    ))

stacked_gen = StackingClassifier(
    estimators       = base_estimators_gen,
    final_estimator  = LogisticRegression(C=1.0, max_iter=2000),
    cv               = 5,
    n_jobs           = -1,
    passthrough      = False,
    stack_method     = "predict_proba"
)
stacked_gen.fit(X_g_tr_sel, y_g_tr)

# =============================================================================
#  B3 – Evaluate all gender models
# =============================================================================
print("\n  [4/4] Evaluation …\n")

gen_rows = []
models_to_eval_gen = [
    ("XGBoost (tuned)",   best_xgbc),
    ("RandomForest",      RandomForestClassifier(n_estimators=400,
                              random_state=RANDOM_STATE, class_weight="balanced",
                              n_jobs=-1).fit(X_g_tr_sel, y_g_tr)),
    ("Stacking Ensemble", stacked_gen),
]
if HAS_LGB:
    models_to_eval_gen.insert(1, ("LightGBM (tuned)", best_lgbc))

for mname, mobj in models_to_eval_gen:
    pred  = mobj.predict(X_g_te_sel)
    proba = mobj.predict_proba(X_g_te_sel)[:, 1]
    acc, f1, auc = classification_report_short(mname, y_g_te, pred, proba)
    cv_sc = cross_val_score(mobj, X_gen_sel, y_gen,
                             cv=cv_gen, scoring="roc_auc", n_jobs=-1)
    gen_rows.append({
        "Model"          : mname,
        "Accuracy"       : round(acc, 4),
        "F1 (weighted)"  : round(f1, 4),
        "ROC-AUC"        : round(auc, 4),
        "CV AUC (mean)"  : round(cv_sc.mean(), 4),
        "CV AUC (std)"   : round(cv_sc.std(), 4),
    })

gen_df = pd.DataFrame(gen_rows).sort_values("CV AUC (mean)", ascending=False).reset_index(drop=True)
print(f"\n{'─'*65}")
print("GENDER CLASSIFICATION  –  FINAL COMPARISON  (sorted by CV AUC)")
print(f"{'─'*65}")
print(gen_df.to_string(index=False))

best_gen_row   = gen_df.iloc[0]
best_gen_name  = best_gen_row["Model"]
best_gen_model = dict(models_to_eval_gen)[best_gen_name]
print(f"\n🏆  Best gender model : {best_gen_name}")
print(f"    Acc={best_gen_row['Accuracy']:.3f}  "
      f"F1={best_gen_row['F1 (weighted)']:.3f}  "
      f"AUC={best_gen_row['ROC-AUC']:.3f}")

print(f"\n  Full classification report ({best_gen_name}):")
print(classification_report(
    y_g_te,
    best_gen_model.predict(X_g_te_sel),
    target_names=["Male", "Female"]
))

# =============================================================================
#  B4 – SHAP feature importance (gender)
# =============================================================================
print("  Computing SHAP values for gender model …")
try:
    shap_model_g = best_xgbc
    explainer_g  = shap.TreeExplainer(shap_model_g)
    shap_vals_g  = explainer_g.shap_values(X_g_te_sel)
    shap_abs_g   = np.abs(shap_vals_g).mean(axis=0)
    shap_df_g    = pd.DataFrame({
        "Feature"    : X_g_te_sel.columns,
        "Mean |SHAP|": shap_abs_g
    }).sort_values("Mean |SHAP|", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(shap_df_g["Feature"][::-1], shap_df_g["Mean |SHAP|"][::-1], color="#DD8452")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top-20 Features – Gender Prediction (XGBoost)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "shap_gender_importance.png"), dpi=150)
    plt.close()
    print("  ✅  shap_gender_importance.png saved")
    print("\n  Top-10 gender features:")
    print(shap_df_g.head(10).to_string(index=False))
except Exception as e:
    print(f"  ⚠  SHAP skipped: {e}")

# =============================================================================
#  SAVE ALL ARTIFACTS
# =============================================================================
print("\n  Saving artifacts …")

joblib.dump(best_age_model,         "best_age_model.pkl")
joblib.dump(rf_sel,                 "age_feature_selector.pkl")
joblib.dump(selected_age_features,  "age_selected_features.pkl")

joblib.dump(best_gen_model,         "best_gender_model.pkl")
joblib.dump(rfc_sel,                "gender_feature_selector.pkl")
joblib.dump(selected_gen_features,  "gender_selected_features.pkl")

age_df.to_excel("age_model_comparison_final.xlsx", index=False)
gen_df.to_excel("gender_model_comparison_final.xlsx", index=False)

with open("best_age_model_name.txt",    "w") as f: f.write(best_age_name)
with open("best_gender_model_name.txt", "w") as f: f.write(best_gen_name)

# =============================================================================
#  INFERENCE HELPER
# =============================================================================
INFERENCE_CODE = '''
# ── Drop-in inference helper for Flask API ─────────────────────────────────
import joblib, pandas as pd

_age_model    = joblib.load("best_age_model.pkl")
_age_sel_feat = joblib.load("age_selected_features.pkl")
_gen_model    = joblib.load("best_gender_model.pkl")
_gen_sel_feat = joblib.load("gender_selected_features.pkl")

def predict_age(feature_dict: dict) -> float:
    """feature_dict: {feature_name: value} for morphological features only.
       Do NOT include "Age" or "Gender" – they are not model inputs."""
    row = pd.DataFrame([feature_dict])[_age_sel_feat]
    return float(_age_model.predict(row)[0])

def predict_gender(feature_dict: dict) -> dict:
    """Returns {label: "Male"/"Female", probability: float}
       Do NOT include "Age" or "Gender" – they are not model inputs."""
    row   = pd.DataFrame([feature_dict])[_gen_sel_feat]
    label = int(_gen_model.predict(row)[0])
    prob  = float(_gen_model.predict_proba(row)[0][label])
    return {"label": "Female" if label == 1 else "Male", "probability": prob}
'''

with open("cvm_inference.py", "w", encoding="utf-8") as f:
    f.write(INFERENCE_CODE)

print("  ✅  best_age_model.pkl          + age_feature_selector.pkl")
print("  ✅  best_gender_model.pkl       + gender_feature_selector.pkl")
print("  ✅  age_model_comparison_final.xlsx")
print("  ✅  gender_model_comparison_final.xlsx")
print("  ✅  cvm_inference.py  (drop-in helper for Flask)")

# =============================================================================
#  FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 65)
print("FINAL SUMMARY")
print("=" * 65)
print(f"  🏆  Age prediction   → {best_age_name}")
print(f"      MAE  = {best_age_row['MAE (yrs)']:.3f} yrs")
print(f"      RMSE = {best_age_row['RMSE (yrs)']:.3f}")
print(f"      R²   = {best_age_row['R²']:.3f}")
print(f"  🏆  Gender predict   → {best_gen_name}")
print(f"      Acc  = {best_gen_row['Accuracy']:.3f}")
print(f"      F1   = {best_gen_row['F1 (weighted)']:.3f}")
print(f"      AUC  = {best_gen_row['ROC-AUC']:.3f}")
print("=" * 65)
print("\n✅  All done.")