
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
