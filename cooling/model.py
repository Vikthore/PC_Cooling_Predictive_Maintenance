import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib, os

def train_iforest(X: pd.DataFrame, random_state: int = 42) -> Pipeline:
    model = Pipeline([("scaler", StandardScaler()), ("if", IsolationForest(n_estimators=300, contamination=0.12, random_state=random_state))])
    model.fit(X)
    return model

def score_iforest(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return -model.decision_function(X)  # higher => worse

def severity_from_scores(scores: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(scores, 5), np.percentile(scores, 95)
    sev = 100 * (scores - lo) / (hi - lo + 1e-9)
    return np.clip(sev, 0, 100)

def band_status(severity: float, green_hi: int = 30, yellow_hi: int = 70) -> str:
    if severity < green_hi:
        return "🟢 Healthy"
    elif severity < yellow_hi:
        return "🟡 Warning"
    else:
        return "🔴 Faulty"

def save_model(model, calib, path: str):
    blob = {"model": model, "calib": calib}
    joblib.dump(blob, path)

def load_model(path: str):
    return joblib.load(path)
