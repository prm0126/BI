"""Per-patient no-show classifier.

Predicts the probability that a scheduled appointment will be a no-show.
Designed to consume a tabular appointment dataset and return per-row scores
plus calibrated risk bands and feature importance.

This intentionally uses scikit-learn so it runs without xgboost installed.
If xgboost is available, `NoShowClassifier(backend="xgb")` will use it.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder


DEFAULT_FEATURES = [
    "age",
    "lead_time_days",
    "prior_no_shows",
    "prior_visits",
    "distance_km",
    "appointment_hour",
    "appointment_dow",
    "specialty",
    "insurance",
    "sms_reminder_sent",
]


@dataclass
class NoShowScore:
    appointment_id: str
    probability: float
    risk_band: str  # low | medium | high
    top_factors: List[str] = field(default_factory=list)


@dataclass
class NoShowEvaluation:
    auc: float
    n_train: int
    feature_importance: Dict[str, float]


class NoShowClassifier:
    def __init__(self, features: Optional[List[str]] = None, backend: str = "rf", random_state: int = 0):
        self.features = features or DEFAULT_FEATURES
        self.backend = backend
        self.random_state = random_state
        self._model = None
        self._encoders: Dict[str, LabelEncoder] = {}
        self._feature_names: List[str] = []

    def _build(self):
        if self.backend == "xgb":
            try:
                from xgboost import XGBClassifier  # type: ignore
                return XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.08,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="auc",
                    random_state=self.random_state,
                    tree_method="hist",
                )
            except Exception:
                pass
        return RandomForestClassifier(
            n_estimators=300, max_depth=10, class_weight="balanced", random_state=self.random_state, n_jobs=-1
        )

    def _encode(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        df = df.copy()
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                if fit:
                    enc = LabelEncoder()
                    df[col] = enc.fit_transform(df[col].astype(str)).astype(int)
                    self._encoders[col] = enc
                else:
                    enc = self._encoders.get(col)
                    if enc is None:
                        df[col] = 0
                    else:
                        mapping = {v: i for i, v in enumerate(enc.classes_)}
                        df[col] = df[col].astype(str).map(mapping).fillna(-1).astype(int)
        return df

    def fit(self, df: pd.DataFrame, target: str = "no_show") -> NoShowEvaluation:
        feature_cols = [c for c in self.features if c in df.columns]
        if not feature_cols:
            raise ValueError(f"None of the expected features {self.features} are in the dataframe.")
        self._feature_names = feature_cols
        X = self._encode(df[feature_cols], fit=True)
        y = df[target].astype(int).values

        self._model = self._build()
        try:
            cv = StratifiedKFold(n_splits=min(5, max(2, np.bincount(y).min())), shuffle=True, random_state=self.random_state)
            aucs = cross_val_score(self._build(), X, y, scoring="roc_auc", cv=cv)
            auc = float(np.mean(aucs))
        except Exception:
            auc = float("nan")

        self._model.fit(X, y)
        imp = {}
        if hasattr(self._model, "feature_importances_"):
            imp = dict(zip(feature_cols, self._model.feature_importances_.tolist()))
            imp = dict(sorted(imp.items(), key=lambda kv: kv[1], reverse=True))
        return NoShowEvaluation(auc=auc, n_train=len(df), feature_importance=imp)

    def predict(self, df: pd.DataFrame, id_col: str = "appointment_id") -> List[NoShowScore]:
        if self._model is None:
            raise RuntimeError("Model not fitted")
        feature_cols = [c for c in self._feature_names if c in df.columns]
        X = self._encode(df[feature_cols], fit=False)
        proba = self._model.predict_proba(X)[:, 1]

        importances = getattr(self._model, "feature_importances_", np.zeros(len(feature_cols)))
        top_global = [c for c, _ in sorted(zip(feature_cols, importances), key=lambda kv: kv[1], reverse=True)[:3]]

        scores: List[NoShowScore] = []
        for i, p in enumerate(proba):
            band = "high" if p >= 0.6 else "medium" if p >= 0.3 else "low"
            ap_id = str(df.iloc[i][id_col]) if id_col in df.columns else f"row_{i}"
            scores.append(NoShowScore(appointment_id=ap_id, probability=float(p), risk_band=band, top_factors=top_global))
        return scores

    def recommend(self, scores: List[NoShowScore]) -> List[dict]:
        """Map scores to operational actions."""
        out = []
        for s in scores:
            if s.risk_band == "high":
                action = "Send SMS + WhatsApp reminder 24h prior; phone-confirm 4h prior; offer slot to waitlist if not confirmed."
            elif s.risk_band == "medium":
                action = "Send SMS reminder 24h prior; allow waitlist overbooking by 1 slot."
            else:
                action = "Standard reminder; no overbooking."
            out.append({
                "appointment_id": s.appointment_id,
                "probability": s.probability,
                "risk_band": s.risk_band,
                "action": action,
                "drivers": s.top_factors,
            })
        return out


def synthesize_appointments(n: int = 2000, seed: int = 3) -> pd.DataFrame:
    """Generate a synthetic appointment dataset for demos/tests."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "appointment_id": [f"A{i:06d}" for i in range(n)],
        "age": rng.integers(5, 90, n),
        "lead_time_days": rng.integers(0, 60, n),
        "prior_no_shows": rng.integers(0, 8, n),
        "prior_visits": rng.integers(0, 40, n),
        "distance_km": rng.gamma(2.0, 4.0, n).round(1),
        "appointment_hour": rng.integers(8, 19, n),
        "appointment_dow": rng.integers(0, 7, n),
        "specialty": rng.choice(["GP", "Cardio", "Derm", "Ortho", "Peds"], n),
        "insurance": rng.choice(["A", "B", "C", "self_pay"], n),
        "sms_reminder_sent": rng.integers(0, 2, n),
    })
    # Truth model — higher risk with prior_no_shows, longer lead times, fewer reminders.
    logit = (
        -2.5
        + 0.30 * df["prior_no_shows"]
        + 0.03 * df["lead_time_days"]
        + 0.04 * df["distance_km"]
        - 0.8 * df["sms_reminder_sent"]
        + (df["specialty"] == "Derm") * 0.4
        + (df["insurance"] == "self_pay") * 0.5
    )
    p = 1 / (1 + np.exp(-logit))
    df["no_show"] = (rng.uniform(0, 1, n) < p).astype(int)
    return df
