"""FastAPI service exposing the AI engine.

Endpoints:
    GET  /health
    POST /forecast            route + forecast + explain + recommend
    POST /whatif              run scenarios against a series
    POST /noshow/score        score a batch of scheduled appointments
    POST /noshow/retrain      retrain the no-show model on synthetic data
    POST /hooks/hl7           accept HL7 v2 ADT/SIU messages
    POST /hooks/fhir          accept FHIR Bundles / single resources
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import math
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..router import ModelRouter
from ..explain import explain_forecast
from ..recommend import recommend_actions
from ..whatif import what_if, WhatIfScenario
from ..noshow import NoShowClassifier, synthesize_appointments
from ..connectors import parse_hl7_message, hl7_to_appointment, hl7_to_admission, fhir_bundle_to_dataframe


# -------------------- request/response models --------------------

class SeriesPoint(BaseModel):
    ds: datetime
    y: float


class ForecastRequest(BaseModel):
    target: str = Field(..., description="Target metric name, e.g. 'admissions'")
    series: List[SeriesPoint]
    horizon: int = 14
    allow: Optional[List[str]] = None


class ScenarioSpec(BaseModel):
    name: str
    multiplier: float = 1.0
    delta: float = 0.0
    on_last_n: Optional[int] = None


class WhatIfRequest(BaseModel):
    target: str
    series: List[SeriesPoint]
    horizon: int = 7
    scenarios: List[ScenarioSpec]


class AppointmentRow(BaseModel):
    appointment_id: str
    age: int
    lead_time_days: int
    prior_no_shows: int = 0
    prior_visits: int = 0
    distance_km: float = 0.0
    appointment_hour: int = 9
    appointment_dow: int = 0
    specialty: str = "GP"
    insurance: str = "A"
    sms_reminder_sent: int = 0


class NoShowRequest(BaseModel):
    appointments: List[AppointmentRow]


class HL7Request(BaseModel):
    message: str


class FHIRRequest(BaseModel):
    resource: Dict[str, Any]
    resource_type: str = "Appointment"


# -------------------- app state --------------------

class _State:
    noshow_model: Optional[NoShowClassifier] = None


def _series_to_pandas(points: List[SeriesPoint]) -> pd.Series:
    df = pd.DataFrame([p.model_dump() for p in points]).sort_values("ds")
    return pd.Series(df["y"].values, index=pd.DatetimeIndex(df["ds"]))


def _clean(obj):
    """Replace NaN/Inf with None recursively for JSON safety."""
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


def _train_default_noshow() -> NoShowClassifier:
    df = synthesize_appointments(n=2000)
    clf = NoShowClassifier()
    clf.fit(df, target="no_show")
    return clf


# -------------------- app factory --------------------

def create_app() -> FastAPI:
    app = FastAPI(title="bi-forecast", version="0.1.0")
    state = _State()

    @app.get("/health")
    def health():
        return {"ok": True, "service": "bi-forecast"}

    @app.post("/forecast")
    def forecast_endpoint(req: ForecastRequest):
        series = _series_to_pandas(req.series)
        router = ModelRouter(allow=req.allow)
        result, decision, _ = router.forecast(series, req.horizon, target_name=req.target)
        explanation = explain_forecast(result, series=series)
        recs = recommend_actions(result, explanation)
        return _clean({
            "model": result.model,
            "candidates": decision.candidates,
            "scores": decision.scores,
            "forecast": [
                {"ds": str(t), "yhat": v, "lower": lo, "upper": hi}
                for t, v, lo, hi in zip(result.forecast_index, result.forecast_values, result.lower, result.upper)
            ],
            "explanation": explanation,
            "recommendations": [r.to_dict() for r in recs],
        })

    @app.post("/whatif")
    def whatif_endpoint(req: WhatIfRequest):
        series = _series_to_pandas(req.series)
        scenarios = [WhatIfScenario(**s.model_dump()) for s in req.scenarios]
        results = what_if(series, req.horizon, scenarios, target_name=req.target)
        base_mean = sum(results["baseline"].forecast_values) / req.horizon
        out: Dict[str, Any] = {"baseline_mean": base_mean, "scenarios": {}}
        for name, r in results.items():
            if name == "baseline":
                continue
            mean = sum(r.forecast_values) / req.horizon
            out["scenarios"][name] = {
                "mean": mean,
                "delta_abs": mean - base_mean,
                "delta_pct": (mean - base_mean) / base_mean * 100 if base_mean else 0.0,
            }
        return _clean(out)

    @app.post("/noshow/score")
    def noshow_score(req: NoShowRequest):
        if state.noshow_model is None:
            state.noshow_model = _train_default_noshow()
        df = pd.DataFrame([a.model_dump() for a in req.appointments])
        scores = state.noshow_model.predict(df)
        actions = state.noshow_model.recommend(scores)
        return _clean({"n": len(actions), "results": actions})

    @app.post("/noshow/retrain")
    def noshow_retrain():
        df = synthesize_appointments(n=3000)
        clf = NoShowClassifier()
        evaluation = clf.fit(df, target="no_show")
        state.noshow_model = clf
        return _clean({
            "auc": evaluation.auc,
            "n_train": evaluation.n_train,
            "feature_importance": evaluation.feature_importance,
        })

    @app.post("/hooks/hl7")
    def hl7_hook(req: HL7Request):
        try:
            msg = parse_hl7_message(req.message)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if msg.message_type.startswith("SIU"):
            return {"type": "appointment", "row": hl7_to_appointment(msg)}
        if msg.message_type.startswith("ADT"):
            return {"type": "admission", "row": hl7_to_admission(msg)}
        return {"type": "unknown", "message_type": msg.message_type}

    @app.post("/hooks/fhir")
    def fhir_hook(req: FHIRRequest):
        res = req.resource
        if res.get("resourceType") == "Bundle":
            df = fhir_bundle_to_dataframe(res, resource_type=req.resource_type)
            return _clean({"type": "bundle", "rows": df.where(pd.notnull(df), None).to_dict(orient="records")})
        from ..connectors.fhir import fhir_appointment_to_row, fhir_encounter_to_row
        mapper = {"Appointment": fhir_appointment_to_row, "Encounter": fhir_encounter_to_row}.get(req.resource_type)
        if mapper is None:
            raise HTTPException(status_code=400, detail=f"Unsupported resource: {req.resource_type}")
        return _clean({"type": req.resource_type, "row": mapper(res)})

    return app


app = create_app()
