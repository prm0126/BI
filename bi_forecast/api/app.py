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
import os
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ..router import ModelRouter
from ..explain import explain_forecast
from ..recommend import recommend_actions
from ..whatif import what_if, WhatIfScenario
from ..noshow import NoShowClassifier, synthesize_appointments
from ..connectors import parse_hl7_message, hl7_to_appointment, hl7_to_admission, fhir_bundle_to_dataframe
from ..engagement import Dispatcher, AppointmentContext
from ..db import init_db, session_scope, repository


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


class EngagementAppt(BaseModel):
    appointment_id: str
    name: str
    phone: str
    appointment_time: str
    doctor: str = "your doctor"
    location: str = "the clinic"
    clinic: str = "your clinic"
    risk_band: str = "low"


class EngagementRequest(BaseModel):
    appointments: List[EngagementAppt]
    channel: str = "console"  # "console" | "sms" | "whatsapp"
    skip_low_risk: bool = False


class HL7Request(BaseModel):
    message: str


class FHIRRequest(BaseModel):
    resource: Dict[str, Any]
    resource_type: str = "Appointment"


# -------------------- app state --------------------

class _State:
    noshow_model: Optional[NoShowClassifier] = None
    model_path: Optional[str] = None
    model_meta: Dict[str, Any] = {}
    dispatcher: Optional[Dispatcher] = None


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

def create_app(database_url: Optional[str] = None) -> FastAPI:
    app = FastAPI(title="bi-forecast", version="0.1.0")
    state = _State()
    init_db(url=database_url)

    @app.get("/health")
    def health():
        from ..db.session import get_database_url
        return {"ok": True, "service": "bi-forecast", "db": get_database_url()}

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
        with session_scope() as s:
            for sc in scores:
                repository.record_prediction(
                    s, sc.appointment_id, sc.probability, sc.risk_band, top_factors=sc.top_factors,
                )
        return _clean({"n": len(actions), "results": actions})

    @app.post("/noshow/retrain")
    def noshow_retrain():
        df = synthesize_appointments(n=3000)
        clf = NoShowClassifier()
        evaluation = clf.fit(df, target="no_show")
        state.noshow_model = clf
        state.model_meta = {"source": "synthetic", "n_train": evaluation.n_train, "auc": evaluation.auc}
        return _clean({
            "auc": evaluation.auc,
            "n_train": evaluation.n_train,
            "feature_importance": evaluation.feature_importance,
        })

    @app.post("/noshow/train")
    def noshow_train_from_upload(
        file: "UploadFile" = File(...),  # type: ignore  # noqa: F821
        target: str = Form("no_show"),
        save_to: Optional[str] = Form(None),
    ):
        df = pd.read_csv(file.file)
        if target not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target}' missing")
        clf = NoShowClassifier()
        evaluation = clf.fit(df, target=target)
        state.noshow_model = clf
        state.model_meta = {
            "source": file.filename, "n_train": evaluation.n_train, "auc": evaluation.auc,
            "trained_at": datetime.utcnow().isoformat(),
        }
        if save_to:
            os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
            clf.save(save_to)
            state.model_path = save_to
        mv_id = None
        with session_scope() as s:
            mv = repository.record_model_version(
                s, name="noshow", path=state.model_path, source=file.filename,
                n_train=evaluation.n_train,
                auc=None if evaluation.auc != evaluation.auc else evaluation.auc,
                feature_importance=evaluation.feature_importance,
            )
            s.flush()
            mv_id = mv.id
        return _clean({
            "auc": evaluation.auc,
            "n_train": evaluation.n_train,
            "feature_importance": evaluation.feature_importance,
            "saved_to": state.model_path,
            "model_version_id": mv_id,
        })

    @app.post("/noshow/load")
    def noshow_load(path: str = Form(...)):
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"Model file not found: {path}")
        state.noshow_model = NoShowClassifier.load(path)
        state.model_path = path
        state.model_meta = {"source": "loaded", "path": path}
        return {"loaded": path, "features": state.noshow_model._feature_names}

    @app.get("/noshow/status")
    def noshow_status():
        return {
            "loaded": state.noshow_model is not None,
            "path": state.model_path,
            "meta": _clean(state.model_meta),
        }

    @app.post("/engagement/preview")
    def engagement_preview(req: EngagementRequest):
        if state.dispatcher is None:
            state.dispatcher = Dispatcher(default_channel="console")
        appts = [AppointmentContext(**a.model_dump()) for a in req.appointments]
        previews = state.dispatcher.preview(appts, channel=req.channel)
        return _clean({
            "n": len(previews),
            "results": [
                {
                    "appointment_id": p.appointment_id,
                    "risk_band": p.risk_band,
                    "template_key": p.template_key,
                    "rendered_body": p.rendered_body,
                    "channel": p.delivery.channel,
                    "to": p.delivery.to,
                }
                for p in previews
            ],
        })

    @app.post("/engagement/send")
    def engagement_send(req: EngagementRequest):
        if state.dispatcher is None:
            state.dispatcher = Dispatcher(default_channel=req.channel or "console")
        appts = [AppointmentContext(**a.model_dump()) for a in req.appointments]
        results = state.dispatcher.send_batch(appts, channel=req.channel, skip_low_risk=req.skip_low_risk)
        with session_scope() as s:
            for r in results:
                repository.record_delivery(
                    s,
                    appointment_external_id=r.appointment_id,
                    channel=r.delivery.channel,
                    template_key=r.template_key,
                    to_address=r.delivery.to,
                    body=r.rendered_body,
                    status=r.delivery.status,
                    provider_id=r.delivery.provider_id,
                    error=r.delivery.error,
                )
        return _clean({
            "n": len(results),
            "results": [
                {
                    "appointment_id": r.appointment_id,
                    "risk_band": r.risk_band,
                    "template_key": r.template_key,
                    "channel": r.delivery.channel,
                    "to": r.delivery.to,
                    "status": r.delivery.status,
                    "provider_id": r.delivery.provider_id,
                    "error": r.delivery.error,
                }
                for r in results
            ],
        })

    @app.get("/engagement/log")
    def engagement_log(limit: int = 100):
        if state.dispatcher is None:
            return {"log": []}
        recent = state.dispatcher.log[-limit:]
        return _clean({
            "n": len(recent),
            "log": [
                {
                    "appointment_id": r.appointment_id,
                    "channel": r.delivery.channel,
                    "status": r.delivery.status,
                    "to": r.delivery.to,
                    "timestamp": r.delivery.timestamp.isoformat(),
                    "body": r.rendered_body,
                    "error": r.delivery.error,
                }
                for r in recent
            ],
        })

    @app.post("/hooks/hl7")
    def hl7_hook(req: HL7Request):
        try:
            msg = parse_hl7_message(req.message)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if msg.message_type.startswith("SIU"):
            row = hl7_to_appointment(msg)
            with session_scope() as s:
                repository.record_hl7(s, msg.message_type, req.message, row)
                if row.get("patient_id"):
                    repository.upsert_patient(s, row["patient_id"], name=row.get("patient_name"))
                if row.get("appointment_id"):
                    repository.upsert_appointment(
                        s, row["appointment_id"],
                        doctor=row.get("doctor"), location=row.get("location"),
                        status=row.get("status") or "scheduled", source="HL7:SIU",
                    )
            return {"type": "appointment", "row": row}
        if msg.message_type.startswith("ADT"):
            row = hl7_to_admission(msg)
            with session_scope() as s:
                repository.record_hl7(s, msg.message_type, req.message, row)
                if row.get("patient_id"):
                    repository.upsert_patient(s, row["patient_id"], name=row.get("patient_name"))
            return {"type": "admission", "row": row}
        with session_scope() as s:
            repository.record_hl7(s, msg.message_type, req.message, {"unrecognized": True})
        return {"type": "unknown", "message_type": msg.message_type}

    # ---------- /history endpoints ----------

    @app.get("/history/predictions")
    def history_predictions(limit: int = 50):
        with session_scope() as s:
            rows = repository.recent_predictions(s, limit=limit)
            return {"n": len(rows), "items": [
                {
                    "id": p.id, "appointment_id": p.appointment_id,
                    "probability": p.probability, "risk_band": p.risk_band,
                    "top_factors": p.top_factors,
                    "created_at": p.created_at.isoformat(),
                } for p in rows
            ]}

    @app.get("/history/deliveries")
    def history_deliveries(limit: int = 50):
        with session_scope() as s:
            rows = repository.recent_deliveries(s, limit=limit)
            return {"n": len(rows), "items": [
                {
                    "id": d.id, "appointment_external_id": d.appointment_external_id,
                    "channel": d.channel, "template_key": d.template_key,
                    "to_address": d.to_address, "status": d.status,
                    "provider_id": d.provider_id, "error": d.error,
                    "sent_at": d.sent_at.isoformat(),
                } for d in rows
            ]}

    @app.get("/history/hl7")
    def history_hl7(limit: int = 50):
        with session_scope() as s:
            rows = repository.recent_hl7(s, limit=limit)
            return {"n": len(rows), "items": [
                {
                    "id": m.id, "message_type": m.message_type,
                    "parsed": m.parsed, "received_at": m.received_at.isoformat(),
                } for m in rows
            ]}

    @app.get("/history/models")
    def history_models():
        with session_scope() as s:
            from sqlalchemy import select, desc
            from ..db.models import ModelVersion
            rows = list(s.scalars(select(ModelVersion).order_by(desc(ModelVersion.created_at))))
            return {"n": len(rows), "items": [
                {
                    "id": m.id, "name": m.name, "path": m.path, "source": m.source,
                    "n_train": m.n_train, "auc": m.auc,
                    "created_at": m.created_at.isoformat(),
                } for m in rows
            ]}

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
