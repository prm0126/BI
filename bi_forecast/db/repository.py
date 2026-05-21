"""Repository functions — narrow data-access API used by the service layer."""

from datetime import datetime
from typing import Iterable, List, Optional

import pandas as pd
from sqlalchemy import select, desc
from sqlalchemy.orm import Session

from .models import Appointment, DeliveryLog, HL7Message, ModelVersion, Patient, Prediction


def upsert_patient(s: Session, mrn: str, **fields) -> Patient:
    p = s.scalar(select(Patient).where(Patient.mrn == mrn))
    if p is None:
        p = Patient(mrn=mrn, **fields)
        s.add(p)
        s.flush()
    else:
        for k, v in fields.items():
            if v is not None:
                setattr(p, k, v)
    return p


def upsert_appointment(s: Session, external_id: str, **fields) -> Appointment:
    a = s.scalar(select(Appointment).where(Appointment.external_id == external_id))
    if a is None:
        a = Appointment(external_id=external_id, **fields)
        s.add(a)
        s.flush()
    else:
        for k, v in fields.items():
            if v is not None:
                setattr(a, k, v)
    return a


def ingest_appointment_dataframe(s: Session, df: pd.DataFrame, source: str = "upload") -> int:
    """Bulk-ingest a DataFrame of appointments with patient links by MRN."""
    n = 0
    for _, row in df.iterrows():
        mrn = str(row.get("patient_id") or row.get("mrn") or "").strip()
        patient = None
        if mrn:
            patient = upsert_patient(
                s, mrn,
                name=str(row["name"]) if "name" in df.columns and row.get("name") else None,
                phone=str(row["phone"]) if "phone" in df.columns and row.get("phone") else None,
            )
        ext = str(row.get("appointment_id") or row.get("external_id") or "").strip()
        if not ext:
            continue
        ap_time = pd.to_datetime(row.get("appointment_time"), errors="coerce")
        upsert_appointment(
            s, ext,
            patient_id=patient.id if patient else None,
            appointment_time=ap_time.to_pydatetime() if pd.notnull(ap_time) else None,
            doctor=str(row["doctor"]) if "doctor" in df.columns and row.get("doctor") else None,
            location=str(row["location"]) if "location" in df.columns and row.get("location") else None,
            specialty=str(row["specialty"]) if "specialty" in df.columns and row.get("specialty") else None,
            status=str(row["status"]) if "status" in df.columns and row.get("status") else "scheduled",
            source=source,
        )
        n += 1
    return n


def record_prediction(
    s: Session, appointment_external_id: str, probability: float, risk_band: str,
    top_factors: Optional[list] = None, model_version_id: Optional[int] = None,
) -> Prediction:
    apt = s.scalar(select(Appointment).where(Appointment.external_id == appointment_external_id))
    if apt is None:
        apt = upsert_appointment(s, appointment_external_id, status="scheduled", source="api")
    p = Prediction(
        appointment_id=apt.id,
        probability=float(probability),
        risk_band=risk_band,
        top_factors=top_factors or [],
        model_version_id=model_version_id,
    )
    s.add(p)
    return p


def record_delivery(s: Session, **fields) -> DeliveryLog:
    log = DeliveryLog(**fields)
    s.add(log)
    return log


def record_hl7(s: Session, message_type: str, raw: str, parsed: dict) -> HL7Message:
    msg = HL7Message(message_type=message_type, raw=raw, parsed=parsed)
    s.add(msg)
    return msg


def record_model_version(s: Session, name: str, **fields) -> ModelVersion:
    mv = ModelVersion(name=name, **fields)
    s.add(mv)
    s.flush()
    return mv


def recent_predictions(s: Session, limit: int = 50) -> List[Prediction]:
    return list(s.scalars(select(Prediction).order_by(desc(Prediction.created_at)).limit(limit)))


def recent_deliveries(s: Session, limit: int = 50) -> List[DeliveryLog]:
    return list(s.scalars(select(DeliveryLog).order_by(desc(DeliveryLog.sent_at)).limit(limit)))


def recent_hl7(s: Session, limit: int = 50) -> List[HL7Message]:
    return list(s.scalars(select(HL7Message).order_by(desc(HL7Message.received_at)).limit(limit)))


def latest_model_version(s: Session, name: str = "noshow") -> Optional[ModelVersion]:
    return s.scalar(select(ModelVersion).where(ModelVersion.name == name).order_by(desc(ModelVersion.created_at)).limit(1))
