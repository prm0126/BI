"""SQLAlchemy ORM models for the bi-forecast platform.

Tables
------
patients          : MRN-keyed patient roster
appointments      : scheduled visits (one per appointment)
predictions       : per-appointment no-show predictions (history)
delivery_logs     : engagement messages dispatched
hl7_messages      : raw inbound HL7 v2 messages (audit trail)
model_versions    : metadata for each trained no-show model
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import String, Integer, Float, DateTime, ForeignKey, JSON, Text, Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    mrn: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[Optional[str]] = mapped_column(String(256))
    phone: Mapped[Optional[str]] = mapped_column(String(32))
    email: Mapped[Optional[str]] = mapped_column(String(256))
    dob: Mapped[Optional[str]] = mapped_column(String(32))
    sex: Mapped[Optional[str]] = mapped_column(String(8))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    appointments: Mapped[list["Appointment"]] = relationship(back_populates="patient")


class Appointment(Base):
    __tablename__ = "appointments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    external_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    patient_id: Mapped[Optional[int]] = mapped_column(ForeignKey("patients.id"), index=True)
    appointment_time: Mapped[Optional[datetime]] = mapped_column(DateTime, index=True)
    duration_min: Mapped[Optional[int]] = mapped_column(Integer)
    doctor: Mapped[Optional[str]] = mapped_column(String(128))
    location: Mapped[Optional[str]] = mapped_column(String(128))
    specialty: Mapped[Optional[str]] = mapped_column(String(64))
    status: Mapped[Optional[str]] = mapped_column(String(32), index=True)
    source: Mapped[Optional[str]] = mapped_column(String(32))  # HL7:SIU | FHIR:Appointment | upload | api
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    patient: Mapped[Optional[Patient]] = relationship(back_populates="appointments")
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="appointment")


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    appointment_id: Mapped[int] = mapped_column(ForeignKey("appointments.id"), index=True)
    model_version_id: Mapped[Optional[int]] = mapped_column(ForeignKey("model_versions.id"))
    probability: Mapped[float] = mapped_column(Float)
    risk_band: Mapped[str] = mapped_column(String(16), index=True)
    top_factors: Mapped[Optional[dict]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    appointment: Mapped[Appointment] = relationship(back_populates="predictions")


class DeliveryLog(Base):
    __tablename__ = "delivery_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    appointment_external_id: Mapped[Optional[str]] = mapped_column(String(128), index=True)
    channel: Mapped[str] = mapped_column(String(16), index=True)
    template_key: Mapped[Optional[str]] = mapped_column(String(64))
    to_address: Mapped[Optional[str]] = mapped_column(String(64))
    body: Mapped[Optional[str]] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), index=True)  # queued|delivered|failed|dry_run
    provider_id: Mapped[Optional[str]] = mapped_column(String(128))
    error: Mapped[Optional[str]] = mapped_column(Text)
    sent_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class HL7Message(Base):
    __tablename__ = "hl7_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    message_type: Mapped[Optional[str]] = mapped_column(String(32), index=True)
    raw: Mapped[str] = mapped_column(Text)
    parsed: Mapped[Optional[dict]] = mapped_column(JSON)
    received_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), index=True)  # e.g. "noshow"
    path: Mapped[Optional[str]] = mapped_column(String(512))
    source: Mapped[Optional[str]] = mapped_column(String(64))  # filename or 'synthetic'
    n_train: Mapped[Optional[int]] = mapped_column(Integer)
    auc: Mapped[Optional[float]] = mapped_column(Float)
    feature_importance: Mapped[Optional[dict]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


Index("ix_predictions_apt_created", Prediction.appointment_id, Prediction.created_at.desc())
