"""FHIR R4 connector.

Maps FHIR Appointment / Encounter / Observation resources to the engine's
canonical tabular schema. Handles single resources or Bundle responses.
"""

from typing import Iterable, List, Optional
import pandas as pd


def _first_participant(actors: list, want: str) -> Optional[str]:
    for p in actors or []:
        ref = (p.get("actor") or {}).get("reference", "")
        if ref.startswith(want):
            return ref.split("/", 1)[1] if "/" in ref else ref
        if want.lower() in (p.get("actor", {}).get("display", "")).lower():
            return p["actor"]["display"]
    return None


def fhir_appointment_to_row(resource: dict) -> dict:
    """Map a FHIR Appointment resource to a canonical row."""
    return {
        "appointment_id": resource.get("id"),
        "status": resource.get("status"),
        "appointment_time": resource.get("start"),
        "end_time": resource.get("end"),
        "duration_min": resource.get("minutesDuration"),
        "patient_id": _first_participant(resource.get("participant", []), "Patient"),
        "doctor": _first_participant(resource.get("participant", []), "Practitioner"),
        "location": _first_participant(resource.get("participant", []), "Location"),
        "service_type": ((resource.get("serviceType") or [{}])[0].get("text"))
        if resource.get("serviceType")
        else None,
        "source": "FHIR:Appointment",
    }


def fhir_encounter_to_row(resource: dict) -> dict:
    """Map a FHIR Encounter resource to a canonical row."""
    period = resource.get("period") or {}
    subject = resource.get("subject") or {}
    return {
        "encounter_id": resource.get("id"),
        "status": resource.get("status"),
        "patient_id": subject.get("reference", "").replace("Patient/", "") or None,
        "start_time": period.get("start"),
        "end_time": period.get("end"),
        "class": (resource.get("class") or {}).get("code"),
        "type": ((resource.get("type") or [{}])[0].get("text"))
        if resource.get("type")
        else None,
        "source": "FHIR:Encounter",
    }


_MAPPERS = {
    "Appointment": fhir_appointment_to_row,
    "Encounter": fhir_encounter_to_row,
}


def fhir_bundle_to_dataframe(bundle: dict, resource_type: str = "Appointment") -> pd.DataFrame:
    """Flatten a FHIR Bundle into a DataFrame of canonical rows."""
    mapper = _MAPPERS.get(resource_type)
    if mapper is None:
        raise ValueError(f"Unsupported FHIR resource: {resource_type}")
    rows: List[dict] = []
    for entry in bundle.get("entry", []):
        r = entry.get("resource") or {}
        if r.get("resourceType") == resource_type:
            rows.append(mapper(r))
    return pd.DataFrame(rows)
