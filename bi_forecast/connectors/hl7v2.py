"""Minimal HL7 v2 parser.

Supports the core ops messages: ADT (admit/discharge/transfer), SIU
(scheduling), ORU (results), ORM (orders). Returns dicts mapped to the
engine's canonical schema instead of raw segment names.

This is intentionally a tiny pure-python parser — production usage
should sit behind Mirth Connect or hl7apy/python-hl7.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


SEGMENT_TERMINATORS = ("\r", "\n", "\r\n")


@dataclass
class HL7Message:
    message_type: str  # e.g. "ADT^A01"
    segments: Dict[str, List[List[str]]] = field(default_factory=dict)
    raw: str = ""

    def first(self, segment: str) -> Optional[List[str]]:
        seg = self.segments.get(segment)
        return seg[0] if seg else None

    def field(self, segment: str, field_idx: int, component_idx: int = 0) -> Optional[str]:
        s = self.first(segment)
        if not s or field_idx >= len(s):
            return None
        cell = s[field_idx]
        if "^" in cell:
            parts = cell.split("^")
            return parts[component_idx] if component_idx < len(parts) else None
        return cell if component_idx == 0 else None


def parse_hl7_message(text: str) -> HL7Message:
    """Parse an HL7 v2 pipe-delimited message into segments."""
    normalized = text.replace("\r\n", "\r").replace("\n", "\r").strip("\r")
    if not normalized.startswith("MSH"):
        raise ValueError("Not an HL7 v2 message (missing MSH).")

    segments: Dict[str, List[List[str]]] = {}
    msg_type = ""
    for line in normalized.split("\r"):
        if not line:
            continue
        fields = line.split("|")
        seg_name = fields[0]
        if seg_name == "MSH":
            fields.insert(1, "|")
            if len(fields) > 9:
                msg_type = fields[9]
        segments.setdefault(seg_name, []).append(fields)
    return HL7Message(message_type=msg_type, segments=segments, raw=text)


def _component(value: Optional[str], idx: int = 0) -> Optional[str]:
    if not value:
        return None
    parts = value.split("^")
    return parts[idx] if idx < len(parts) else None


def hl7_to_appointment(msg: HL7Message) -> dict:
    """Map an SIU message to the canonical appointment schema."""
    pid = msg.first("PID") or []
    sch = msg.first("SCH") or []
    pv1 = msg.first("PV1") or []

    return {
        "appointment_id": sch[1] if len(sch) > 1 else None,
        "patient_id": _component(pid[3] if len(pid) > 3 else None, 0),
        "patient_name": _component(pid[5] if len(pid) > 5 else None, 0),
        "appointment_time": sch[11] if len(sch) > 11 else None,
        "duration_min": sch[9] if len(sch) > 9 else None,
        "location": _component(pv1[3] if len(pv1) > 3 else None, 0),
        "doctor": _component(pv1[7] if len(pv1) > 7 else None, 1),
        "status": sch[25] if len(sch) > 25 else None,
        "source": "HL7:SIU",
    }


def hl7_to_admission(msg: HL7Message) -> dict:
    """Map an ADT message to a canonical admission/encounter row."""
    pid = msg.first("PID") or []
    pv1 = msg.first("PV1") or []
    evn = msg.first("EVN") or []

    return {
        "patient_id": _component(pid[3] if len(pid) > 3 else None, 0),
        "patient_name": _component(pid[5] if len(pid) > 5 else None, 0),
        "event_type": msg.message_type,
        "event_time": evn[2] if len(evn) > 2 else None,
        "patient_class": pv1[2] if len(pv1) > 2 else None,
        "location": _component(pv1[3] if len(pv1) > 3 else None, 0),
        "admit_reason": pv1[4] if len(pv1) > 4 else None,
        "doctor": _component(pv1[7] if len(pv1) > 7 else None, 1),
        "source": "HL7:ADT",
    }
