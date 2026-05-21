"""Dispatches reminders for scored appointments.

Inputs:
    - scored appointments (output of NoShowClassifier.recommend / predict)
    - patient contact info (phone, name)
    - appointment context (time, doctor, location, clinic)

Output:
    - DeliveryResult per appointment, plus an in-memory delivery log.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from .channels import Channel, ConsoleChannel, DeliveryResult, build_channel
from .templates import Template, pick_template, render


@dataclass
class AppointmentContext:
    appointment_id: str
    name: str
    phone: str
    appointment_time: str
    doctor: str = "your doctor"
    location: str = "the clinic"
    clinic: str = "your clinic"
    risk_band: str = "low"  # "low" | "medium" | "high"
    extra: Dict[str, str] = field(default_factory=dict)

    def as_context(self) -> Dict[str, str]:
        ctx = {
            "appointment_id": self.appointment_id,
            "name": self.name,
            "phone": self.phone,
            "appointment_time": self.appointment_time,
            "doctor": self.doctor,
            "location": self.location,
            "clinic": self.clinic,
        }
        ctx.update(self.extra)
        return ctx


@dataclass
class DispatchResult:
    appointment_id: str
    risk_band: str
    template_key: str
    rendered_body: str
    delivery: DeliveryResult


class Dispatcher:
    """Routes scored appointments to the right template + channel."""

    def __init__(self, default_channel: str = "sms", channels: Optional[Dict[str, Channel]] = None):
        self.default_channel_name = default_channel
        self.channels: Dict[str, Channel] = channels or {}
        self.log: List[DispatchResult] = []

    def _get_channel(self, name: str) -> Channel:
        if name not in self.channels:
            self.channels[name] = build_channel(name)
        return self.channels[name]

    def preview(self, appts: Iterable[AppointmentContext], channel: Optional[str] = None) -> List[DispatchResult]:
        """Render messages without sending — useful for dashboards/dry-runs."""
        out: List[DispatchResult] = []
        for a in appts:
            ch = channel or self.default_channel_name
            tpl = pick_template(a.risk_band, channel=ch)
            if tpl is None:
                continue
            body = render(tpl, a.as_context())
            out.append(DispatchResult(
                appointment_id=a.appointment_id,
                risk_band=a.risk_band,
                template_key=tpl.key,
                rendered_body=body,
                delivery=DeliveryResult(channel=ch, to=a.phone, status="dry_run", body=body),
            ))
        return out

    def send_batch(
        self,
        appts: Iterable[AppointmentContext],
        channel: Optional[str] = None,
        skip_low_risk: bool = False,
    ) -> List[DispatchResult]:
        results: List[DispatchResult] = []
        ch_name = channel or self.default_channel_name
        ch = self._get_channel(ch_name)
        for a in appts:
            if skip_low_risk and a.risk_band == "low":
                continue
            tpl = pick_template(a.risk_band, channel=ch_name)
            if tpl is None:
                continue
            body = render(tpl, a.as_context())
            delivery = ch.send(a.phone, body)
            r = DispatchResult(
                appointment_id=a.appointment_id,
                risk_band=a.risk_band,
                template_key=tpl.key,
                rendered_body=body,
                delivery=delivery,
            )
            results.append(r)
            self.log.append(r)
        return results
