"""Message templates for patient engagement."""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class Template:
    key: str
    channel: str  # "sms" | "whatsapp" | "email"
    body: str  # supports {name}, {appointment_time}, {doctor}, {location}, {clinic}, {confirm_url}
    language: str = "en"


DEFAULT_TEMPLATES: Dict[str, Template] = {
    # High-risk: 24h reminder + confirmation ask.
    "noshow_high_24h_sms": Template(
        key="noshow_high_24h_sms",
        channel="sms",
        body=(
            "Hi {name}, this is {clinic}. You have an appointment with "
            "{doctor} on {appointment_time} at {location}. "
            "Reply YES to confirm or call us to reschedule. — Reply STOP to opt out."
        ),
    ),
    "noshow_high_24h_whatsapp": Template(
        key="noshow_high_24h_whatsapp",
        channel="whatsapp",
        body=(
            "Hello {name} 👋\n"
            "This is *{clinic}*. Reminder of your appointment:\n"
            "📅 {appointment_time}\n"
            "👨‍⚕️ {doctor}\n"
            "📍 {location}\n\n"
            "Please reply *YES* to confirm or *NO* to reschedule."
        ),
    ),
    # Medium-risk: standard reminder, no confirmation pressure.
    "noshow_medium_24h_sms": Template(
        key="noshow_medium_24h_sms",
        channel="sms",
        body=(
            "Reminder: {name}, your appointment with {doctor} is on "
            "{appointment_time} at {location}. — {clinic}"
        ),
    ),
    # Low-risk: light-touch reminder.
    "noshow_low_24h_sms": Template(
        key="noshow_low_24h_sms",
        channel="sms",
        body="See you {appointment_time} with {doctor}. — {clinic}",
    ),
    # Follow-up after a missed appointment.
    "post_no_show_followup_sms": Template(
        key="post_no_show_followup_sms",
        channel="sms",
        body=(
            "Hi {name}, we missed you at your appointment on {appointment_time}. "
            "Reply 1 to reschedule or call {clinic} to talk to us."
        ),
    ),
}


def render(template: Template, context: Dict[str, str]) -> str:
    """Render a template with safe substitution — missing keys become '?'."""
    safe = {k: context.get(k, "?") for k in _placeholders(template.body)}
    return template.body.format_map(_DefaultDict(safe))


def pick_template(risk_band: str, channel: str = "sms") -> Optional[Template]:
    key = {
        "high": f"noshow_high_24h_{channel}",
        "medium": f"noshow_medium_24h_{channel}",
        "low": f"noshow_low_24h_{channel}",
    }.get(risk_band)
    if key in DEFAULT_TEMPLATES:
        return DEFAULT_TEMPLATES[key]
    # Fall back to SMS template if channel-specific doesn't exist.
    return DEFAULT_TEMPLATES.get(f"noshow_{risk_band}_24h_sms")


def _placeholders(body: str) -> set:
    import re
    return set(re.findall(r"{(\w+)}", body))


class _DefaultDict(dict):
    def __missing__(self, key: str) -> str:
        return "?"
