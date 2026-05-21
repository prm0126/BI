"""Outbound message channels.

A `Channel` takes a rendered message + recipient and returns a `DeliveryResult`.
The Twilio channels are HTTP-only (no SDK dependency) so the package stays
light. Both SMS and WhatsApp go through Twilio's REST API — set these env vars:

    TWILIO_ACCOUNT_SID
    TWILIO_AUTH_TOKEN
    TWILIO_FROM_SMS          e.g. +14155551234
    TWILIO_FROM_WHATSAPP     e.g. whatsapp:+14155238886
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
import logging
import os
from typing import Optional

import httpx


logger = logging.getLogger(__name__)


@dataclass
class DeliveryResult:
    channel: str
    to: str
    status: str  # "queued" | "delivered" | "failed" | "dry_run"
    provider_id: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    body: str = ""


class Channel(ABC):
    name: str

    @abstractmethod
    def send(self, to: str, body: str) -> DeliveryResult: ...


class ConsoleChannel(Channel):
    """Dev/test channel — prints to stdout, never calls a network."""
    name = "console"

    def send(self, to: str, body: str) -> DeliveryResult:
        print(f"[console] → {to}\n{body}\n")
        return DeliveryResult(channel="console", to=to, status="dry_run", body=body)


class _TwilioBase(Channel):
    """Shared Twilio REST plumbing."""

    def __init__(self, from_var: str, channel_label: str, to_prefix: str = ""):
        self.account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
        self.auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
        self.from_number = os.environ.get(from_var)
        self.channel_label = channel_label
        self.to_prefix = to_prefix

    def _configured(self) -> Optional[str]:
        if not self.account_sid:
            return "TWILIO_ACCOUNT_SID not set"
        if not self.auth_token:
            return "TWILIO_AUTH_TOKEN not set"
        if not self.from_number:
            return "Twilio from-number env var not set"
        return None

    def send(self, to: str, body: str) -> DeliveryResult:
        missing = self._configured()
        if missing:
            logger.warning("Twilio %s not configured: %s — falling back to dry_run", self.channel_label, missing)
            print(f"[{self.channel_label}:dry_run] → {to}\n{body}\n")
            return DeliveryResult(channel=self.channel_label, to=to, status="dry_run", body=body, error=missing)

        url = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Messages.json"
        to_addr = f"{self.to_prefix}{to}" if self.to_prefix and not to.startswith(self.to_prefix) else to
        data = {"From": self.from_number, "To": to_addr, "Body": body}
        try:
            r = httpx.post(url, data=data, auth=(self.account_sid, self.auth_token), timeout=15.0)
            if r.status_code >= 400:
                return DeliveryResult(
                    channel=self.channel_label, to=to, status="failed",
                    error=f"HTTP {r.status_code}: {r.text[:200]}", body=body,
                )
            sid = r.json().get("sid")
            return DeliveryResult(channel=self.channel_label, to=to, status="queued", provider_id=sid, body=body)
        except Exception as e:
            return DeliveryResult(channel=self.channel_label, to=to, status="failed", error=str(e), body=body)


class TwilioSMSChannel(_TwilioBase):
    name = "sms"

    def __init__(self):
        super().__init__(from_var="TWILIO_FROM_SMS", channel_label="sms")


class TwilioWhatsAppChannel(_TwilioBase):
    name = "whatsapp"

    def __init__(self):
        super().__init__(from_var="TWILIO_FROM_WHATSAPP", channel_label="whatsapp", to_prefix="whatsapp:")


def build_channel(name: str) -> Channel:
    """Resolve a channel by name. Falls back to console if Twilio creds absent."""
    name = name.lower()
    if name == "console":
        return ConsoleChannel()
    if name == "sms":
        return TwilioSMSChannel()
    if name == "whatsapp":
        return TwilioWhatsAppChannel()
    raise ValueError(f"Unknown channel: {name}")
