"""Higgsfield AI video client (HTTP only, no SDK dependency).

Wraps the Higgsfield platform API used for image-to-video generation:

    POST https://platform.higgsfield.ai/<model-path>            submit a job
    GET  https://platform.higgsfield.ai/requests/{id}/status    poll a job
    POST https://platform.higgsfield.ai/files/generate-upload-url  get a signed
         upload URL so a local screenshot can be referenced by ``image_url``

Credentials (either style works; the ``HF_*`` names match the official SDK):

    HIGGSFIELD_API_KEY / HIGGSFIELD_API_SECRET
    HF_API_KEY         / HF_API_SECRET
    HF_KEY                              "key:secret" in one variable
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://platform.higgsfield.ai"
USER_AGENT = "bi-forecast-video/0.1"

# Short aliases for documented image-to-video model paths. Any value containing
# a "/" is treated as a raw model path and sent unchanged.
MODELS: Dict[str, str] = {
    "dop": "higgsfield-ai/dop/standard",
    "kling": "kling-video/v2.1/pro/image-to-video",
    "kling-standard": "kling-video/v2.1/standard/image-to-video",
}
DEFAULT_MODEL = "dop"

TERMINAL_STATUSES = {"completed", "failed", "nsfw", "canceled", "cancelled"}


class HiggsfieldError(RuntimeError):
    """Raised for credential, HTTP, or generation failures."""


class HiggsfieldNotConfigured(HiggsfieldError):
    """Raised when no API credentials are available."""


def resolve_credentials(
    api_key: Optional[str] = None, api_secret: Optional[str] = None
) -> Optional[str]:
    """Return the ``key:secret`` token Higgsfield expects, or None if unset."""
    if api_key and api_secret:
        return f"{api_key}:{api_secret}"
    if api_key and ":" in api_key:
        return api_key

    env = os.environ
    key = env.get("HIGGSFIELD_API_KEY") or env.get("HF_API_KEY")
    secret = env.get("HIGGSFIELD_API_SECRET") or env.get("HF_API_SECRET")
    if key and secret:
        return f"{key}:{secret}"
    combined = env.get("HIGGSFIELD_KEY") or env.get("HF_KEY")
    if combined and ":" in combined:
        return combined
    return None


def is_configured() -> bool:
    return resolve_credentials() is not None


def resolve_model(model: Optional[str]) -> str:
    if not model:
        model = DEFAULT_MODEL
    if "/" in model:
        return model.strip("/")
    try:
        return MODELS[model]
    except KeyError:
        raise HiggsfieldError(
            f"Unknown Higgsfield model alias '{model}'. Use one of {sorted(MODELS)} "
            "or pass a full model path such as 'higgsfield-ai/dop/standard'."
        )


@dataclass
class HiggsfieldJob:
    request_id: str
    status: str
    status_url: str
    cancel_url: Optional[str] = None
    model: Optional[str] = None
    video_url: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def ok(self) -> bool:
        return self.status == "completed"


class HiggsfieldClient:
    """Minimal synchronous client for Higgsfield image-to-video jobs."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        base_url: str = BASE_URL,
        timeout: float = 90.0,
        transport: Optional[httpx.BaseTransport] = None,
    ):
        token = resolve_credentials(api_key, api_secret)
        if token is None:
            raise HiggsfieldNotConfigured(
                "Higgsfield credentials missing. Set HIGGSFIELD_API_KEY and "
                "HIGGSFIELD_API_SECRET (or HF_KEY='key:secret')."
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Key {token}",
                "Accept": "application/json",
                "User-Agent": USER_AGENT,
            },
            timeout=timeout,
            transport=transport,
        )
        # Uploads go to a pre-signed URL on a different host: no auth header.
        self._raw = httpx.Client(timeout=timeout, transport=transport, follow_redirects=True)

    # ---------------------------------------------------------------- helpers

    def _check(self, r: httpx.Response, what: str) -> Dict[str, Any]:
        if r.status_code >= 400:
            raise HiggsfieldError(f"{what} failed: HTTP {r.status_code}: {r.text[:300]}")
        try:
            return r.json()
        except ValueError:
            raise HiggsfieldError(f"{what} returned non-JSON body: {r.text[:200]}")

    @staticmethod
    def _job_from_payload(data: Dict[str, Any], model: Optional[str] = None) -> HiggsfieldJob:
        video = data.get("video") or {}
        return HiggsfieldJob(
            request_id=data["request_id"],
            status=str(data.get("status", "queued")).lower(),
            status_url=data.get("status_url") or f"{BASE_URL}/requests/{data['request_id']}/status",
            cancel_url=data.get("cancel_url"),
            model=model,
            video_url=video.get("url") if isinstance(video, dict) else None,
            raw=data,
        )

    # ---------------------------------------------------------------- uploads

    def upload_bytes(self, data: bytes, content_type: str) -> str:
        """Upload raw bytes; return the public URL the API can read from."""
        meta = self._check(
            self._client.post("/files/generate-upload-url", json={"content_type": content_type}),
            "generate-upload-url",
        )
        put = self._raw.put(meta["upload_url"], content=data, headers={"Content-Type": content_type})
        if put.status_code >= 400:
            raise HiggsfieldError(f"upload PUT failed: HTTP {put.status_code}: {put.text[:200]}")
        return meta["public_url"]

    def upload_file(self, path: os.PathLike) -> str:
        p = Path(path)
        mime = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
        return self.upload_bytes(p.read_bytes(), mime)

    # ------------------------------------------------------------ generation

    def submit_image_to_video(
        self,
        image_url: str,
        prompt: str,
        model: Optional[str] = None,
        duration: Optional[int] = 5,
        extra: Optional[Dict[str, Any]] = None,
        webhook_url: Optional[str] = None,
    ) -> HiggsfieldJob:
        path = resolve_model(model)
        body: Dict[str, Any] = {"image_url": image_url, "prompt": prompt}
        if duration is not None:
            body["duration"] = int(duration)
        if extra:
            body.update(extra)
        params = {"hf_webhook": webhook_url} if webhook_url else None
        data = self._check(self._client.post(f"/{path}", json=body, params=params), f"submit {path}")
        job = self._job_from_payload(data, model=path)
        logger.info("Higgsfield job %s queued on %s", job.request_id, path)
        return job

    def get_status(self, job_or_id) -> HiggsfieldJob:
        if isinstance(job_or_id, HiggsfieldJob):
            url, model = job_or_id.status_url, job_or_id.model
        else:
            url, model = f"{self.base_url}/requests/{job_or_id}/status", None
        data = self._check(self._client.get(url), "status")
        data.setdefault("request_id", getattr(job_or_id, "request_id", job_or_id))
        return self._job_from_payload(data, model=model)

    def cancel(self, job: HiggsfieldJob) -> None:
        if job.cancel_url:
            self._client.post(job.cancel_url)

    def wait(
        self,
        job: HiggsfieldJob,
        poll_interval: float = 3.0,
        timeout: float = 900.0,
        on_update: Optional[Callable[[HiggsfieldJob], None]] = None,
    ) -> HiggsfieldJob:
        deadline = time.monotonic() + timeout
        current = job
        while not current.done:
            if time.monotonic() > deadline:
                raise HiggsfieldError(f"job {job.request_id} timed out after {timeout:.0f}s")
            time.sleep(poll_interval)
            current = self.get_status(current)
            if on_update:
                on_update(current)
        if not current.ok:
            raise HiggsfieldError(f"job {job.request_id} ended with status '{current.status}'")
        if not current.video_url:
            raise HiggsfieldError(f"job {job.request_id} completed without a video URL: {current.raw}")
        return current

    def download(self, url: str, dest: os.PathLike) -> Path:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with self._raw.stream("GET", url) as r:
            if r.status_code >= 400:
                raise HiggsfieldError(f"download failed: HTTP {r.status_code}")
            with open(dest, "wb") as fh:
                for chunk in r.iter_bytes():
                    fh.write(chunk)
        return dest

    def image_to_video(
        self,
        image_path: os.PathLike,
        prompt: str,
        out_path: os.PathLike,
        model: Optional[str] = None,
        duration: Optional[int] = 5,
        extra: Optional[Dict[str, Any]] = None,
        poll_interval: float = 3.0,
        timeout: float = 900.0,
        on_update: Optional[Callable[[HiggsfieldJob], None]] = None,
    ) -> Path:
        """Upload a local image, animate it, and save the resulting clip."""
        image_url = self.upload_file(image_path)
        job = self.submit_image_to_video(image_url, prompt, model=model, duration=duration, extra=extra)
        job = self.wait(job, poll_interval=poll_interval, timeout=timeout, on_update=on_update)
        return self.download(job.video_url, out_path)

    def close(self) -> None:
        self._client.close()
        self._raw.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
