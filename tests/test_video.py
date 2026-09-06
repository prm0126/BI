"""Tests for the video package: storyboard, Higgsfield client (mocked), local render, API."""

import json
import os
from pathlib import Path

import httpx
import pytest

from bi_forecast.video import (
    Scene, Storyboard, aircollab_storyboard, HiggsfieldClient, HiggsfieldError,
    HiggsfieldNotConfigured, choose_provider, generate_walkthrough,
)
from bi_forecast.video import higgsfield as hf
from bi_forecast.video.render import find_ffmpeg, render_storyboard, concat_clips, render_scene

LANDING = Path(__file__).resolve().parents[1] / "examples/aircollab/screenshots/01_landing.png"
needs_ffmpeg = pytest.mark.skipif(find_ffmpeg() is None, reason="ffmpeg not available")


@pytest.fixture(autouse=True)
def _no_credentials(monkeypatch):
    for var in ("HIGGSFIELD_API_KEY", "HIGGSFIELD_API_SECRET", "HIGGSFIELD_KEY", "HF_KEY", "HF_API_KEY", "HF_API_SECRET"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def tiny_storyboard():
    return Storyboard(
        name="tiny", product="AIRCollab", url="https://app.airnd.ai", style="clean",
        scenes=[
            Scene(key="a", title="Welcome to AIRCollab", card="Welcome to AIRCollab", narration="Hello.",
                  motion="push in", image=str(LANDING), duration=1, start=(0.5, 0.3, 1.0), end=(0.2, 0.2, 0.5)),
            Scene(key="b", title="Please login first", narration="Login.", motion="zoom",
                  image=str(LANDING), duration=1),
        ],
    )


# ---------------------------------------------------------------- storyboard

def test_default_storyboard_covers_every_window():
    sb = aircollab_storyboard()
    assert sb.missing_images() == []
    prompt = sb.walkthrough_prompt()
    assert "Welcome to AIRCollab" in prompt
    assert "Please login first" in prompt
    for window in ("Landing", "login", "Dashboard", "Discover", "Saved", "Data Center", "chat",
                   "Shipment", "quotation", "Equipment"):
        assert window.lower() in prompt.lower(), window
    assert sb.total_duration == sum(s.duration for s in sb.scenes)
    assert len({s.key for s in sb.scenes}) == len(sb.scenes)


def test_storyboard_json_roundtrip(tmp_path):
    sb = aircollab_storyboard()
    path = tmp_path / "sb.json"
    sb.to_json(path)
    back = Storyboard.from_json(path)
    assert back.to_dict() == sb.to_dict()
    assert isinstance(back.scenes[0].start, tuple)


def test_scene_prompt_mentions_card_motion_and_narration():
    s = aircollab_storyboard().scenes[0]
    p = s.prompt("Style: x")
    assert 'title "Welcome to AIRCollab"' in p and s.motion in p and "Voice-over" in p and "Style: x" in p


# ---------------------------------------------------------------- credentials

def test_not_configured_without_env():
    assert hf.is_configured() is False
    with pytest.raises(HiggsfieldNotConfigured):
        HiggsfieldClient()
    assert choose_provider("auto") == "local"


def test_configured_via_env(monkeypatch):
    monkeypatch.setenv("HIGGSFIELD_API_KEY", "k")
    monkeypatch.setenv("HIGGSFIELD_API_SECRET", "s")
    assert hf.resolve_credentials() == "k:s"
    assert choose_provider("auto") == "higgsfield"
    monkeypatch.delenv("HIGGSFIELD_API_KEY"); monkeypatch.delenv("HIGGSFIELD_API_SECRET")
    monkeypatch.setenv("HF_KEY", "a:b")
    assert hf.resolve_credentials() == "a:b"


def test_resolve_model():
    assert hf.resolve_model(None) == "higgsfield-ai/dop/standard"
    assert hf.resolve_model("kling") == "kling-video/v2.1/pro/image-to-video"
    assert hf.resolve_model("vendor/custom/path") == "vendor/custom/path"
    with pytest.raises(HiggsfieldError):
        hf.resolve_model("nope")


# ---------------------------------------------------------------- client (mock transport)

class FakeHiggsfield:
    """Stateful stand-in for platform.higgsfield.ai + the upload bucket."""

    def __init__(self, final_status="completed", polls_before_done=2):
        self.final_status = final_status
        self.polls_before_done = polls_before_done
        self.calls = []
        self.uploaded = None
        self.submitted = None
        self.polls = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append((request.method, request.url.path))
        path = request.url.path
        if path == "/files/generate-upload-url":
            assert request.headers["Authorization"] == "Key k:s"
            return httpx.Response(200, json={"public_url": "https://cdn.test/img.png",
                                             "upload_url": "https://bucket.test/put"})
        if request.url.host == "bucket.test":
            self.uploaded = (request.headers["Content-Type"], request.content)
            return httpx.Response(200)
        if path == "/higgsfield-ai/dop/standard":
            self.submitted = json.loads(request.content)
            return httpx.Response(200, json={
                "status": "queued", "request_id": "req-1",
                "status_url": "https://platform.higgsfield.ai/requests/req-1/status",
                "cancel_url": "https://platform.higgsfield.ai/requests/req-1/cancel",
            })
        if path == "/requests/req-1/status":
            self.polls += 1
            if self.polls < self.polls_before_done:
                return httpx.Response(200, json={"status": "in_progress", "request_id": "req-1"})
            body = {"status": self.final_status, "request_id": "req-1"}
            if self.final_status == "completed":
                body["video"] = {"url": "https://video.test/out.mp4"}
            return httpx.Response(200, json=body)
        if request.url.host == "video.test":
            return httpx.Response(200, content=b"MP4DATA", headers={"Content-Type": "video/mp4"})
        return httpx.Response(404, text=f"unexpected {path}")


def _client(fake):
    return HiggsfieldClient(api_key="k", api_secret="s", transport=httpx.MockTransport(fake.handler))


def test_image_to_video_happy_path(tmp_path):
    fake = FakeHiggsfield()
    out = _client(fake).image_to_video(LANDING, "pan across the hero", tmp_path / "clip.mp4",
                                       duration=5, poll_interval=0)
    assert out.read_bytes() == b"MP4DATA"
    assert fake.uploaded[0] == "image/png" and len(fake.uploaded[1]) == LANDING.stat().st_size
    assert fake.submitted == {"image_url": "https://cdn.test/img.png", "prompt": "pan across the hero", "duration": 5}
    assert fake.polls == 2
    assert ("POST", "/files/generate-upload-url") in fake.calls


def test_failed_job_raises(tmp_path):
    fake = FakeHiggsfield(final_status="failed")
    with pytest.raises(HiggsfieldError, match="failed"):
        _client(fake).image_to_video(LANDING, "x", tmp_path / "c.mp4", poll_interval=0)


def test_http_error_surfaces():
    def handler(request):
        return httpx.Response(401, text="bad key")
    c = HiggsfieldClient(api_key="k:s", transport=httpx.MockTransport(handler))
    with pytest.raises(HiggsfieldError, match="401"):
        c.upload_bytes(b"x", "image/png")


def test_submit_with_webhook_and_extra():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"status": "queued", "request_id": "r", "status_url": "u", "cancel_url": "c"})

    c = HiggsfieldClient(api_key="k:s", transport=httpx.MockTransport(handler))
    job = c.submit_image_to_video("https://x/img.png", "p", model="kling", duration=None,
                                  extra={"aspect_ratio": "16:9"}, webhook_url="https://hook.test/cb")
    assert job.request_id == "r" and not job.done
    assert seen["url"].startswith("https://platform.higgsfield.ai/kling-video/v2.1/pro/image-to-video?hf_webhook=")
    assert seen["body"] == {"image_url": "https://x/img.png", "prompt": "p", "aspect_ratio": "16:9"}


# ---------------------------------------------------------------- local render

@needs_ffmpeg
def test_local_render_and_concat(tmp_path, tiny_storyboard):
    out = render_storyboard(tiny_storyboard, tmp_path / "tiny.mp4", size=(320, 180), fps=8)
    assert out.exists() and out.stat().st_size > 1000
    clip = render_scene(tiny_storyboard.scenes[1], tmp_path / "one.mp4", size=(256, 144), fps=8)
    stitched = concat_clips([out, clip], tmp_path / "all.mp4", size=(320, 180), fps=8)
    assert stitched.exists() and stitched.stat().st_size > out.stat().st_size


@needs_ffmpeg
def test_generate_walkthrough_local_provider(tmp_path, tiny_storyboard):
    msgs = []
    r = generate_walkthrough(tiny_storyboard, tmp_path / "w.mp4", provider="local",
                             size=(320, 180), fps=8, on_progress=msgs.append)
    assert r.provider == "local" and r.path.exists()
    assert any("Welcome to AIRCollab" in m for m in msgs)
    assert "Please login first" in r.prompt


@needs_ffmpeg
def test_generate_walkthrough_higgsfield_provider(tmp_path, tiny_storyboard):
    """Higgsfield provider: clips come back from the (mocked) API and get stitched."""
    # The mocked API returns bytes that are not a real video, so pre-seed the
    # clip cache with real renders and verify the pipeline reuses + stitches them.
    workdir = tmy = tmp_path / "clips"
    workdir.mkdir()
    for i, s in enumerate(tiny_storyboard.scenes, 1):
        render_scene(s, workdir / f"{i:02d}_{s.key}.mp4", size=(256, 144), fps=8)
    fake = FakeHiggsfield()
    r = generate_walkthrough(tiny_storyboard, tmp_path / "hf.mp4", provider="higgsfield", workdir=workdir,
                             size=(320, 180), fps=8, client=_client(fake), poll_interval=0)
    assert r.provider == "higgsfield" and r.path.exists() and len(r.clips) == 2
    assert fake.submitted is None  # cache hit: nothing was sent to the API


# ---------------------------------------------------------------- API

@needs_ffmpeg
def test_video_api_roundtrip(tmp_path, tiny_storyboard):
    import time
    from fastapi.testclient import TestClient
    from bi_forecast.api.app import create_app

    app = create_app(database_url=f"sqlite:///{tmp_path / 'api.db'}")
    client = TestClient(app)

    p = client.get("/video/prompt").json()
    assert p["higgsfield_configured"] is False and "Welcome to AIRCollab" in p["prompt"]
    assert len(client.get("/video/storyboard").json()["scenes"]) == 14

    r = client.post("/video/generate", json={
        "provider": "local", "storyboard": tiny_storyboard.to_dict(),
        "out_dir": str(tmp_path / "videos"), "width": 320, "height": 180, "fps": 8,
    })
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    for _ in range(600):
        job = client.get(f"/video/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed"):
            break
        time.sleep(0.1)
    assert job["status"] == "completed", job
    dl = client.get(f"/video/jobs/{job_id}/download")
    assert dl.status_code == 200 and dl.headers["content-type"] == "video/mp4" and len(dl.content) > 1000
    assert client.get("/video/jobs/nope").status_code == 404
    assert client.post("/video/generate", json={"provider": "bogus"}).status_code == 400


# ---------------------------------------------------------------- narration (TTS)

from bi_forecast.video import tts as tts_mod  # noqa: E402

needs_tts = pytest.mark.skipif(tts_mod.available_engines() == ["none"], reason="no TTS engine (piper/espeak) available")


def test_choose_engine_none_and_invalid():
    assert tts_mod.choose_engine("none") == "none"
    with pytest.raises(tts_mod.TTSError):
        tts_mod.choose_engine("bogus")


def test_synthesize_none_writes_empty_wav(tmp_path):
    n = tts_mod.synthesize("hello", tmp_path / "n.wav", engine="none")
    assert n.engine == "none" and n.seconds == 0.0 and n.path.exists()


def test_build_track_pads_each_slot(tmp_path):
    import numpy as np
    import wave
    # A 0.5s tone at 8 kHz placed in a 2s slot, followed by a silent 1s slot.
    rate = 8000
    tone = (np.sin(np.linspace(0, 2 * np.pi * 440 * 0.5, rate // 2)) * 10000).astype("<i2")
    src = tmp_path / "tone.wav"
    with wave.open(str(src), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(tone.tobytes())
    out = tts_mod.build_track([(src, 2.0), (None, 1.0)], tmp_path / "track.wav", sample_rate=22050)
    assert abs(tts_mod.wav_seconds(out) - 3.0) < 0.01
    data, r = tts_mod._read_mono_pcm16(out)
    assert r == 22050
    assert np.abs(data[int(0.4 * r):int(0.8 * r)]).max() > 1000   # tone present after lead-in
    assert np.abs(data[int(2.0 * r):]).max() == 0                  # second slot silent


@needs_tts
def test_synthesize_speech_has_duration(tmp_path):
    n = tts_mod.synthesize("Welcome to AIRCollab. Please login first.", tmp_path / "v.wav")
    assert n.engine in ("piper", "espeak") and 1.0 < n.seconds < 10.0


@needs_ffmpeg
@needs_tts
def test_render_with_narration_has_audio_and_stretches(tmp_path, tiny_storyboard):
    from bi_forecast.video.render import media_duration
    from bi_forecast.video import higgsfield as _hf  # noqa: F401
    import subprocess
    tiny_storyboard.scenes[0].narration = (
        "Welcome to AIRCollab, the platform that connects researchers worldwide for real time "
        "communication, document sharing, virtual meetings and project management."
    )
    out = render_storyboard(tiny_storyboard, tmp_path / "voiced.mp4", size=(320, 180), fps=8,
                            narration="auto", workdir=tmp_path / "work")
    banner = subprocess.run([find_ffmpeg(), "-hide_banner", "-i", str(out)], capture_output=True, text=True).stderr
    assert "Audio:" in banner
    assert media_duration(out) > 2.0 + 3.0   # storyboard floor was 2s; voice-over stretched scene 1
    assert (tmp_path / "work" / "narration_track.wav").exists()


@needs_ffmpeg
@needs_tts
def test_concat_pads_clips_to_narration(tmp_path, tiny_storyboard):
    from bi_forecast.video.render import media_duration, plan_narration
    clips = [render_scene(s, tmp_path / f"{i}.mp4", size=(256, 144), fps=8) for i, s in enumerate(tiny_storyboard.scenes)]
    durations, track, engine = plan_narration(tiny_storyboard.scenes, tmp_path / "w", engine="auto",
                                              min_durations=[media_duration(c) for c in clips])
    assert engine != "none" and track is not None and all(d >= 1.0 for d in durations)
    out = concat_clips(clips, tmp_path / "all.mp4", size=(320, 180), fps=8, min_durations=durations, audio=track)
    assert abs(media_duration(out) - sum(durations)) < 0.6


@needs_ffmpeg
def test_generate_walkthrough_silent_when_requested(tmp_path, tiny_storyboard):
    r = generate_walkthrough(tiny_storyboard, tmp_path / "s.mp4", provider="local", size=(320, 180), fps=8,
                             narration="none")
    assert r.narration == "none" and r.path.exists()
