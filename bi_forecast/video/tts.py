"""Voice-over (text-to-speech) for walkthrough videos.

Engines, in order of preference for ``engine="auto"``:

    piper   neural, offline. Needs a Piper ``.onnx`` voice: set PIPER_VOICE,
            drop one into ~/.local/share/bi-forecast/piper/, or run
            ``bi-forecast video voice-download`` (fetches en-us-lessac-medium).
    espeak  the ``espeak-ng`` command line tool (robotic but always available
            on most Linux images: ``apt-get install espeak-ng``).
    none    silent video (captions only).
"""

from __future__ import annotations

from dataclasses import dataclass
import glob
import io
import os
import shutil
import subprocess
import tarfile
import wave
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import httpx

ENGINES = ("auto", "piper", "espeak", "none")
DEFAULT_VOICE_NAME = "en-us-lessac-medium"
PIPER_VOICE_URL = "https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-{name}.tar.gz"
VOICE_DIR = Path(os.environ.get("BI_VOICE_DIR", str(Path.home() / ".local" / "share" / "bi-forecast" / "piper")))


class TTSError(RuntimeError):
    pass


@dataclass
class Narration:
    path: Path
    seconds: float
    engine: str
    text: str


# ------------------------------------------------------------------ discovery

def find_piper_voice(voice: Optional[str] = None) -> Optional[Path]:
    cands: List[str] = []
    if voice:
        cands.append(voice)
    env = os.environ.get("PIPER_VOICE")
    if env:
        cands.append(env)
    for d in (VOICE_DIR, Path("voices"), Path.cwd()):
        cands += sorted(glob.glob(str(d / "**" / "*.onnx"), recursive=True))
    for c in cands:
        p = Path(c)
        if p.is_file() and p.suffix == ".onnx":
            return p
    return None


def piper_available(voice: Optional[str] = None) -> bool:
    try:
        import piper  # noqa: F401
    except ImportError:
        return False
    return find_piper_voice(voice) is not None


def espeak_binary() -> Optional[str]:
    return shutil.which("espeak-ng") or shutil.which("espeak")


def available_engines(voice: Optional[str] = None) -> List[str]:
    out = []
    if piper_available(voice):
        out.append("piper")
    if espeak_binary():
        out.append("espeak")
    out.append("none")
    return out


def choose_engine(engine: str = "auto", voice: Optional[str] = None) -> str:
    if engine not in ENGINES:
        raise TTSError(f"engine must be one of {ENGINES}")
    avail = available_engines(voice)
    if engine == "auto":
        return avail[0]
    if engine not in avail:
        hint = {
            "piper": "pip install piper-tts, then `bi-forecast video voice-download` or set PIPER_VOICE",
            "espeak": "apt-get install espeak-ng",
        }.get(engine, "")
        raise TTSError(f"TTS engine '{engine}' is not available. {hint}".strip())
    return engine


def download_piper_voice(name: str = DEFAULT_VOICE_NAME, dest_dir: Optional[os.PathLike] = None,
                         timeout: float = 600.0) -> Path:
    """Fetch a Piper voice tarball from the Piper GitHub release and unpack the .onnx + .json."""
    dest = Path(dest_dir) if dest_dir else VOICE_DIR
    dest.mkdir(parents=True, exist_ok=True)
    existing = dest / f"{name}.onnx"
    if existing.exists():
        return existing
    url = PIPER_VOICE_URL.format(name=name)
    buf = io.BytesIO()
    with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as r:
        if r.status_code >= 400:
            raise TTSError(f"voice download failed: HTTP {r.status_code} for {url}")
        for chunk in r.iter_bytes():
            buf.write(chunk)
    buf.seek(0)
    onnx: Optional[Path] = None
    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        for m in tar.getmembers():
            if not m.isfile() or not (m.name.endswith(".onnx") or m.name.endswith(".onnx.json")):
                continue
            target = dest / Path(m.name).name
            with tar.extractfile(m) as src, open(target, "wb") as dst:  # type: ignore[arg-type]
                shutil.copyfileobj(src, dst)
            if target.suffix == ".onnx":
                onnx = target
    if onnx is None:
        raise TTSError(f"no .onnx voice found in {url}")
    return onnx


# ------------------------------------------------------------------ synthesis

_PIPER_CACHE: Dict[str, object] = {}


def _piper_voice(path: Path):
    key = str(path)
    if key not in _PIPER_CACHE:
        from piper import PiperVoice
        _PIPER_CACHE[key] = PiperVoice.load(key)
    return _PIPER_CACHE[key]


def wav_seconds(path: os.PathLike) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def synthesize(text: str, out_wav: os.PathLike, engine: str = "auto", voice: Optional[str] = None,
               rate: float = 1.0) -> Narration:
    """Render ``text`` to a mono 16-bit WAV. ``rate`` > 1 speaks faster."""
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    chosen = choose_engine(engine, voice)
    text = " ".join(text.split())

    if chosen == "none":
        with wave.open(str(out_wav), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(22050)
            w.writeframes(b"")
        return Narration(out_wav, 0.0, "none", text)

    if chosen == "piper":
        model = find_piper_voice(voice)
        assert model is not None
        v = _piper_voice(model)
        syn = None
        try:
            from piper import SynthesisConfig
            syn = SynthesisConfig(length_scale=1.0 / max(rate, 0.1))
        except Exception:
            pass
        with wave.open(str(out_wav), "wb") as w:
            if syn is not None:
                v.synthesize_wav(text, w, syn_config=syn)
            else:
                v.synthesize_wav(text, w)
        return Narration(out_wav, wav_seconds(out_wav), "piper", text)

    exe = espeak_binary()
    assert exe is not None
    cmd = [exe, "-v", "en-us", "-s", str(int(165 * rate)), "-w", str(out_wav), text]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not out_wav.exists():
        raise TTSError(f"espeak failed: {res.stderr.strip()[:300]}")
    return Narration(out_wav, wav_seconds(out_wav), "espeak", text)


# ------------------------------------------------------------------ track assembly

def _read_mono_pcm16(path: os.PathLike) -> Tuple["np.ndarray", int]:  # noqa: F821
    import numpy as np
    with wave.open(str(path), "rb") as w:
        rate, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if sw != 2:
        raise TTSError(f"{path}: only 16-bit PCM WAV is supported (got {sw * 8}-bit)")
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data, rate


def _resample(data, src_rate: int, dst_rate: int):
    import numpy as np
    if src_rate == dst_rate or data.size == 0:
        return data
    n_dst = int(round(data.size * dst_rate / src_rate))
    x_src = np.linspace(0.0, 1.0, num=data.size, endpoint=False)
    x_dst = np.linspace(0.0, 1.0, num=n_dst, endpoint=False)
    return np.interp(x_dst, x_src, data).astype(np.float32)


def build_track(segments: Sequence[Tuple[Optional[os.PathLike], float]], out_wav: os.PathLike,
                sample_rate: int = 22050, lead_in: float = 0.35, gain: float = 0.9) -> Path:
    """Lay narration clips onto a silent timeline.

    ``segments`` is a list of (wav_path_or_None, slot_seconds). Each clip starts
    ``lead_in`` seconds into its slot; the slot is padded with silence (or the
    clip is faded out at the slot end if it is longer).
    """
    import numpy as np
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    parts = []
    for path, slot in segments:
        n_slot = max(1, int(round(slot * sample_rate)))
        buf = np.zeros(n_slot, dtype=np.float32)
        if path and Path(path).exists():
            data, rate = _read_mono_pcm16(path)
            data = _resample(data, rate, sample_rate) * gain
            start = min(int(lead_in * sample_rate), n_slot - 1)
            n = min(data.size, n_slot - start)
            buf[start:start + n] = data[:n]
            if data.size > n and n > 0:  # clip longer than the slot: fade the tail
                fade = min(n, int(0.25 * sample_rate))
                buf[start + n - fade:start + n] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        parts.append(buf)
    track = np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)
    pcm = np.clip(track, -32768, 32767).astype("<i2").tobytes()
    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sample_rate)
        w.writeframes(pcm)
    return out_wav
