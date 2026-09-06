"""Local video rendering with Pillow + ffmpeg (no AI, no network).

Used for two things:

* ``render_storyboard`` — a Ken Burns style walkthrough built straight from the
  screenshots, with caption bands and title cards. This is the fallback when
  Higgsfield credentials are absent, and a fast preview of a storyboard.
* ``concat_clips`` — stitch per-scene clips (e.g. Higgsfield outputs) into one
  MP4, normalising resolution and frame rate.

ffmpeg discovery order: ``FFMPEG_BIN`` env var, ``ffmpeg`` on PATH, the
``imageio-ffmpeg`` wheel, then a Playwright-bundled build.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from .storyboard import Scene, Storyboard

Size = Tuple[int, int]

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "C:/Windows/Fonts/arialbd.ttf",
]


class RenderError(RuntimeError):
    pass


def find_ffmpeg() -> Optional[str]:
    env = os.environ.get("FFMPEG_BIN")
    if env and Path(env).exists():
        return env
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    try:
        import imageio_ffmpeg  # type: ignore
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    pw = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", str(Path.home() / ".cache" / "ms-playwright"))
    for cand in sorted(glob.glob(os.path.join(pw, "ffmpeg-*", "ffmpeg-*")), reverse=True):
        if os.access(cand, os.X_OK):
            return cand
    return None


def _require_ffmpeg() -> str:
    exe = find_ffmpeg()
    if not exe:
        raise RenderError(
            "ffmpeg not found. Install it, set FFMPEG_BIN, or `pip install imageio-ffmpeg`."
        )
    return exe


def _has_encoder(exe: str, name: str) -> bool:
    try:
        out = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return False
    return f" {name} " in out


def _video_codec_args(exe: str, out_path: Path) -> List[str]:
    if out_path.suffix.lower() == ".webm" or not _has_encoder(exe, "libx264"):
        return ["-c:v", "libvpx", "-b:v", "2M", "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]


def _load_font(size: int):
    from PIL import ImageFont
    for cand in _FONT_CANDIDATES:
        if Path(cand).exists():
            try:
                return ImageFont.truetype(cand, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _ease(t: float) -> float:
    return t * t * (3 - 2 * t)  # smoothstep


def _viewport(img_size: Size, key: Sequence[float], out_size: Size) -> Tuple[float, float, float, float]:
    """Translate a (cx, cy, width-fraction) keyframe to a crop box in pixels."""
    W, H = img_size
    ow, oh = out_size
    aspect = ow / oh
    vw = max(0.05, min(1.0, key[2])) * W
    vh = vw / aspect
    if vh > H:
        vh = H
        vw = vh * aspect
    if vw > W:
        vw = W
        vh = vw / aspect
    left = min(max(key[0] * W - vw / 2, 0), W - vw)
    top = min(max(key[1] * H - vh / 2, 0), H - vh)
    return left, top, left + vw, top + vh


def _wrap(draw, text: str, font, max_width: int) -> List[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if draw.textlength(trial, font=font) <= max_width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _caption_overlay(size: Size, scene: Scene, index: int, total: int):
    """Pre-rendered RGBA caption band for a scene (composited onto every frame)."""
    from PIL import Image, ImageDraw
    ow, oh = size
    band_h = int(oh * 0.2)
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle([0, oh - band_h, ow, oh], fill=(12, 18, 32, 200))
    title_font = _load_font(int(oh * 0.045))
    body_font = _load_font(int(oh * 0.03))
    pad = int(ow * 0.035)
    y = oh - band_h + int(band_h * 0.14)
    draw.text((pad, y), scene.title, font=title_font, fill=(255, 255, 255, 255))
    chip = f"{index}/{total}"
    cw = draw.textlength(chip, font=body_font)
    draw.text((ow - pad - cw, y + int(oh * 0.008)), chip, font=body_font, fill=(150, 170, 210, 255))
    y += int(oh * 0.065)
    for line in _wrap(draw, scene.narration, body_font, ow - 2 * pad)[:3]:
        draw.text((pad, y), line, font=body_font, fill=(220, 226, 240, 255))
        y += int(oh * 0.04)
    return overlay


def _card_overlay(size: Size, text: str):
    from PIL import Image, ImageDraw
    ow, oh = size
    overlay = Image.new("RGBA", size, (10, 20, 40, 150))
    draw = ImageDraw.Draw(overlay)
    font = _load_font(int(oh * 0.11))
    tw = draw.textlength(text, font=font)
    draw.text(((ow - tw) / 2, oh * 0.38), text, font=font, fill=(255, 255, 255, 255))
    return overlay


def iter_scene_frames(
    scene: Scene, size: Size, fps: int, index: int = 1, total: int = 1, captions: bool = True
) -> Iterable[bytes]:
    """Yield raw RGB frames for one scene (Ken Burns + captions + fades)."""
    from PIL import Image

    img_path = scene.image_path()
    if img_path is None or not img_path.exists():
        raise RenderError(f"scene '{scene.key}' has no screenshot at {scene.image}")
    src = Image.open(img_path).convert("RGB")
    # Pre-shrink very large screenshots: keeps per-frame crops cheap.
    max_w = size[0] * 2
    if src.width > max_w:
        src = src.resize((max_w, int(src.height * max_w / src.width)), Image.LANCZOS)

    n = max(1, int(round(scene.duration * fps)))
    fade = min(int(fps * 0.4), n // 3)
    card_frames = int(fps * min(2.0, scene.duration * 0.4)) if scene.card else 0
    caption = _caption_overlay(size, scene, index, total) if captions else None
    card = _card_overlay(size, scene.card) if scene.card else None
    a, b = scene.start, scene.end

    for i in range(n):
        t = _ease(i / max(1, n - 1))
        key = tuple(a[k] + (b[k] - a[k]) * t for k in range(3))
        box = _viewport(src.size, key, size)
        frame = src.resize(size, Image.BILINEAR, box=box).convert("RGBA")
        if card is not None and i < card_frames:
            alpha = 1.0 if i < card_frames - fade else (card_frames - i) / fade
            frame = Image.alpha_composite(frame, _scaled_alpha(card, alpha))
        if caption is not None and i >= card_frames // 2:
            frame = Image.alpha_composite(frame, caption)
        if fade and (i < fade or i >= n - fade):
            k = (i + 1) / fade if i < fade else (n - i) / fade
            frame = Image.blend(Image.new("RGBA", size, (0, 0, 0, 255)), frame, max(0.0, min(1.0, k)))
        yield frame.convert("RGB").tobytes()


def _scaled_alpha(overlay, alpha: float):
    if alpha >= 1.0:
        return overlay
    from PIL import Image
    r, g, b, a = overlay.split()
    a = a.point(lambda v: int(v * alpha))
    return Image.merge("RGBA", (r, g, b, a))


def _open_encoder(exe: str, size: Size, fps: int, out_path: Path) -> subprocess.Popen:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        exe, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-",
        *_video_codec_args(exe, out_path), "-an", str(out_path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def _finish_encoder(proc: subprocess.Popen, out_path: Path) -> Path:
    proc.stdin.close()
    err = proc.stderr.read().decode(errors="replace")
    if proc.wait() != 0:
        raise RenderError(f"ffmpeg failed: {err.strip()[:500]}")
    return out_path


def render_scene(scene: Scene, out_path: os.PathLike, size: Size = (1280, 720), fps: int = 24,
                 index: int = 1, total: int = 1, captions: bool = True) -> Path:
    """Render one scene to its own clip."""
    exe = _require_ffmpeg()
    out_path = Path(out_path)
    proc = _open_encoder(exe, size, fps, out_path)
    for frame in iter_scene_frames(scene, size, fps, index, total, captions):
        proc.stdin.write(frame)
    return _finish_encoder(proc, out_path)


def render_storyboard(
    storyboard: Storyboard,
    out_path: os.PathLike,
    size: Size = (1280, 720),
    fps: int = 24,
    captions: bool = True,
    on_progress: Optional[Callable[[int, int, Scene], None]] = None,
) -> Path:
    """Render the whole storyboard to a single video file."""
    exe = _require_ffmpeg()
    missing = storyboard.missing_images()
    if missing:
        raise RenderError("missing screenshots: " + ", ".join(f"{s.key} -> {s.image}" for s in missing))
    out_path = Path(out_path)
    proc = _open_encoder(exe, size, fps, out_path)
    total = len(storyboard.scenes)
    for i, scene in enumerate(storyboard.scenes, 1):
        if on_progress:
            on_progress(i, total, scene)
        for frame in iter_scene_frames(scene, size, fps, i, total, captions):
            proc.stdin.write(frame)
    return _finish_encoder(proc, out_path)


def concat_clips(clips: Sequence[os.PathLike], out_path: os.PathLike, size: Size = (1280, 720), fps: int = 24) -> Path:
    """Concatenate clips of any size/codec into one file (re-encodes)."""
    exe = _require_ffmpeg()
    clips = [Path(c) for c in clips]
    if not clips:
        raise RenderError("no clips to concatenate")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    inputs: List[str] = []
    filters: List[str] = []
    for i, c in enumerate(clips):
        inputs += ["-i", str(c)]
        filters.append(
            f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},format=yuv420p[v{i}]"
        )
    chain = "".join(f"[v{i}]" for i in range(len(clips)))
    filters.append(f"{chain}concat=n={len(clips)}:v=1:a=0[v]")
    cmd = [exe, "-y", "-hide_banner", "-loglevel", "error", *inputs,
           "-filter_complex", ";".join(filters), "-map", "[v]",
           *_video_codec_args(exe, out_path), "-an", str(out_path)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RenderError(f"ffmpeg concat failed: {res.stderr.strip()[:500]}")
    return out_path
