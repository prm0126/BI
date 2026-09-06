"""Walkthrough generation: storyboard -> per-scene clips -> one video.

Providers:
    higgsfield  each scene's screenshot is animated by Higgsfield image-to-video
    local       Ken Burns render from the screenshots (no AI, no credentials)
    auto        higgsfield when credentials are configured, otherwise local
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from . import higgsfield as hf
from .render import concat_clips, media_duration, plan_narration, render_scene, render_storyboard
from . import tts
from .storyboard import Scene, Storyboard

logger = logging.getLogger(__name__)

PROVIDERS = ("auto", "higgsfield", "local")
ProgressFn = Callable[[str], None]


@dataclass
class WalkthroughResult:
    path: Path
    provider: str
    prompt: str
    clips: List[Path] = field(default_factory=list)
    jobs: List[str] = field(default_factory=list)
    narration: str = "none"

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "provider": self.provider,
            "clips": [str(c) for c in self.clips],
            "jobs": self.jobs,
            "narration": self.narration,
            "prompt": self.prompt,
        }


def choose_provider(provider: str = "auto") -> str:
    if provider not in PROVIDERS:
        raise ValueError(f"provider must be one of {PROVIDERS}")
    if provider == "auto":
        return "higgsfield" if hf.is_configured() else "local"
    return provider


def generate_walkthrough(
    storyboard: Storyboard,
    out_path,
    provider: str = "auto",
    model: Optional[str] = None,
    workdir=None,
    size: Tuple[int, int] = (1280, 720),
    fps: int = 24,
    captions: bool = True,
    poll_interval: float = 3.0,
    job_timeout: float = 900.0,
    on_progress: Optional[ProgressFn] = None,
    client: Optional[hf.HiggsfieldClient] = None,
    narration: str = "auto",
    voice: Optional[str] = None,
    speech_rate: float = 1.0,
) -> WalkthroughResult:
    """Produce the walkthrough video and return where it landed.

    ``narration`` picks the voice-over engine ("auto", "piper", "espeak", "none").
    """
    say = on_progress or (lambda msg: logger.info(msg))
    out_path = Path(out_path)
    chosen = choose_provider(provider)
    prompt = storyboard.walkthrough_prompt()
    engine = tts.choose_engine(narration, voice)
    if narration != "none" and engine == "none":
        say("No text-to-speech engine available (install piper-tts + a voice, or espeak-ng): rendering silent.")
    elif engine != "none":
        say(f"Voice-over: {engine}")
    workdir = Path(workdir) if workdir else out_path.parent / (out_path.stem + "_clips")
    workdir.mkdir(parents=True, exist_ok=True)

    if chosen == "local":
        say(f"Rendering {len(storyboard.scenes)} scenes locally -> {out_path}")
        render_storyboard(
            storyboard, out_path, size=size, fps=fps, captions=captions,
            on_progress=lambda i, n, s: say(f"[{i}/{n}] {s.title}"),
            narration=engine, voice=voice, speech_rate=speech_rate, workdir=workdir,
        )
        return WalkthroughResult(path=out_path, provider="local", prompt=prompt, narration=engine)

    client = client or hf.HiggsfieldClient()
    clips: List[Path] = []
    jobs: List[str] = []
    total = len(storyboard.scenes)
    for i, scene in enumerate(storyboard.scenes, 1):
        clip = workdir / f"{i:02d}_{scene.key}.mp4"
        if clip.exists():
            say(f"[{i}/{total}] {scene.title}: reusing {clip.name}")
            clips.append(clip)
            continue
        image = scene.image_path()
        if image is None or not image.exists():
            say(f"[{i}/{total}] {scene.title}: no screenshot, rendering title card locally")
            clips.append(render_scene(scene, clip, size=size, fps=fps, index=i, total=total, captions=captions))
            continue
        say(f"[{i}/{total}] {scene.title}: uploading + generating on Higgsfield ({model or hf.DEFAULT_MODEL})")
        image_url = client.upload_file(image)
        job = client.submit_image_to_video(
            image_url, scene.prompt(storyboard.style), model=model, duration=scene.duration,
        )
        jobs.append(job.request_id)
        job = client.wait(
            job, poll_interval=poll_interval, timeout=job_timeout,
            on_update=lambda j, i=i: say(f"[{i}/{total}] {j.request_id}: {j.status}"),
        )
        clips.append(client.download(job.video_url, clip))

    # Voice-over: each clip is held on its last frame until its narration ends.
    clip_durations = [media_duration(c) for c in clips]
    durations, track, engine = plan_narration(
        storyboard.scenes, workdir, engine=engine, voice=voice, rate=speech_rate, min_durations=clip_durations,
    )
    say(f"Stitching {len(clips)} clips -> {out_path}")
    concat_clips(clips, out_path, size=size, fps=fps, min_durations=durations, audio=track)
    return WalkthroughResult(path=out_path, provider="higgsfield", prompt=prompt, clips=clips, jobs=jobs,
                             narration=engine)
