"""Storyboards: an ordered list of screenshot-backed scenes plus the prompts
that describe how each one should be animated.

The default storyboard is the AIRCollab (app.airnd.ai) product walkthrough.
Every scene carries:

    image      screenshot used as the first frame (path, relative to the
               repo ``examples/`` directory when not absolute)
    narration  one or two sentences of voice-over / caption text
    motion     camera direction for the image-to-video model
    card       optional large title flashed at the start of the scene
    start/end  Ken Burns keyframes as (cx, cy, width) fractions of the image,
               used by the local renderer (width = share of image width shown)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
from typing import List, Optional, Tuple

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"

Keyframe = Tuple[float, float, float]


@dataclass
class Scene:
    key: str
    title: str
    narration: str
    motion: str
    image: Optional[str] = None
    card: Optional[str] = None
    duration: int = 5
    start: Keyframe = (0.5, 0.5, 1.0)
    end: Keyframe = (0.5, 0.5, 1.0)

    def image_path(self) -> Optional[Path]:
        if not self.image:
            return None
        p = Path(self.image)
        if not p.is_absolute() and not p.exists():
            p = EXAMPLES_DIR / self.image
        return p

    def prompt(self, style: str = "") -> str:
        """Prompt sent to the image-to-video model for this scene."""
        parts = []
        if self.card:
            parts.append(f'Open with the on-screen title "{self.card}".')
        parts.append(self.motion)
        parts.append(f"Voice-over: {self.narration}")
        if style:
            parts.append(style)
        return " ".join(p.strip() for p in parts if p and p.strip())


@dataclass
class Storyboard:
    name: str
    product: str
    url: str
    style: str
    scenes: List[Scene] = field(default_factory=list)

    # ------------------------------------------------------------ serialise

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: Optional[Path] = None, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), indent=indent)
        if path:
            Path(path).write_text(text)
        return text

    @classmethod
    def from_dict(cls, data: dict) -> "Storyboard":
        scenes = []
        for s in data.get("scenes", []):
            s = dict(s)
            for k in ("start", "end"):
                if k in s and s[k] is not None:
                    s[k] = tuple(float(v) for v in s[k])
            scenes.append(Scene(**s))
        return cls(
            name=data["name"],
            product=data.get("product", data["name"]),
            url=data.get("url", ""),
            style=data.get("style", ""),
            scenes=scenes,
        )

    @classmethod
    def from_json(cls, path: Path) -> "Storyboard":
        return cls.from_dict(json.loads(Path(path).read_text()))

    # ----------------------------------------------------------- utilities

    @property
    def total_duration(self) -> int:
        return sum(s.duration for s in self.scenes)

    def missing_images(self) -> List[Scene]:
        return [s for s in self.scenes if s.image and not s.image_path().exists()]

    def walkthrough_prompt(self) -> str:
        """One long, self-contained prompt for a full walkthrough video.

        Suitable for pasting into Higgsfield (or any video model) together with
        the referenced screenshots as context images.
        """
        lines = [
            f"Create a {self.total_duration}-second product walkthrough video of {self.product} ({self.url}).",
            self.style,
            "",
            "Use the attached screenshots as the exact reference frames for each window; keep all UI "
            "text crisp and legible, do not invent buttons or change the layout. Smooth cursor movement, "
            "gentle camera pushes and pans, soft cross-dissolves between scenes.",
            "",
            "Scenes, in order:",
        ]
        for i, s in enumerate(self.scenes, 1):
            ref = f" (reference image: {s.image})" if s.image else ""
            lines.append(f"{i}. {s.title} — {s.duration}s{ref}")
            if s.card:
                lines.append(f'   On-screen title: "{s.card}"')
            lines.append(f"   Camera: {s.motion}")
            lines.append(f"   Voice-over: {s.narration}")
        lines += [
            "",
            "Windows that must each get a dedicated moment: "
            + ", ".join(s.title for s in self.scenes)
            + ".",
            "End on the call to action and the product name.",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Default: AIRCollab walkthrough
# ---------------------------------------------------------------------------

_SHOTS = "examples/aircollab/screenshots"
LANDING = f"{_SHOTS}/01_landing.png"
LOGIN = f"{_SHOTS}/02_login.png"
DASHBOARD = f"{_SHOTS}/03_dashboard.png"
DISCOVER = f"{_SHOTS}/04_discover.png"
DISCOVER_RESULTS = f"{_SHOTS}/05_discover_results.png"

AIRCOLLAB_STYLE = (
    "Style: clean, modern SaaS screen-recording look, 16:9, bright white and blue palette matching "
    "the AIRCollab interface, calm confident narration, light ambient music, no watermarks."
)


def _sidebar_scene(key, title, narration, cy, motion=None) -> Scene:
    """A scene that zooms into one item of the dashboard's left sidebar."""
    return Scene(
        key=key,
        title=title,
        narration=narration,
        motion=motion or (
            f"Start on the full dashboard, then push in on the left sidebar and highlight the "
            f"'{title}' menu item as the cursor moves over it and clicks."
        ),
        image=DASHBOARD,
        start=(0.5, 0.45, 1.0),
        end=(0.2, cy, 0.42),
    )


def aircollab_storyboard() -> Storyboard:
    scenes = [
        Scene(
            key="intro",
            title="Welcome to AIRCollab",
            card="Welcome to AIRCollab",
            narration=(
                "Welcome to AIRCollab, the platform that connects researchers worldwide. Real-time "
                "communication, document sharing, virtual meetings and project management, all in one place."
            ),
            motion=(
                "Slow push-in on the hero headline 'Connect, Collaborate, Discover', then drift right "
                "toward the collage of researchers around the globe."
            ),
            image=LANDING,
            duration=6,
            start=(0.5, 0.35, 1.0),
            end=(0.2, 0.21, 0.55),
        ),
        Scene(
            key="landing",
            title="Landing page",
            narration=(
                "The public home page states the mission, showcases Featured Collaborators such as "
                "Dr. Sarah Chen and Prof. Marcus Rodriguez, and invites you to Start Collaborating."
            ),
            motion=(
                "Pan down from the hero section past the 'Start Collaborating' button to the Featured "
                "Collaborators cards; the cursor hovers the primary call-to-action button."
            ),
            image=LANDING,
            start=(0.5, 0.3, 0.95),
            end=(0.5, 0.7, 0.8),
        ),
        Scene(
            key="login",
            title="Please login first",
            card="Please login first",
            narration=(
                "To reach the secure workspace, please login first. Enter your email and password and "
                "press Sign In, or use Sign up if you are new. Get Help is one click away."
            ),
            motion=(
                "Gentle zoom onto the centered 'Welcome back' card; the cursor moves from the Email field "
                "to the Password field and rests on the dark Sign In button."
            ),
            image=LOGIN,
            start=(0.5, 0.5, 1.0),
            end=(0.37, 0.55, 0.55),
        ),
        Scene(
            key="dashboard",
            title="Dashboard overview",
            narration=(
                "Good Afternoon, Patrik Jane. The dashboard greets you personally and summarises Total "
                "Collaborations, Total Collaborators, Total Collaboration Days and Total Meetings, then "
                "lists your active collaboration with its In Progress status, dates and calendar."
            ),
            motion=(
                "Sweep left to right across the four summary cards, then tilt down to the My Collaboration "
                "panel, pausing on the green 'In Progress' badge and the September calendar."
            ),
            image=DASHBOARD,
            duration=7,
            start=(0.43, 0.18, 0.65),
            end=(0.43, 0.6, 0.7),
        ),
        Scene(
            key="discover",
            title="Discover collaborators",
            narration=(
                "Discover Collaborators lets you switch between Best Matching and Search More, search by "
                "name or interest, and sort by relevance."
            ),
            motion=(
                "Push in on the Best Matching / Search More toggle and the search box as the cursor "
                "clicks 'Search More'."
            ),
            image=DISCOVER,
            start=(0.5, 0.4, 1.0),
            end=(0.3, 0.17, 0.5),
        ),
        Scene(
            key="discover_results",
            title="Search, filter and save peers",
            narration=(
                "Filter researchers by expertise and by the resources they have, such as ideas, "
                "manuscripts, human samples or animal models, then save a peer with the heart icon."
            ),
            motion=(
                "Slow pan down the results table; the cursor clicks the heart icon next to a researcher "
                "and it fills blue."
            ),
            image=DISCOVER_RESULTS,
            duration=6,
            start=(0.45, 0.45, 0.75),
            end=(0.65, 0.62, 0.55),
        ),
        _sidebar_scene(
            "saved", "Saved collaborators",
            "Saved Collaborators keeps every researcher you shortlisted one click away.", 0.187,
        ),
        Scene(
            key="collaboration",
            title="Active collaboration",
            narration=(
                "Open an active collaboration to follow its status, timeline and members, and to manage "
                "the project from start to finish."
            ),
            motion=(
                "Start on the sidebar 'Collaboration' item, then glide right to the My Collaboration card "
                "showing the date range and the collaborator list."
            ),
            image=DASHBOARD,
            start=(0.2, 0.285, 0.42),
            end=(0.43, 0.6, 0.65),
        ),
        _sidebar_scene(
            "data_center", "Data Center",
            "The Data Center is the secure workspace for file sharing, metadata tracking and document "
            "management across your collaboration.", 0.367,
        ),
        _sidebar_scene(
            "chat", "Real-time chat",
            "Chat keeps the conversation going in real time with every collaborator on the project.", 0.326,
        ),
        _sidebar_scene(
            "shipment", "Shipment assistant",
            "Supporting services start with Shipment: an AI assistant for logistics and shipping of "
            "samples and materials between labs.", 0.466,
        ),
        _sidebar_scene(
            "quotation", "Pricing and quotations",
            "Quotation produces pricing and quotations for services and materials in a few clicks.", 0.507,
        ),
        _sidebar_scene(
            "equipment", "Equipment rentals",
            "Equipment lets you rent instruments and lab equipment from partner institutions.", 0.547,
        ),
        Scene(
            key="outro",
            title="Start collaborating",
            card="Start Collaborating",
            narration=(
                "Join the world's largest network of researchers. Start collaborating today at app.airnd.ai."
            ),
            motion=(
                "Return to the landing page and settle on the blue 'Start Collaborating' button; the "
                "cursor clicks it and the frame softens to the AIRCollab logo."
            ),
            image=LANDING,
            duration=6,
            start=(0.5, 0.4, 1.0),
            end=(0.12, 0.6, 0.4),
        ),
    ]
    return Storyboard(
        name="aircollab-walkthrough",
        product="AIRCollab",
        url="https://app.airnd.ai",
        style=AIRCOLLAB_STYLE,
        scenes=scenes,
    )


def load_storyboard(path: Optional[str] = None) -> Storyboard:
    """Load a storyboard JSON file, or the default AIRCollab storyboard."""
    if path:
        return Storyboard.from_json(Path(path))
    return aircollab_storyboard()
