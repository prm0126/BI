"""AI video generation: screenshots -> storyboard -> Higgsfield (or local) walkthrough."""

from .storyboard import Scene, Storyboard, aircollab_storyboard, load_storyboard
from .higgsfield import HiggsfieldClient, HiggsfieldError, HiggsfieldNotConfigured, is_configured
from .tts import synthesize, available_engines, download_piper_voice, TTSError
from .pipeline import generate_walkthrough, choose_provider, WalkthroughResult, PROVIDERS

__all__ = [
    "Scene",
    "Storyboard",
    "aircollab_storyboard",
    "load_storyboard",
    "HiggsfieldClient",
    "HiggsfieldError",
    "HiggsfieldNotConfigured",
    "is_configured",
    "generate_walkthrough",
    "choose_provider",
    "WalkthroughResult",
    "PROVIDERS",
    "synthesize",
    "available_engines",
    "download_piper_voice",
    "TTSError",
]
