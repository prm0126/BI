"""Capture website screenshots with Playwright (Chromium).

    pip install 'bi-forecast[video]' && playwright install chromium

Screenshots become the reference frames of a storyboard. Pages behind a login
can be captured by passing a Playwright storage-state JSON (cookies + local
storage) exported from an authenticated browser session.
"""

from __future__ import annotations

import glob
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


class ScreenshotError(RuntimeError):
    pass


def find_chromium() -> Optional[str]:
    """Best-effort Chromium binary when Playwright's own download is absent."""
    env = os.environ.get("CHROMIUM_PATH")
    if env and Path(env).exists():
        return env
    pw = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if pw:
        for cand in [os.path.join(pw, "chromium"), *sorted(glob.glob(os.path.join(pw, "chromium-*", "chrome-linux*", "chrome")), reverse=True)]:
            if os.path.exists(cand) and os.access(cand, os.X_OK):
                return cand
    for name in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable"):
        found = shutil.which(name)
        if found:
            return found
    return None


def capture_screenshot(
    url: str,
    out_path: os.PathLike,
    viewport: Tuple[int, int] = (1440, 900),
    full_page: bool = False,
    wait_ms: int = 1500,
    wait_for_selector: Optional[str] = None,
    storage_state: Optional[os.PathLike] = None,
    device_scale_factor: float = 2.0,
    executable_path: Optional[str] = None,
) -> Path:
    """Open ``url`` in headless Chromium and save a PNG screenshot."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:  # pragma: no cover
        raise ScreenshotError("playwright is not installed: pip install 'bi-forecast[video]'") from e

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    launch: Dict = {"headless": True}
    if executable_path:
        launch["executable_path"] = executable_path

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(**launch)
        except Exception as first:  # Playwright's bundled build missing: try a system/preinstalled Chromium.
            fallback = None if executable_path else find_chromium()
            if not fallback:
                raise ScreenshotError(
                    f"Could not launch Chromium ({str(first).splitlines()[0]}). Run `playwright install chromium` "
                    "or set CHROMIUM_PATH to a Chrome/Chromium binary."
                ) from first
            browser = p.chromium.launch(headless=True, executable_path=fallback)
        try:
            context = browser.new_context(
                viewport={"width": viewport[0], "height": viewport[1]},
                device_scale_factor=device_scale_factor,
                storage_state=str(storage_state) if storage_state else None,
            )
            page = context.new_page()
            page.goto(url, wait_until="networkidle", timeout=60_000)
            if wait_for_selector:
                page.wait_for_selector(wait_for_selector, timeout=30_000)
            if wait_ms:
                page.wait_for_timeout(wait_ms)
            page.screenshot(path=str(out_path), full_page=full_page)
        finally:
            browser.close()
    return out_path


def capture_many(
    targets: Iterable[Tuple[str, os.PathLike]], **kwargs
) -> List[Path]:
    """Capture several (url, out_path) pairs with the same settings."""
    return [capture_screenshot(url, out, **kwargs) for url, out in targets]
