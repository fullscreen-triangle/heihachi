"""
downloader.py -- acquire audio from a YouTube / SoundCloud URL via yt-dlp.

This is the URL-acquisition front end of the whole-item search pipeline. It is
deliberately isolated: yt-dlp is fragile to platform changes and has ToS
considerations, so everything URL-facing lives here and nowhere else. The rest
of the pipeline sees only a local wav path.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile

logger = logging.getLogger(__name__)

_SUPPORTED = re.compile(
    r"^(https?://)?(www\.)?(youtube\.com|youtu\.be|soundcloud\.com)/", re.I
)


class DownloadError(RuntimeError):
    """Raised when a URL cannot be acquired as audio."""


def is_supported(url: str) -> bool:
    """True iff the URL is a YouTube or SoundCloud link we can acquire."""
    return bool(url and _SUPPORTED.match(url.strip()))


def download_audio(url: str, out_dir: str | None = None, sr: int = 44100) -> str:
    """
    Download the audio of a continuous item and return a local WAV path.

    Args:
        url: A YouTube or SoundCloud URL.
        out_dir: Directory to write into (a temp dir by default).
        sr: Target sample rate; the pipeline's detectors assume 44100.

    Returns:
        Path to a mono/stereo WAV file on disk. Caller owns cleanup.

    Raises:
        DownloadError: on unsupported URL or download/extraction failure.
    """
    if not is_supported(url):
        raise DownloadError(f"Unsupported URL (need YouTube/SoundCloud): {url!r}")

    try:
        import yt_dlp  # imported lazily so the rest of the app runs without it
    except ImportError as e:  # pragma: no cover - environment dependent
        raise DownloadError(
            "yt-dlp is not installed. `pip install yt-dlp` on the worker host."
        ) from e

    out_dir = out_dir or tempfile.mkdtemp(prefix="yokozuna_dl_")
    os.makedirs(out_dir, exist_ok=True)
    outtmpl = os.path.join(out_dir, "%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        # Resample at extraction so downstream detectors get the sr they expect.
        "postprocessor_args": ["-ar", str(sr)],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as e:  # yt-dlp raises many subclasses
        raise DownloadError(f"Download failed for {url!r}: {e}") from e

    # Resolve the produced wav path (extension changed by the postprocessor).
    vid = info.get("id", "")
    wav_path = os.path.join(out_dir, f"{vid}.wav")
    if not os.path.exists(wav_path):
        # Fall back to any wav yt-dlp left in out_dir.
        wavs = [f for f in os.listdir(out_dir) if f.lower().endswith(".wav")]
        if not wavs:
            raise DownloadError(f"No WAV produced for {url!r}")
        wav_path = os.path.join(out_dir, wavs[0])

    logger.info("Acquired audio for %s -> %s", url, wav_path)
    return wav_path
