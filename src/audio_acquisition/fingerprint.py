"""
fingerprint.py -- acoustic identification of a residual segment.

Chromaprint (`fpcalc`) computes an acoustic fingerprint; the AcoustID web API
maps it to a MusicBrainz recording (title + artist). This is the paper's
`Acoustic` recogniser (Alg. 1), invoked ONLY on residual constituents -- the
unnamed / name-colliding positions a cheap symbolic handle could not place
(Cor. 4.8). Unreleased "ID" tracks legitimately return None: they are not in
the reference database, and stay in the honest residual (Rem. 6.11).
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import wave

logger = logging.getLogger(__name__)

ACOUSTID_ENDPOINT = "https://api.acoustid.org/v2/lookup"


class FingerprintError(RuntimeError):
    """Raised when fingerprinting cannot run (missing fpcalc / API key)."""


def _have_fpcalc() -> bool:
    from shutil import which

    return which("fpcalc") is not None


def cut_segment(wav_path: str, start: float, end: float, out_dir: str | None = None) -> str:
    """
    Write [start, end) seconds of a WAV to a new WAV and return its path.
    Uses the stdlib wave module (no ffmpeg dependency for the cut itself).
    """
    out_dir = out_dir or tempfile.mkdtemp(prefix="yokozuna_seg_")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"seg_{int(start)}_{int(end)}.wav")

    with wave.open(wav_path, "rb") as w:
        sr = w.getframerate()
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        start_frame = max(0, int(start * sr))
        end_frame = min(w.getnframes(), int(end * sr))
        w.setpos(start_frame)
        frames = w.readframes(max(0, end_frame - start_frame))

    with wave.open(out_path, "wb") as o:
        o.setnchannels(n_channels)
        o.setsampwidth(sampwidth)
        o.setframerate(sr)
        o.writeframes(frames)

    return out_path


def _fpcalc(segment_wav: str) -> tuple[int, str]:
    """Run fpcalc; return (duration_seconds, fingerprint)."""
    if not _have_fpcalc():
        raise FingerprintError(
            "fpcalc (Chromaprint) not found on PATH. Install libchromaprint-tools."
        )
    try:
        out = subprocess.run(
            ["fpcalc", "-json", segment_wav],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        raise FingerprintError(f"fpcalc failed: {e}") from e

    import json

    data = json.loads(out.stdout)
    return int(round(data["duration"])), data["fingerprint"]


def identify_segment(
    wav_path: str,
    start: float,
    end: float,
    api_key: str | None = None,
) -> dict | None:
    """
    Identify one residual segment acoustically.

    Returns {"name": "Artist - Title", "artist": ..., "title": ..., "score": ...}
    on a confident match, or None when the segment cannot be identified (an
    unreleased ID track, or no database hit) -- which is the correct honest
    outcome, not an error.

    Raises FingerprintError only when fingerprinting itself cannot run.
    """
    api_key = api_key or os.environ.get("ACOUSTID_API_KEY")
    if not api_key:
        raise FingerprintError(
            "ACOUSTID_API_KEY not set. Get a free key at https://acoustid.org/new-application"
        )

    seg_wav = cut_segment(wav_path, start, end)
    try:
        duration, fingerprint = _fpcalc(seg_wav)
    finally:
        try:
            os.remove(seg_wav)
        except OSError:
            pass

    try:
        import requests
    except ImportError as e:  # pragma: no cover
        raise FingerprintError("`requests` is required for AcoustID lookup.") from e

    params = {
        "client": api_key,
        "duration": duration,
        "fingerprint": fingerprint,
        "meta": "recordings",
    }
    try:
        resp = requests.get(ACOUSTID_ENDPOINT, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as e:
        logger.warning("AcoustID lookup failed: %s", e)
        return None

    results = payload.get("results", [])
    if not results:
        return None

    # Best result by AcoustID score; take its first recording.
    best = max(results, key=lambda r: r.get("score", 0.0))
    recordings = best.get("recordings", [])
    if not recordings:
        return None

    rec = recordings[0]
    title = rec.get("title")
    artists = rec.get("artists", [])
    artist = ", ".join(a.get("name", "") for a in artists) if artists else None
    if not title:
        return None

    name = f"{artist} - {title}" if artist else title
    return {
        "name": name,
        "artist": artist,
        "title": title,
        "score": float(best.get("score", 0.0)),
    }
