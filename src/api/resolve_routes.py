"""
resolve_routes.py -- URL resolve endpoint (acoustic-descent worker).

Realises the acoustic stage of the whole-item search algorithm
(docs/audio-search-algorithm) as an async HTTP job:

    POST /api/v1/resolve            {url, known_names?} -> {job_id}
    GET  /api/v1/resolve/jobs/<id>                       -> {status, result?}

Pipeline per job:
  1. download_audio(url)                        (yt-dlp)
  2. TransitionDetector.detect + SegmentClusterer  -> track-region boundaries
  3. reconcile boundaries with known_names (metadata) -> residual segments
  4. identify_segment(...) on the residual ONLY   (Chromaprint + AcoustID)
  5. return the completed signature + remaining residual

The job store here is self-contained (a small threaded dict) so it does not
couple to the fixed dispatch in job_manager.py. Acoustic identification runs on
the residual and nowhere else (Cor. 4.8); unresolved IDs stay in the residual
(Rem. 6.11 -- honest output).
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import uuid
from datetime import datetime

import numpy as np
from flask import Blueprint, jsonify, request

logger = logging.getLogger(__name__)

resolve_bp = Blueprint("resolve", __name__)

# ---- self-contained job store ---------------------------------------------
_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()


def _set_job(job_id: str, **fields):
    with _JOBS_LOCK:
        _JOBS.setdefault(job_id, {})
        _JOBS[job_id].update(fields)


def _get_job(job_id: str) -> dict | None:
    with _JOBS_LOCK:
        j = _JOBS.get(job_id)
        return dict(j) if j else None


# ---- core pipeline ---------------------------------------------------------
def _load_mono(wav_path: str, sr: int = 44100) -> np.ndarray:
    import librosa

    audio, _ = librosa.load(wav_path, sr=sr, mono=True)
    return audio


def _segment_boundaries(audio: np.ndarray) -> list[dict]:
    """Track-region boundaries via the existing DJ-mix detectors."""
    from ..annotation.transition_detector import TransitionDetector
    from ..annotation.segment_clustering import SegmentClusterer

    transitions = TransitionDetector().detect(audio)
    trans_list = transitions.get("transitions", []) if isinstance(transitions, dict) else []
    result = SegmentClusterer().analyze(audio, trans_list)
    return result.get("segments", [])


def _reconcile(segments: list[dict], known_names: list[dict]) -> list[dict]:
    """
    Build the signature by assigning each known name to the segment whose
    region contains its onset; segments with no assigned name are residual.
    Returns [{pos, tStart, tEnd, name, artist, source}] in time order.
    """
    segs = sorted(segments, key=lambda s: s.get("start_time", 0.0))
    # index known names by onset for containment assignment
    names = sorted(
        [k for k in (known_names or []) if k.get("tStart") is not None],
        key=lambda k: k["tStart"],
    )

    sig = []
    for pos, s in enumerate(segs):
        start = float(s.get("start_time", 0.0))
        end = float(s.get("end_time", start))
        placed = None
        for k in names:
            if start - 2.0 <= k["tStart"] < end + 2.0:  # small tolerance
                placed = k
                break
        if placed and placed.get("name"):
            sig.append(
                {
                    "pos": pos,
                    "tStart": start,
                    "tEnd": end,
                    "name": placed["name"],
                    "artist": placed.get("artist"),
                    "source": "meta",
                }
            )
        else:
            sig.append(
                {
                    "pos": pos,
                    "tStart": start,
                    "tEnd": end,
                    "name": None,
                    "artist": None,
                    "source": "residual",
                }
            )
    return sig


def _run_job(job_id: str, url: str, known_names: list[dict]):
    work_dir = None
    try:
        _set_job(job_id, status="processing", started_at=datetime.now().isoformat())
        from ..audio_acquisition.downloader import download_audio
        from ..audio_acquisition.fingerprint import identify_segment, FingerprintError

        wav_path = download_audio(url)
        work_dir = os.path.dirname(wav_path)

        audio = _load_mono(wav_path)
        segments = _segment_boundaries(audio)
        signature = _reconcile(segments, known_names)

        residual_positions = [s["pos"] for s in signature if not s["name"]]
        acoustic_calls = 0
        for s in signature:
            if s["name"]:
                continue  # placed by a name; nothing owed (Thm. 4.7)
            try:
                hit = identify_segment(wav_path, s["tStart"], s["tEnd"])
                acoustic_calls += 1
            except FingerprintError as fe:
                _set_job(job_id, warning=str(fe))
                hit = None
            if hit:
                s["name"] = hit["name"]
                s["artist"] = hit.get("artist")
                s["source"] = "acoustic"
            else:
                s["source"] = "id"  # honest residual: an unidentified ID

        residual = [s["pos"] for s in signature if not s["name"]]
        result = {
            "url": url,
            "signature": signature,
            "residual": residual,
            "segment_count": len(signature),
            "residual_before_acoustic": len(residual_positions),
            "acoustic_calls": acoustic_calls,  # == residual size (Cor. 4.8)
        }
        _set_job(
            job_id,
            status="completed",
            completed_at=datetime.now().isoformat(),
            result=result,
        )
        logger.info(
            "resolve job %s done: %d segments, %d residual, %d acoustic calls",
            job_id,
            len(signature),
            len(residual),
            acoustic_calls,
        )
    except Exception as e:
        logger.exception("resolve job %s failed", job_id)
        _set_job(job_id, status="failed", error=str(e), completed_at=datetime.now().isoformat())
    finally:
        if work_dir and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)


# ---- routes ----------------------------------------------------------------
@resolve_bp.route("/resolve", methods=["POST"])
def submit_resolve():
    body = request.get_json(silent=True) or {}
    url = body.get("url")
    if not url:
        return jsonify({"error": "No url provided"}), 400
    known_names = body.get("known_names", [])

    job_id = str(uuid.uuid4())
    _set_job(job_id, status="pending", created_at=datetime.now().isoformat())
    threading.Thread(target=_run_job, args=(job_id, url, known_names), daemon=True).start()
    return jsonify({"job_id": job_id, "status": "pending"}), 202


@resolve_bp.route("/resolve/jobs/<job_id>", methods=["GET"])
def resolve_job_status(job_id):
    job = _get_job(job_id)
    if not job:
        return jsonify({"error": "Unknown job"}), 404
    return jsonify(job)
