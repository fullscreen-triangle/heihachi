"""
Audio acquisition from continuous-item URLs (YouTube / SoundCloud) and
acoustic fingerprinting of residual segments.

These modules realise the acoustic-descent stage of the whole-item search
algorithm (docs/audio-search-algorithm): they run ONLY on the residual
constituents a cheap symbolic name could not place.
"""

from .downloader import download_audio, DownloadError  # noqa: F401
from .fingerprint import identify_segment, FingerprintError  # noqa: F401
