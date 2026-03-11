"""
Yokozuna Format Decoder — Reads .ykz container and reconstructs both channels.

Extracts:
  - Physical channel: raw PCM audio
  - Categorical channel: state trajectory, S-entropy stream, partition coordinates
  - Metadata: manifest, harmonics, full analysis results
"""

import json
import struct
import zipfile
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional

FORMAT_MAGIC = b"YKZ\x01"


@dataclass
class YokozunaFile:
    """Decoded contents of a .ykz file."""
    manifest: Dict[str, Any]
    audio: np.ndarray           # float32 mono PCM
    sample_rate: int
    trajectory: np.ndarray      # Nx6: time, mode_index, phase, amplitude, frequency, flat_index
    entropy_stream: np.ndarray  # Nx3: S_k, S_t, S_e
    partitions: np.ndarray      # Nx4: n, l, m, s
    harmonics: Dict[str, Any]
    analysis: Dict[str, Any]

    @property
    def duration(self) -> float:
        return len(self.audio) / self.sample_rate

    @property
    def n_trajectory_points(self) -> int:
        return len(self.trajectory)

    @property
    def track_name(self) -> str:
        return self.manifest.get('track_name', 'unknown')

    @property
    def C_phys(self) -> float:
        return self.manifest['channels']['physical']['C_phys_bits_per_s']

    @property
    def C_cat(self) -> float:
        return self.manifest['channels']['categorical']['C_cat_bits']

    def summary(self) -> str:
        m = self.manifest
        lines = [
            f"Yokozuna File: {m['track_name']}",
            f"  Format version: {m['version']}",
            f"  Duration: {m['total_duration_s']:.1f}s, SR: {m['sample_rate']} Hz",
            f"  Physical channel: {self.C_phys:.0f} bits/s",
            f"  Categorical channel: {self.C_cat:.0f} bits",
            f"  Trajectory: {self.n_trajectory_points} points",
            f"  Triple equivalence verified: {m['triple_equivalence']['verified']}",
        ]
        return '\n'.join(lines)


class YokozunaDecoder:
    """Decodes a .ykz file into its constituent channels and metadata."""

    def decode(self, path: str) -> YokozunaFile:
        """Read and decode a .ykz file.

        Args:
            path: Path to .ykz file.

        Returns:
            YokozunaFile with all channels and metadata.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"YKZ file not found: {path}")

        with zipfile.ZipFile(str(path), 'r') as zf:
            manifest = json.loads(zf.read("manifest.json"))
            audio_raw = zf.read("audio.pcm")
            trajectory_raw = zf.read("trajectory.bin")
            entropy_raw = zf.read("entropy.bin")
            partitions_raw = zf.read("partitions.bin")
            harmonics = json.loads(zf.read("harmonics.json"))
            analysis = json.loads(zf.read("analysis.json"))

        # Decode audio
        audio = np.frombuffer(audio_raw, dtype=np.float32)
        sr = manifest['sample_rate']

        # Decode trajectory
        trajectory = self._unpack_trajectory(trajectory_raw)

        # Decode entropy stream
        entropy_stream = self._unpack_entropy(entropy_raw)

        # Decode partitions
        partitions = self._unpack_partitions(partitions_raw)

        return YokozunaFile(
            manifest=manifest,
            audio=audio,
            sample_rate=sr,
            trajectory=trajectory,
            entropy_stream=entropy_stream,
            partitions=partitions,
            harmonics=harmonics,
            analysis=analysis,
        )

    def _unpack_trajectory(self, data: bytes) -> np.ndarray:
        """Unpack binary trajectory into Nx6 array."""
        magic = data[:4]
        if magic != FORMAT_MAGIC:
            raise ValueError("Invalid trajectory stream: bad magic number")
        count = struct.unpack('<I', data[4:8])[0]
        # Each record: d(8) + h(2) + f(4) + f(4) + f(4) + i(4) = 26 bytes
        record_size = struct.calcsize('<d h f f f i')
        result = np.zeros((count, 6), dtype=np.float64)
        offset = 8
        for i in range(count):
            t, mode_idx, phase, amp, freq, flat_idx = struct.unpack_from(
                '<d h f f f i', data, offset
            )
            result[i] = [t, mode_idx, phase, amp, freq, flat_idx]
            offset += record_size
        return result

    def _unpack_entropy(self, data: bytes) -> np.ndarray:
        """Unpack S-entropy stream into Nx3 array."""
        magic = data[:4]
        if magic != FORMAT_MAGIC:
            raise ValueError("Invalid entropy stream: bad magic number")
        count = struct.unpack('<I', data[4:8])[0]
        arr = np.frombuffer(data[8:], dtype=np.float64).reshape(count, 3)
        return arr.copy()

    def _unpack_partitions(self, data: bytes) -> np.ndarray:
        """Unpack partition coordinates into Nx4 array (n, l, m, s)."""
        magic = data[:4]
        if magic != FORMAT_MAGIC:
            raise ValueError("Invalid partition stream: bad magic number")
        count = struct.unpack('<I', data[4:8])[0]
        record_size = struct.calcsize('<h h h b')
        result = np.zeros((count, 4), dtype=np.float64)
        offset = 8
        for i in range(count):
            n, l, m, s_byte = struct.unpack_from('<h h h b', data, offset)
            result[i] = [n, l, m, 0.5 if s_byte else -0.5]
            offset += record_size
        return result
