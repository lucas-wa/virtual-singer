"""Utilidades de E/S e mixagem de áudio (independentes de GPU)."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def load_wav(path: str | Path, sr: int | None = None) -> Tuple[np.ndarray, int]:
    """Carrega um arquivo de áudio como mono float32 em [-1, 1]."""
    import soundfile as sf

    audio, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # mono
    if sr is not None and sr != file_sr:
        audio = resample(audio, file_sr, sr)
        file_sr = sr
    return audio, file_sr


def save_wav(path: str | Path, audio: np.ndarray, sr: int) -> None:
    import soundfile as sf

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak  # evita clipping
    sf.write(str(path), audio.astype(np.float32), sr)


def resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    import librosa

    return librosa.resample(audio, orig_sr=src_sr, target_sr=dst_sr)


def mix(vocal: np.ndarray, instrumental: np.ndarray,
        vocal_gain: float = 1.0, inst_gain: float = 0.7) -> np.ndarray:
    """Mixa voz + instrumental, alinhando o comprimento com zero-padding."""
    n = max(len(vocal), len(instrumental))
    v = np.pad(vocal, (0, n - len(vocal)))
    i = np.pad(instrumental, (0, n - len(instrumental)))
    out = vocal_gain * v + inst_gain * i
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:
        out = out / peak
    return out
