"""Separação de fontes via Demucs (instrumental / limpeza de amostras de voz)."""
from __future__ import annotations

from .demucs_wrap import separate

__all__ = ["separate"]
