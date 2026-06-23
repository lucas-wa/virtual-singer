"""Conversão de timbre.

PADRÃO: Seed-VC (zero-shot, sem treino) — `convert` + `resolve_reference`.
OPCIONAL: RVC (treino por voz) — importe `from src.voice.rvc import train_voice, convert`.
"""
from __future__ import annotations

from .seedvc import convert, resolve_reference, finetune, ft_paths

__all__ = ["convert", "resolve_reference", "finetune", "ft_paths"]
