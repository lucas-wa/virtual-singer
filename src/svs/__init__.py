"""Motor SVS: Performance -> voz-guia cantada (DiffSinger + NSF-HiFiGAN)."""
from __future__ import annotations

from .ds_format import performance_to_ds, midi_to_note_name
from .engine import synthesize

__all__ = ["performance_to_ds", "midi_to_note_name", "synthesize"]
