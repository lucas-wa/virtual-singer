"""Motor SVS: Performance -> voz-guia cantada (DiffSinger + NSF-HiFiGAN)."""
from __future__ import annotations

from .ds_format import performance_to_ds, midi_to_note_name
from .dsp_guide import synthesize_dsp

__all__ = ["performance_to_ds", "midi_to_note_name", "synthesize_dsp", "synthesize"]


def synthesize(*args, **kwargs):
    """Import tardio do motor DiffSinger (deps pesadas só quando realmente usado)."""
    from .engine import synthesize as _synth
    return _synth(*args, **kwargs)
