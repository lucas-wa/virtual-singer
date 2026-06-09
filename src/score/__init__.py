"""Front-end de partitura: MIDI/MusicXML + letra -> fonemas alinhados às notas.

Entrada típica:
    build_performance("data/songs/demo")  # pasta com .mid/.musicxml + lyrics.txt

Saída: um objeto `Performance` que o motor SVS consome (notas, durações, fonemas).
"""
from __future__ import annotations

from .model import Note, Phoneme, Performance
from .build import build_performance, load_score_dir

__all__ = ["Note", "Phoneme", "Performance", "build_performance", "load_score_dir"]
