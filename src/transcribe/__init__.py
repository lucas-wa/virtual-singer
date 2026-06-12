"""Transcrição Automática (AMT): áudio melódico -> notas -> MIDI.

Entrada esperada: a SUA voz/instrumento cantando ou tocando a melodia (mono, monofônico).
Saída: lista de `Note` e, via script, um score.mid pronto para o importador.
"""
from __future__ import annotations

from .amt import extract_f0, f0_to_notes, transcribe_audio, notes_to_midi

__all__ = ["extract_f0", "f0_to_notes", "transcribe_audio", "notes_to_midi"]
