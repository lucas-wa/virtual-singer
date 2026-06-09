"""Ponto de entrada do front-end: pasta de partitura -> objeto `Performance`.

Layout esperado de uma pasta de música (ex.: data/songs/demo/):
    score.musicxml         (preferido — já traz a letra)
  ou
    score.mid + lyrics.txt (MIDI da melodia + letra, uma sílaba por nota)
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from .model import Note, Performance
from .parse import parse_midi, parse_musicxml, read_lyrics_file
from .phonemize import phonemize_notes


def _find(folder: Path, *names: str) -> Path | None:
    for name in names:
        hits = list(folder.glob(name))
        if hits:
            return hits[0]
    return None


def load_score_dir(folder: str | Path) -> tuple[List[Note], float, str]:
    """Carrega notas (sem fonemas) de uma pasta de música. Retorna (notas, bpm, título)."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"pasta de música não encontrada: {folder}")

    xml = _find(folder, "*.musicxml", "*.xml", "*.mxl")
    if xml is not None:
        notes, bpm = parse_musicxml(xml)
        return notes, bpm, folder.name

    midi = _find(folder, "*.mid", "*.midi")
    if midi is None:
        raise FileNotFoundError(
            f"nenhum .musicxml ou .mid encontrado em {folder}"
        )
    lyr = _find(folder, "lyrics.txt", "*.txt")
    lyrics = read_lyrics_file(lyr) if lyr else None
    notes, bpm = parse_midi(midi, lyrics)
    return notes, bpm, folder.name


def build_performance(folder: str | Path, language: str = "en") -> Performance:
    """Carrega a partitura e preenche os fonemas — pronto para o motor SVS."""
    notes, bpm, title = load_score_dir(folder)
    phonemize_notes(notes)
    return Performance(notes=notes, tempo_bpm=bpm, language=language, title=title)


if __name__ == "__main__":
    import sys

    perf = build_performance(sys.argv[1] if len(sys.argv) > 1 else "data/songs/demo")
    print(perf.summary())
    for n in perf.notes[:12]:
        phs = " ".join(f"{p.symbol}:{p.duration:.2f}" for p in n.phonemes)
        kind = "rest" if n.is_rest else f"pitch={n.midi_pitch}"
        print(f"  t={n.start:6.2f} dur={n.duration:.2f} {kind:12} '{n.lyric}'  [{phs}]")
