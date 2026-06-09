"""Converte uma `Performance` para o formato de entrada do DiffSinger (.ds).

O DiffSinger (openvpi) consome um JSON com segmentos contendo:
  ph_seq   — fonemas separados por espaço
  ph_dur   — duração de cada fonema (s)
  note_seq — nota de cada *fonema* (nome, ex. C4; "rest" em pausas)
  note_dur — duração de cada nota (s)  [aqui, por fonema, espelhando ph_dur]
  ph_num   — quantos fonemas por nota

Esta conversão é determinística e testável sem GPU.
"""
from __future__ import annotations

from typing import List

from ..score.model import Performance

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_note_name(midi_pitch: int) -> str:
    """60 -> 'C4'. Pitches <= 0 viram 'rest'."""
    if midi_pitch <= 0:
        return "rest"
    octave = midi_pitch // 12 - 1
    return f"{_NOTE_NAMES[midi_pitch % 12]}{octave}"


def performance_to_ds(perf: Performance) -> List[dict]:
    """Gera um único segmento .ds cobrindo a performance inteira."""
    ph_seq: List[str] = []
    ph_dur: List[float] = []
    note_seq: List[str] = []
    note_dur: List[float] = []
    ph_num: List[int] = []
    text_tokens: List[str] = []

    for note in perf.notes:
        name = midi_to_note_name(note.midi_pitch)
        if note.lyric.strip():
            text_tokens.append(note.lyric.replace("-", ""))
        for ph in note.phonemes:
            ph_seq.append(ph.symbol)
            ph_dur.append(round(ph.duration, 4))
            note_seq.append(name)
            note_dur.append(round(ph.duration, 4))
        ph_num.append(len(note.phonemes))

    def join(xs):
        return " ".join(str(x) for x in xs)

    return [{
        "offset": 0.0,
        "text": " ".join(text_tokens),
        "ph_seq": join(ph_seq),
        "ph_dur": join(ph_dur),
        "note_seq": join(note_seq),
        "note_dur": join(note_dur),
        "ph_num": join(ph_num),
        "input_type": "phoneme",
    }]


if __name__ == "__main__":
    import json
    import sys

    from ..score import build_performance

    perf = build_performance(sys.argv[1] if len(sys.argv) > 1 else "data/songs/demo")
    print(json.dumps(performance_to_ds(perf), ensure_ascii=False, indent=2))
