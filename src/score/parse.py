"""Parse de partituras: MIDI (pretty_midi) e MusicXML (music21).

Ambos os formatos produzem uma lista de `Note` SEM fonemas ainda — a sílaba (`lyric`)
vem embutida no MusicXML ou é casada a partir de um arquivo de letra externo no MIDI.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .model import Note


def parse_musicxml(path: Path) -> tuple[List[Note], float]:
    """Lê MusicXML; usa as sílabas (lyrics) embutidas quando presentes."""
    import music21 as m21

    score = m21.converter.parse(str(path))
    tempo_bpm = 120.0
    mm = score.flatten().getElementsByClass(m21.tempo.MetronomeMark)
    if mm:
        tempo_bpm = float(mm[0].number or 120.0)

    spb = 60.0 / tempo_bpm  # segundos por batida (quarter note)
    notes: List[Note] = []
    part = score.parts[0] if score.parts else score
    for el in part.flatten().notesAndRests:
        start = float(el.offset) * spb
        dur = float(el.duration.quarterLength) * spb
        if isinstance(el, m21.note.Rest):
            notes.append(Note(start, dur, 0, ""))
            continue
        # acorde -> usa a nota mais aguda (melodia)
        pitch = el.pitches[-1].midi if el.isChord else el.pitch.midi
        lyric = el.lyric or ""
        notes.append(Note(start, dur, int(pitch), lyric))
    return notes, tempo_bpm


def parse_midi(path: Path, lyrics: Optional[List[str]] = None) -> tuple[List[Note], float]:
    """Lê MIDI monofônico (melodia). Casa `lyrics` (uma sílaba por nota) na ordem.

    Se houver eventos de letra no próprio MIDI, eles têm prioridade.
    """
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(path))
    tempo_bpm = 120.0
    try:
        _, tempi = pm.get_tempo_changes()
        if len(tempi):
            tempo_bpm = float(tempi[0])
    except Exception:  # noqa: BLE001
        pass

    # Letras embutidas no MIDI (eventos lyric), se existirem.
    embedded = [ly.text for ly in getattr(pm, "lyrics", [])]

    # Pega o primeiro instrumento não-percussivo como melodia.
    inst = next((i for i in pm.instruments if not i.is_drum), None)
    midi_notes = sorted(inst.notes, key=lambda n: n.start) if inst else []

    syllables = embedded or lyrics or []
    notes: List[Note] = []
    for idx, mn in enumerate(midi_notes):
        lyric = syllables[idx] if idx < len(syllables) else ""
        notes.append(Note(mn.start, mn.end - mn.start, int(mn.pitch), lyric))
    return notes, tempo_bpm


def read_lyrics_file(path: Path) -> List[str]:
    """Lê um arquivo de letra e devolve a lista de sílabas/tokens, na ordem.

    Convenção: separe sílabas por espaço e use '-' para juntar sílabas de uma palavra.
    Linhas iniciadas por '#' são comentários. Ex.:
        I'm so- rry
        for the things I said
    """
    tokens: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tokens.extend(line.split())
    return tokens
