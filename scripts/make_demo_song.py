"""Gera a música de demonstração (domínio público) em data/songs/demo/.

'Twinkle Twinkle Little Star' — melodia e letra de domínio público. Serve só para
exercitar o pipeline ponta a ponta; troque por qualquer partitura sua.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import paths  # noqa: E402

# (pitch MIDI, batidas) — C maior, 120 bpm
MELODY = [
    (60, 1), (60, 1), (67, 1), (67, 1), (69, 1), (69, 1), (67, 2),  # Twinkle twinkle little star
    (65, 1), (65, 1), (64, 1), (64, 1), (62, 1), (62, 1), (60, 2),  # how I wonder what you are
]
SYLLABLES = [
    "Twin", "kle", "twin", "kle", "lit", "tle", "star",
    "how", "I", "won", "der", "what", "you", "are",
]
TEMPO_BPM = 120


def main() -> None:
    import pretty_midi

    demo_dir = paths.SONGS / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)

    pm = pretty_midi.PrettyMIDI(initial_tempo=TEMPO_BPM)
    inst = pretty_midi.Instrument(program=0)  # voz/piano
    spb = 60.0 / TEMPO_BPM
    t = 0.0
    for pitch, beats in MELODY:
        dur = beats * spb
        inst.notes.append(
            pretty_midi.Note(velocity=90, pitch=pitch, start=t, end=t + dur)
        )
        t += dur
    pm.instruments.append(inst)

    midi_path = demo_dir / "score.mid"
    pm.write(str(midi_path))

    lyrics_path = demo_dir / "lyrics.txt"
    lyrics_path.write_text(
        "# Twinkle Twinkle Little Star (domínio público)\n"
        + " ".join(SYLLABLES) + "\n",
        encoding="utf-8",
    )

    print(f"Demo criada:\n  {midi_path}\n  {lyrics_path}")
    print(f"{len(MELODY)} notas, {len(SYLLABLES)} sílabas.")


if __name__ == "__main__":
    main()
