"""Transcreve um áudio melódico em uma partitura (score.mid) pronta para o pipeline.

Grave VOCÊ cantando/assobiando/tocando a melodia (mono, uma nota por vez) e rode:

    python scripts/transcribe.py minha_melodia.wav --name minha_musica
    python scripts/transcribe.py melodia.wav --name drivers_license --lyrics letra.txt

Opcionalmente passe a letra (--lyrics, uma sílaba por nota) para já casar com as notas.
O áudio de entrada deve ser seu (ou de domínio público / licença aberta). Veja CONSENT.md.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import paths  # noqa: E402
from src.score import build_performance, validate  # noqa: E402
from src.transcribe import notes_to_midi, transcribe_audio  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", help="arquivo de áudio com a melodia (wav/mp3/flac...)")
    ap.add_argument("--name", default=None, help="nome da música (default: nome do áudio)")
    ap.add_argument("--lyrics", default=None, help="arquivo de letra (uma sílaba por nota)")
    ap.add_argument("--fmin", type=float, default=65.0, help="f0 mínimo em Hz")
    ap.add_argument("--fmax", type=float, default=1047.0, help="f0 máximo em Hz")
    ap.add_argument("--min-note", type=float, default=0.08, help="duração mínima de nota (s)")
    args = ap.parse_args()

    audio = Path(args.audio)
    if not audio.exists():
        raise SystemExit(f"áudio não encontrado: {audio}")

    name = args.name or audio.stem
    dest_dir = paths.SONGS / name
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"[transcribe] estimando pitch (pYIN) em {audio.name} ...")
    notes = transcribe_audio(audio, fmin=args.fmin, fmax=args.fmax,
                             min_note_s=args.min_note)
    print(f"[transcribe] {len(notes)} notas detectadas")

    midi_path = dest_dir / "score.mid"
    notes_to_midi(notes, midi_path)
    print(f"[transcribe] MIDI escrito: {midi_path}")

    if args.lyrics:
        shutil.copy2(args.lyrics, dest_dir / "lyrics.txt")
        print(f"[transcribe] letra copiada: {dest_dir / 'lyrics.txt'}")

    perf = build_performance(dest_dir)
    print(f"[transcribe] {perf.summary()}")
    rep = validate(perf)
    print("[validacao]")
    print(rep.render())
    print(f"\nProximo passo: revise no MuseScore se quiser, depois\n"
          f"  python -m src.pipeline --song {dest_dir} --voice <nome>")


if __name__ == "__main__":
    main()
