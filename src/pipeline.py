"""Orquestra a cadeia completa do cantor virtual.

    partitura  ──►  voz-guia (DiffSinger)  ──►  timbre do usuário (RVC)  ──►  [mix]

Uso:
    python -m src.pipeline --song data/songs/demo --voice meu_nome --out out/demo.wav
    python -m src.pipeline --song data/songs/demo --voice meu_nome \
        --instrumental caminho/instrumental.wav --out out/demo.wav
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import audio, paths
from .hardware import detect_profile
from .score import build_performance
from .svs import synthesize
from .voice import VoiceModel, convert


def run(song_dir: str | Path, voice_name: str, out_wav: str | Path,
        instrumental: str | Path | None = None, transpose: int = 0) -> Path:
    profile = detect_profile()
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    # 1. partitura -> performance (fonemas alinhados)
    perf = build_performance(song_dir)
    print(f"[pipeline] {perf.summary()}")

    # 2. performance -> voz-guia cantada
    guide = out_wav.parent / "_guide.wav"
    synthesize(perf, guide, profile)

    # 3. voz-guia -> timbre do usuário
    model = VoiceModel.for_name(voice_name)
    converted = out_wav.parent / "_converted.wav"
    convert(guide, model, converted, transpose=transpose, profile=profile)

    # 4. mix opcional com instrumental
    if instrumental:
        voc, sr = audio.load_wav(converted)
        inst, _ = audio.load_wav(instrumental, sr=sr)
        mixed = audio.mix(voc, inst)
        audio.save_wav(out_wav, mixed, sr)
    else:
        voc, sr = audio.load_wav(converted)
        audio.save_wav(out_wav, voc, sr)

    print(f"[pipeline] pronto -> {out_wav}")
    return out_wav


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--song", required=True, help="pasta da música (MIDI/MusicXML + letra)")
    ap.add_argument("--voice", required=True, help="nome do modelo de voz treinado")
    ap.add_argument("--out", default=str(paths.OUT / "output.wav"), help="WAV de saída")
    ap.add_argument("--instrumental", default=None, help="WAV instrumental para a mix (opcional)")
    ap.add_argument("--transpose", type=int, default=0, help="semitons de transposição")
    args = ap.parse_args()
    run(args.song, args.voice, args.out, args.instrumental, args.transpose)


if __name__ == "__main__":
    main()
