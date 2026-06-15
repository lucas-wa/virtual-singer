"""Orquestra a cadeia completa do cantor virtual.

    partitura -> voz-guia (DiffSinger) -> timbre do usuario (RVC) -> [mix] -> [avatar]

Uso:
    python -m src.pipeline --song data/songs/demo --voice meu_nome --out out/demo.wav
    python -m src.pipeline --song data/songs/demo --voice meu_nome \
        --instrumental caminho/instrumental.wav --out out/demo.wav
    python -m src.pipeline --song data/songs/demo --voice meu_nome \
        --avatar-image data/faces/meu_avatar.jpg --out out/demo.wav   # gera out/demo.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import audio, paths
from .hardware import detect_profile
from .score import build_performance
from .svs import synthesize, synthesize_dsp
from .voice import VoiceModel, convert


def run(song_dir: str | Path, voice_name: str, out_wav: str | Path,
        instrumental: str | Path | None = None, transpose: int = 0,
        avatar_image: str | Path | None = None, avatar_motion: bool = False,
        engine: str = "dsp") -> Path:
    """Gera o canto e, se `avatar_image` for dado, também o vídeo (mesmo nome, .mp4).

    engine: "dsp" (sintetizador embutido, padrão — sempre roda) ou "diffsinger"
    (qualidade maior, precisa de voicebank EN + GPU). Retorna .mp4 se houver avatar,
    senão o .wav.
    """
    profile = detect_profile()
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    # 1. partitura -> performance (fonemas alinhados)
    perf = build_performance(song_dir)
    print(f"[pipeline] {perf.summary()} | motor de voz-guia: {engine}")

    # 2. performance -> voz-guia cantada
    guide = out_wav.parent / "_guide.wav"
    if engine == "diffsinger":
        synthesize(perf, guide, profile)
    else:
        synthesize_dsp(perf, guide)

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
    print(f"[pipeline] áudio pronto -> {out_wav}")

    # 5. avatar opcional: foto + áudio -> vídeo
    if avatar_image:
        from .avatar import animate  # import tardio: deps pesadas só quando usado

        out_video = out_wav.with_suffix(".mp4")
        animate(avatar_image, out_wav, out_video, profile=profile, still=not avatar_motion)
        print(f"[pipeline] vídeo pronto -> {out_video}")
        return out_video

    return out_wav


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--song", required=True, help="pasta da música (MIDI/MusicXML + letra)")
    ap.add_argument("--voice", required=True, help="nome do modelo de voz treinado")
    ap.add_argument("--out", default=str(paths.OUT / "output.wav"), help="WAV de saída")
    ap.add_argument("--instrumental", default=None, help="WAV instrumental para a mix (opcional)")
    ap.add_argument("--transpose", type=int, default=0, help="semitons de transposição")
    ap.add_argument("--engine", choices=["dsp", "diffsinger"], default="dsp",
                    help="motor da voz-guia: dsp (padrão, sempre roda) ou diffsinger (GPU+voicebank)")
    ap.add_argument("--avatar-image", default=None,
                    help="foto do rosto (sintético/próprio/consentido) p/ gerar vídeo cantando")
    ap.add_argument("--avatar-motion", action="store_true",
                    help="permitir mais movimento de cabeça (default: estável p/ canto)")
    args = ap.parse_args()
    run(args.song, args.voice, args.out, args.instrumental, args.transpose,
        avatar_image=args.avatar_image, avatar_motion=args.avatar_motion, engine=args.engine)


if __name__ == "__main__":
    main()
