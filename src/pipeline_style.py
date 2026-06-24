"""Cascade: nova melodia + transferência de estilo (canto + gênero) + timbre.

    partitura ALVO ─► [estilo de canto: TCSinger] ─► [timbre: Seed-VC] ─► vocal
                                                                            │
                  melodia ─► [gênero: MusicGen] ─► acompanhamento ──────────┤
                                                                            ▼
                                                                  [mix] ─► .wav final

Cada estágio é opcional (flags) para isolar/depurar. Tudo sobre material legítimo:
melodia/letra suas, referência de estilo = gravação de vocês, timbre = voz consentida,
gênero = prompt de texto.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import audio, paths
from .hardware import detect_profile
from .score import build_performance


def run(target_dir: str | Path, voice_name: str, out_wav: str | Path,
        style_ref_dir: str | Path | None = None, style_wav: str | Path | None = None,
        genre: str | None = None, do_style: bool = True, do_timbre: bool = True,
        do_genre: bool = True) -> Path:
    profile = detect_profile()
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    target_perf = build_performance(target_dir)
    print(f"[style] alvo: {target_perf.summary()}")

    # 1. Estilo de canto (TCSinger) — ou voz-guia DSP se pular o estilo.
    if do_style:
        if not (style_ref_dir and style_wav):
            raise SystemExit("estilo ligado exige --style-ref (partitura) e --style-wav (gravação)")
        from .style import style_transfer
        style_perf = build_performance(style_ref_dir)
        vocal = out_wav.parent / "_styled.wav"
        style_transfer(target_perf, style_perf, style_wav, vocal, profile)
    else:
        from .svs import synthesize_dsp
        vocal = out_wav.parent / "_guide.wav"
        synthesize_dsp(target_perf, vocal)

    # 2. Timbre (Seed-VC zero-shot) — impõe a voz alvo.
    if do_timbre:
        from .voice import convert, resolve_reference
        reference = resolve_reference(voice_name)
        retimbred = out_wav.parent / "_retimbred.wav"
        convert(vocal, reference, retimbred, profile=profile)
        vocal = retimbred

    # 3. Gênero/arranjo (MusicGen) + mix — ou só o vocal.
    if do_genre and genre:
        from .arrange import generate_accompaniment
        accomp = generate_accompaniment(vocal, out_wav.parent / "_accomp.wav", genre, profile)
        voc, sr = audio.load_wav(vocal)
        inst, _ = audio.load_wav(accomp, sr=sr)
        audio.save_wav(out_wav, audio.mix(voc, inst), sr)
    else:
        voc, sr = audio.load_wav(vocal)
        audio.save_wav(out_wav, voc, sr)

    print(f"[style] pronto -> {out_wav}")
    return out_wav


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True, help="pasta da partitura ALVO (melodia+letra)")
    ap.add_argument("--voice", required=True, help="voz de timbre (pasta/clip de referência)")
    ap.add_argument("--out", default=str(paths.OUT / "style_out.wav"))
    ap.add_argument("--style-ref", default=None, help="pasta da partitura da REFERÊNCIA de estilo")
    ap.add_argument("--style-wav", default=None, help="gravação de vocês (áudio de estilo, 48k)")
    ap.add_argument("--genre", default=None, help="prompt de gênero p/ o acompanhamento")
    ap.add_argument("--no-style", dest="do_style", action="store_false")
    ap.add_argument("--no-timbre", dest="do_timbre", action="store_false")
    ap.add_argument("--no-genre", dest="do_genre", action="store_false")
    ap.set_defaults(do_style=True, do_timbre=True, do_genre=True)
    args = ap.parse_args()
    run(args.target, args.voice, args.out, style_ref_dir=args.style_ref,
        style_wav=args.style_wav, genre=args.genre, do_style=args.do_style,
        do_timbre=args.do_timbre, do_genre=args.do_genre)


if __name__ == "__main__":
    main()
