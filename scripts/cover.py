"""Converte as músicas de um cantor (FONTE) para o timbre de outro (ALVO), via Seed-VC.

Ex.: pegar as gravações da Priscila e fazê-las soar na SUA voz (Lucas).
- FONTE: pasta com os áudios já cantados (cada arquivo é uma música).
- ALVO:  pasta com a voz cujo timbre queremos — usamos um clipe curto como referência.
Seed-VC é zero-shot: nada de treino. Use vozes próprias / de voluntários que consentem
(CONSENT.md).

Dica de qualidade: o ideal é que a FONTE seja vocal (a cappella). Se o MP3 tiver
instrumental junto, a conversão pega a mistura — separe os vocais antes (Demucs) p/ melhor
resultado.

Uso:
    python scripts/cover.py --songs-dir cantores/priscila --voice-dir cantores/lucas
    python scripts/cover.py --songs-dir cantores/priscila --voice-dir cantores/lucas \
        --out-dir out/covers --limit 3 --semitone 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import audio, paths  # noqa: E402
from src.hardware import detect_profile  # noqa: E402
from src.voice import convert, resolve_reference  # noqa: E402

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--songs-dir", required=True, help="pasta com as músicas FONTE (ex.: cantores/priscila)")
    ap.add_argument("--voice-dir", required=True, help="pasta com a voz ALVO (ex.: cantores/lucas)")
    ap.add_argument("--out-dir", default=str(paths.OUT / "covers"), help="pasta de saída")
    ap.add_argument("--limit", type=int, default=0, help="processar no máximo N músicas (0 = todas)")
    ap.add_argument("--semitone", type=int, default=0, help="transpor em semitons (ajuste de tessitura)")
    ap.add_argument("--diffusion-steps", type=int, default=30,
                    help="passos de difusão do Seed-VC (mais = melhor/mais lento; ex.: 50-100)")
    ap.add_argument("--no-separate", dest="separate", action="store_false",
                    help="NÃO separar vocais com Demucs (use se a fonte já for a cappella)")
    ap.add_argument("--no-remix", dest="remix", action="store_false",
                    help="não remixar com o instrumental (sai só o vocal convertido)")
    ap.add_argument("--checkpoint", default=None,
                    help="checkpoint Seed-VC fine-tunado (melhor fidelidade; opcional)")
    ap.add_argument("--config", default=None, help="config do checkpoint fine-tunado")
    ap.add_argument("--engine", choices=["seedvc", "yingmusic"], default="seedvc",
                    help="motor de conversão: seedvc (py3.12) | yingmusic (SOTA, py3.10)")
    ap.set_defaults(separate=True, remix=True)
    args = ap.parse_args()

    songs_dir = Path(args.songs_dir)
    voice_dir = Path(args.voice_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not songs_dir.is_dir():
        raise SystemExit(f"pasta de músicas não encontrada: {songs_dir}")

    # 1 clipe da voz alvo serve de referência p/ todas as músicas (zero-shot).
    reference = resolve_reference(str(voice_dir))
    print(f"[cover] voz-alvo (referência): {reference}")

    songs = sorted(p for p in songs_dir.iterdir() if p.suffix.lower() in AUDIO_EXTS)
    if not songs:
        raise SystemExit(f"nenhum áudio em {songs_dir} (extensões: {sorted(AUDIO_EXTS)})")
    if args.limit:
        songs = songs[: args.limit]

    profile = detect_profile()
    target = voice_dir.name
    sep_txt = "com separação Demucs" if args.separate else "sem separação (fonte a cappella)"
    print(f"[cover] {len(songs)} música(s) -> timbre de '{target}' | {sep_txt} | {profile.summary()}")

    work = out_dir / "_work"
    done = []
    for i, song in enumerate(songs, 1):
        out = out_dir / f"{song.stem}__{target}.wav"
        print(f"[cover] ({i}/{len(songs)}) {song.name} -> {out.name}")
        try:
            # 1. separar o vocal (se pedido); o instrumental é guardado p/ a remix
            instrumental = None
            if args.separate:
                from src.separate import separate
                vocals, instrumental = separate(song, work / song.stem)
                source = vocals
            else:
                source = song

            vocal_out = out_dir / f"{song.stem}__{target}_vocal.wav"

            if args.engine == "yingmusic":
                # YingMusic faz o remix internamente (recebe vocal + acompanhamento).
                from src.voice import yingmusic
                yingmusic.convert(source, reference, vocal_out, accompany=None,
                                  profile=profile, diffusion_steps=args.diffusion_steps,
                                  checkpoint=args.checkpoint)   # só o vocal convertido
                if args.separate and args.remix and instrumental:
                    yingmusic.convert(source, reference, out, accompany=instrumental,
                                      profile=profile, diffusion_steps=args.diffusion_steps,
                                      checkpoint=args.checkpoint)  # mix convertida
                else:
                    audio.save_wav(out, *audio.load_wav(vocal_out))
            else:
                # Seed-VC: converte o vocal e remixa via numpy (com o instrumental original).
                converted = work / f"{song.stem}__{target}_voc.wav"
                convert(source, reference, converted, profile=profile, semitone_shift=args.semitone,
                        diffusion_steps=args.diffusion_steps,
                        checkpoint=args.checkpoint, config=args.config)
                voc, sr = audio.load_wav(converted)
                audio.save_wav(vocal_out, voc, sr)        # vocal isolado (p/ avatar)
                if args.separate and args.remix and instrumental:
                    inst, _ = audio.load_wav(instrumental, sr=sr)
                    audio.save_wav(out, audio.mix(voc, inst), sr)
                else:
                    audio.save_wav(out, voc, sr)
            done.append(out)
        except Exception as e:  # noqa: BLE001
            print(f"[cover] FALHOU em {song.name}: {e}")

    print(f"\n[cover] concluído: {len(done)}/{len(songs)} em {out_dir}")


if __name__ == "__main__":
    main()
