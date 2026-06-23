"""Fine-tuna o Seed-VC na voz de um cantor, para melhorar a fidelidade do timbre.

Prepara os dados do jeito que o Seed-VC pede (clipes de 1-30 s, voz LIMPA, sem música)
e roda o fine-tune. Etapas:
  1. (opcional) Demucs para extrair só o vocal de cada arquivo (remove música/ruído);
  2. fatiar em pedaços de <= N segundos (o Seed-VC ignora arquivos > 30 s);
  3. treinar -> checkpoint em third_party/seed-vc/runs/<run-name>/ft_model.pth.

Use voz própria / de voluntário que consente (CONSENT.md). Voz falada limpa serve muito bem.

Uso:
    python scripts/finetune_voice.py --voice-dir cantores/lucas --run-name lucas
    python scripts/finetune_voice.py --voice-dir cantores/lucas --run-name lucas \
        --no-separate --max-steps 1500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import audio, paths  # noqa: E402
from src.hardware import detect_profile  # noqa: E402
from src.voice import finetune  # noqa: E402

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".opus"}


def _slice_clean(voice_dir: Path, out_dir: Path, separate: bool, seg_s: float) -> int:
    """Gera clipes limpos de <= seg_s segundos em out_dir. Retorna quantos clipes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    srcs = sorted(p for p in voice_dir.iterdir() if p.suffix.lower() in AUDIO_EXTS)
    if not srcs:
        raise SystemExit(f"nenhum áudio em {voice_dir}")

    n = 0
    for src in srcs:
        # 1. limpar (Demucs) se pedido -> usa só o vocal
        if separate:
            from src.separate import separate as demucs_sep
            voc, _ = demucs_sep(src, out_dir / "_demucs" / src.stem)
            wav_path = voc
        else:
            wav_path = src

        # 2. fatiar em pedaços de seg_s
        y, sr = audio.load_wav(wav_path)
        step = int(seg_s * sr)
        for i in range(0, len(y), step):
            chunk = y[i:i + step]
            if len(chunk) < int(1.0 * sr):   # descarta < 1 s (Seed-VC ignora)
                continue
            audio.save_wav(out_dir / f"{src.stem}_{i // step:03d}.wav", chunk, sr)
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voice-dir", required=True, help="pasta com a voz do alvo (limpa de preferência)")
    ap.add_argument("--run-name", required=True, help="nome do fine-tune (ex.: lucas)")
    ap.add_argument("--no-separate", dest="separate", action="store_false",
                    help="NÃO passar pelo Demucs (use se a voz já for limpa, sem música)")
    ap.add_argument("--seg-seconds", type=float, default=20.0, help="duração dos clipes (<=30)")
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.set_defaults(separate=True)
    args = ap.parse_args()

    voice_dir = Path(args.voice_dir)
    if not voice_dir.is_dir():
        raise SystemExit(f"pasta não encontrada: {voice_dir}")

    dataset = paths.DATA / "finetune" / args.run_name
    print(f"[ft] preparando dados limpos (separate={args.separate}, seg={args.seg_seconds}s)...")
    n = _slice_clean(voice_dir, dataset, args.separate, args.seg_seconds)
    print(f"[ft] {n} clipe(s) em {dataset}")
    if n == 0:
        raise SystemExit("nenhum clipe válido gerado (>=1 s)")

    ckpt, cfg = finetune(dataset, args.run_name, profile=detect_profile(),
                         max_steps=args.max_steps, batch_size=args.batch_size)
    print(f"\n[ft] pronto. Use no cover com:\n  COVER_CHECKPOINT={ckpt}\n  COVER_CONFIG={cfg}")


if __name__ == "__main__":
    main()
