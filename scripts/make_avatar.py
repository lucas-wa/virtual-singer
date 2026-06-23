"""Gera o vídeo do avatar (SadTalker) a partir de um áudio JÁ sintetizado + uma foto.

Roda separado do pipeline de áudio porque o SadTalker exige Python 3.10 (no py3.12 ele
nem instala). Use rosto PRÓPRIO / sintético / de voluntário que consente — nunca a imagem
de uma pessoa real sem consentimento (deepfake visual). Veja CONSENT.md.

Uso:
    python scripts/make_avatar.py --image data/faces/meu_rosto.jpg \
        --audio out/vocalset_female1_demo.wav --out out/meu_avatar.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import paths  # noqa: E402
from src.avatar import animate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="foto do rosto (sua/sintética/consentida)")
    ap.add_argument("--audio", required=True, help="áudio cantado já gerado (.wav)")
    ap.add_argument("--out", default=str(paths.OUT / "avatar.mp4"), help="vídeo de saída (.mp4)")
    ap.add_argument("--motion", action="store_true",
                    help="permitir mais movimento de cabeça (default: estável p/ canto)")
    args = ap.parse_args()

    out = animate(args.image, args.audio, args.out, still=not args.motion)
    print(f"[make_avatar] vídeo pronto -> {out}")


if __name__ == "__main__":
    main()
