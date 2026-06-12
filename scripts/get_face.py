"""Baixa um rosto SINTÉTICO (gerado por GAN, não corresponde a pessoa real) para o avatar.

Fonte: thispersondoesnotexist.com — cada acesso devolve um rosto inteiramente gerado por
IA, que não é de nenhuma pessoa real. Ideal para um "cantor virtual" fictício.

Uso:
    python scripts/get_face.py --name meu_avatar

Alternativas igualmente válidas (sem este script):
  - usar a SUA própria foto;
  - usar a foto de um voluntário que consente (ver CONSENT.md);
  - desenhar/gerar um personagem ilustrado.
NUNCA use a imagem de uma pessoa real sem consentimento.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import paths  # noqa: E402

URL = "https://thispersondoesnotexist.com/"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="synthetic_face", help="nome do arquivo (sem extensão)")
    args = ap.parse_args()

    import requests

    paths.FACES.mkdir(parents=True, exist_ok=True)
    dest = paths.FACES / f"{args.name}.jpg"

    print(f"[get_face] baixando rosto sintético de {URL}")
    resp = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    print(f"[get_face] salvo: {dest}")
    print("Lembrete: este rosto é gerado por IA e não é de uma pessoa real.")


if __name__ == "__main__":
    main()
