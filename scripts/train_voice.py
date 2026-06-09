"""Treina o modelo de timbre (RVC) a partir das gravações de um usuário.

Antes de rodar, coloque >= 5 min de áudio limpo (voz solo) em data/voices/<nome>/.
Opcionalmente, use --clean para passar as amostras pelo Demucs e remover ruído/música.

Uso:
    python scripts/train_voice.py --voice data/voices/meu_nome
    python scripts/train_voice.py --voice data/voices/meu_nome --clean --epochs 150
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import paths  # noqa: E402
from src.hardware import detect_profile  # noqa: E402
from src.voice import train_voice  # noqa: E402


def _clean_samples(voice_dir: Path) -> Path:
    """Passa cada amostra pelo Demucs e devolve uma pasta só com os vocais limpos."""
    from src.separate import separate

    cleaned = voice_dir / "_cleaned"
    cleaned.mkdir(exist_ok=True)
    for wav in list(voice_dir.glob("*.wav")) + list(voice_dir.glob("*.mp3")):
        if wav.parent.name == "_cleaned":
            continue
        vocals, _ = separate(wav, voice_dir / "_demucs")
        target = cleaned / f"{wav.stem}.wav"
        Path(vocals).replace(target)
    return cleaned


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voice", required=True, help="pasta com as gravações do usuário")
    ap.add_argument("--name", default=None, help="nome do modelo (default: nome da pasta)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--clean", action="store_true", help="limpar amostras com Demucs antes")
    args = ap.parse_args()

    voice_dir = Path(args.voice)
    if not voice_dir.is_dir():
        raise SystemExit(f"pasta não encontrada: {voice_dir}")

    profile = detect_profile()
    print(f"[train] {profile.summary()}")

    dataset = _clean_samples(voice_dir) if args.clean else voice_dir
    model = train_voice(dataset, name=args.name or voice_dir.name,
                        epochs=args.epochs, profile=profile)
    print(f"[train] modelo salvo: {model.weight}")


if __name__ == "__main__":
    main()
