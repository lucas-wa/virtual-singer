"""Treino e inferência RVC (Retrieval-based Voice Conversion).

RVC aprende o *timbre* do usuário a partir de poucos minutos de áudio e converte
qualquer sinal de canto para esse timbre. Aqui encapsulamos:
  - train_voice(): preprocess + extração de features + treino + índice FAISS;
  - convert():     aplica o modelo treinado à voz-guia do DiffSinger.

Encapsula os scripts do repositório RVC (vendored em third_party/rvc).
Requer ambiente RVC (Python 3.10 + GPU). Veja README.md.
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .. import paths
from ..hardware import RuntimeProfile, detect_profile


@dataclass
class VoiceModel:
    name: str
    weight: Path   # .pth treinado (timbre)
    index: Path    # índice FAISS (retrieval de features)

    @classmethod
    def for_name(cls, name: str) -> "VoiceModel":
        d = paths.MODELS / "voices" / name
        return cls(name=name, weight=d / "model.pth", index=d / "added.index")

    def exists(self) -> bool:
        return self.weight.exists()


def _rvc_script(*parts: str) -> Path:
    return paths.RVC_REPO.joinpath(*parts)


def train_voice(voice_dir: str | Path, name: str | None = None,
                epochs: int = 100, profile: RuntimeProfile | None = None) -> VoiceModel:
    """Treina um modelo de timbre a partir das gravações em `voice_dir`."""
    profile = profile or detect_profile()
    voice_dir = Path(voice_dir)
    name = name or voice_dir.name
    model = VoiceModel.for_name(name)
    model.weight.parent.mkdir(parents=True, exist_ok=True)

    train = _rvc_script("train.py")
    if not train.exists():
        raise RuntimeError(
            f"RVC não encontrado em {train}. Rode: python scripts/setup_models.py --only repos"
        )

    # batch ajustado pela VRAM detectada
    cmd = [
        sys.executable, str(train),
        "--exp", name,
        "--dataset", str(voice_dir),
        "--epochs", str(epochs),
        "--batch", str(profile.batch_size),
        "--fp16", "1" if profile.use_fp16 else "0",
        "--sr", "40000",
    ]
    print(f"[RVC train] {profile.summary()}")
    print(f"[RVC train] $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(paths.RVC_REPO))
    return model


def convert(input_wav: str | Path, model: VoiceModel, out_wav: str | Path,
            transpose: int = 0, profile: RuntimeProfile | None = None) -> Path:
    """Converte `input_wav` (voz-guia) para o timbre de `model`."""
    profile = profile or detect_profile()
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    if not model.exists():
        raise RuntimeError(f"modelo de voz '{model.name}' não treinado ({model.weight})")

    infer = _rvc_script("infer.py")
    cmd = [
        sys.executable, str(infer),
        "--model", str(model.weight),
        "--index", str(model.index),
        "--input", str(input_wav),
        "--output", str(out_wav),
        "--transpose", str(transpose),
        "--f0method", "rmvpe",
    ]
    print(f"[RVC infer] $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(paths.RVC_REPO))
    return out_wav
