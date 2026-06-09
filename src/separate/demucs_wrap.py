"""Wrapper de linha de comando do Demucs.

Dois usos no projeto:
  - extrair o instrumental de uma faixa que o usuário tenha direito de usar (para a mix);
  - limpar amostras de voz do usuário (remover música/ruído de fundo) antes de treinar.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Tuple


def separate(input_path: str | Path, out_dir: str | Path,
             model: str = "htdemucs", two_stems: str | None = "vocals") -> Tuple[Path, Path]:
    """Separa `input_path` em (vocais, acompanhamento).

    Com `two_stems='vocals'`, o Demucs produz 'vocals.wav' e 'no_vocals.wav'.
    Retorna (caminho_vocais, caminho_instrumental).
    """
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, "-m", "demucs", "-n", model, "-o", str(out_dir)]
    if two_stems:
        cmd += ["--two-stems", two_stems]
    cmd.append(str(input_path))

    print(f"[Demucs] $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    stem_dir = out_dir / model / input_path.stem
    vocals = stem_dir / "vocals.wav"
    instrumental = stem_dir / "no_vocals.wav"
    return vocals, instrumental
