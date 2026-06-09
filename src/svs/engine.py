"""Adapter de inferência do DiffSinger.

Recebe uma `Performance`, escreve o arquivo .ds e chama o script de inferência do
repositório DiffSinger (vendored em third_party/DiffSinger), produzindo a voz-guia.

Requer o ambiente do DiffSinger preparado (Python 3.10 + GPU). Veja README.md.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from .. import paths
from ..hardware import RuntimeProfile, detect_profile
from ..score.model import Performance
from .ds_format import performance_to_ds


def _write_ds(perf: Performance, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_path = out_dir / "input.ds"
    ds_path.write_text(
        json.dumps(performance_to_ds(perf), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return ds_path


def synthesize(perf: Performance, out_wav: str | Path,
               profile: RuntimeProfile | None = None) -> Path:
    """Sintetiza a voz-guia cantada para `out_wav` e devolve o caminho.

    Encapsula a chamada ao DiffSinger; o front-end (.ds) é gerado por nós, então a
    parte específica do projeto fica isolada aqui.
    """
    profile = profile or detect_profile()
    out_wav = Path(out_wav)
    work = out_wav.parent / "_svs_work"
    ds_path = _write_ds(perf, work)

    infer = paths.DIFFSINGER_REPO / "scripts" / "infer.py"
    if not infer.exists():
        raise RuntimeError(
            f"DiffSinger não encontrado em {infer}. "
            "Rode: python scripts/setup_models.py --only repos"
        )

    cmd = [
        sys.executable, str(infer), "acoustic", str(ds_path),
        "--exp", str(paths.DIFFSINGER_DIR),
        "--out", str(out_wav.parent),
    ]
    print(f"[SVS] {profile.summary()}")
    print(f"[SVS] $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(paths.DIFFSINGER_REPO))

    produced = out_wav.parent / (ds_path.stem + ".wav")
    if produced.exists() and produced != out_wav:
        produced.replace(out_wav)
    return out_wav
