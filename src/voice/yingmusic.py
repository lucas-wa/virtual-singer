"""Conversão de timbre via YingMusic-SVC — SVC zero-shot SOTA (2026), sobre Seed-VC.

Diferenciais vs Seed-VC base: timbre-shifter treinado em canto + adaptador F0 + RL
(Flow-GRPO), melhor fidelidade de timbre e de agudos. Recebe o **vocal** e, opcionalmente,
o **acompanhamento** (instrumental) e devolve a mix já convertida.

Encapsula o `my_inference.py` do repo GiantAILab/YingMusic-SVC (vendored em third_party/).
EXIGE Python 3.10 (igual ao SadTalker) — rode no container py3.10, não no vllm py3.12.

Ref.: https://github.com/GiantAILab/YingMusic-SVC
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .. import paths
from ..hardware import RuntimeProfile, detect_profile

# Config do YingMusic dentro do repo, e checkpoint baixado em models/yingmusic/.
YING_CONFIG = "configs/YingMusic-SVC.yml"
YING_CKPT = "YingMusic-SVC-full.pt"


def _newest_wav(folder: Path, since: float) -> Path | None:
    wavs = [p for p in folder.rglob("*.wav") if p.stat().st_mtime >= since]
    return max(wavs, key=lambda p: p.stat().st_mtime) if wavs else None


def convert(source_vocal: str | Path, reference: str | Path, out_wav: str | Path,
            accompany: str | Path | None = None, profile: RuntimeProfile | None = None,
            diffusion_steps: int = 100, checkpoint: str | Path | None = None) -> Path:
    """Converte `source_vocal` para o timbre de `reference` (clipe curto).

    Se `accompany` (instrumental) for dado, o YingMusic devolve a MIX completa convertida;
    senão, devolve só o vocal convertido. Caminhos resolvidos p/ absoluto (cwd=repo).
    """
    profile = profile or detect_profile()
    source_vocal = Path(source_vocal).resolve()
    reference = Path(reference).resolve()
    out_wav = Path(out_wav).resolve()
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    infer = paths.YINGMUSIC_REPO / "my_inference.py"
    if not infer.exists():
        raise RuntimeError(
            f"YingMusic-SVC não encontrado em {infer}. "
            "Rode: python scripts/setup_models.py --with-yingmusic"
        )
    ckpt = Path(checkpoint).resolve() if checkpoint else (paths.YINGMUSIC_CKPT / YING_CKPT)
    if not ckpt.exists():
        raise RuntimeError(f"checkpoint do YingMusic não encontrado: {ckpt}")

    expname = f"vsinger_{out_wav.stem}"
    import time
    since = time.time()
    cmd = [
        sys.executable, str(infer),
        "--source", str(source_vocal),
        "--target", str(reference),
        "--diffusion-steps", str(diffusion_steps),
        "--checkpoint", str(ckpt),
        "--expname", expname,
        "--cuda", "0",
        "--fp16", "True" if profile.use_fp16 else "False",
        "--config", YING_CONFIG,
    ]
    if accompany:
        cmd += ["--accompany", str(Path(accompany).resolve())]

    print(f"[yingmusic] {profile.summary()} | steps={diffusion_steps} | mix={bool(accompany)}")
    print(f"[yingmusic] $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(paths.YINGMUSIC_REPO))

    # O my_inference.py escreve em uma pasta do repo (results/<expname>...); pega o mais novo.
    produced = _newest_wav(paths.YINGMUSIC_REPO, since)
    if produced is None:
        raise RuntimeError(f"YingMusic não gerou .wav novo em {paths.YINGMUSIC_REPO}")
    produced.replace(out_wav)
    print(f"[yingmusic] convertido -> {out_wav}")
    return out_wav
