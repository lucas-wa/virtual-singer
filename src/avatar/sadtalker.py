"""Adapter do SadTalker: foto (rosto) + áudio -> vídeo de cabeça cantando.

Recebe uma imagem e o .wav gerado pelo pipeline e chama o script de inferência do
SadTalker (vendored em third_party/SadTalker). O SadTalker grava um .mp4 com nome
baseado em timestamp no result_dir; localizamos o mais recente e movemos para out_video.

Requer o ambiente do SadTalker (Python 3.10 + GPU + checkpoints). Veja GUIDE.md.

ESCOPO: use rosto sintético/fictício, próprio, ou de voluntário que consente — nunca a
imagem de pessoa real sem consentimento (deepfake visual). Veja CONSENT.md.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .. import paths
from ..hardware import RuntimeProfile, detect_profile


def _newest_mp4(folder: Path) -> Path | None:
    mp4s = list(folder.rglob("*.mp4"))
    if not mp4s:
        return None
    return max(mp4s, key=lambda p: p.stat().st_mtime)


def animate(image_path: str | Path, audio_path: str | Path, out_video: str | Path,
            profile: RuntimeProfile | None = None, still: bool = True,
            preprocess: str = "full", enhancer: str | None = "gfpgan") -> Path:
    """Gera o vídeo do rosto cantando o áudio e devolve o caminho de `out_video`.

    - still=True reduz o movimento de cabeça (mais estável para canto sustentado);
    - preprocess='full' usa o quadro inteiro (melhor para foto de rosto/ombros);
    - enhancer='gfpgan' melhora a nitidez do rosto (None desativa, mais rápido).
    """
    profile = profile or detect_profile()
    # ABSOLUTOS: o SadTalker roda com cwd=repo, caminhos relativos quebrariam.
    image_path = Path(image_path).resolve()
    audio_path = Path(audio_path).resolve()
    out_video = Path(out_video).resolve()
    out_video.parent.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        raise FileNotFoundError(f"imagem do rosto não encontrada: {image_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"áudio não encontrado: {audio_path}")

    infer = paths.SADTALKER_REPO / "inference.py"
    if not infer.exists():
        raise RuntimeError(
            f"SadTalker não encontrado em {infer}. "
            "Rode: python scripts/setup_models.py --only avatar"
        )

    result_dir = out_video.parent / "_avatar_work"
    result_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(infer),
        "--source_image", str(image_path),
        "--driven_audio", str(audio_path),
        "--result_dir", str(result_dir),
        "--checkpoint_dir", str(paths.SADTALKER_CKPT),
        "--preprocess", preprocess,
    ]
    if still:
        cmd.append("--still")
    if enhancer:
        cmd += ["--enhancer", enhancer]
    if profile.device == "cpu":
        cmd.append("--cpu")

    print(f"[avatar] {profile.summary()}")
    print(f"[avatar] $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(paths.SADTALKER_REPO))

    produced = _newest_mp4(result_dir)
    if produced is None:
        raise RuntimeError(f"SadTalker não gerou .mp4 em {result_dir}")
    produced.replace(out_video)
    print(f"[avatar] vídeo -> {out_video}")
    return out_video
