"""Baixa os pesos pré-treinados necessários para o pipeline.

Componentes:
  1. RVC base    — HuBERT/ContentVec (features de conteúdo) + RMVPE (extração de F0).
  2. NSF-HiFiGAN — vocoder neural (mel -> forma de onda).
  3. DiffSinger  — modelo acústico SVS em inglês (clonado como repositório + checkpoint).

Os repositórios DiffSinger e RVC não são pacotes PyPI limpos, então clonamos via git e
baixamos os checkpoints do Hugging Face Hub. Os IDs abaixo podem ser ajustados se você
preferir outra fonte/versão.

Uso:
    python scripts/setup_models.py            # baixa tudo
    python scripts/setup_models.py --only rvc # baixa só um componente
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import paths  # noqa: E402

# --- Fontes de pesos (Hugging Face Hub) ----------------------------------------
# (repo_id, arquivo_no_repo, destino_local)
RVC_FILES = [
    ("lj1995/VoiceConversionWebUI", "hubert_base.pt", paths.RVC_BASE_DIR / "hubert_base.pt"),
    ("lj1995/VoiceConversionWebUI", "rmvpe.pt", paths.RVC_BASE_DIR / "rmvpe.pt"),
]

# NSF-HiFiGAN (vocoder) — release do OpenVPI. Baixe o zip e extraia em models/nsf_hifigan.
VOCODER_HF = ("openvpi/vocoders", "nsf_hifigan_44.1k_hop512_128bin_2024.02/model.ckpt",
              paths.VOCODER_DIR / "model.ckpt")
VOCODER_CONFIG = ("openvpi/vocoders", "nsf_hifigan_44.1k_hop512_128bin_2024.02/config.yaml",
                  paths.VOCODER_DIR / "config.yaml")

# Repositórios para clonar
REPOS = [
    ("https://github.com/openvpi/DiffSinger.git", paths.DIFFSINGER_REPO),
    ("https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git", paths.RVC_REPO),
    ("https://github.com/OpenTalker/SadTalker.git", paths.SADTALKER_REPO),
]

# Checkpoint acústico do DiffSinger em inglês (modelo da comunidade / OpenUtau).
# Ajuste o repo_id para o modelo EN que você escolher usar.
DIFFSINGER_EN = ("AnnaWegmann/diffsinger-en-placeholder", "model.ckpt",
                 paths.DIFFSINGER_DIR / "model.ckpt")

# Checkpoints do SadTalker (avatar). Disponíveis no Hugging Face (vinthony/SadTalker).
# Os pesos do GFPGAN são baixados automaticamente pelo SadTalker em tempo de execução.
SADTALKER_FILES = [
    ("vinthony/SadTalker", "SadTalker_V0.0.2_256.safetensors",
     paths.SADTALKER_CKPT / "SadTalker_V0.0.2_256.safetensors"),
    ("vinthony/SadTalker", "SadTalker_V0.0.2_512.safetensors",
     paths.SADTALKER_CKPT / "SadTalker_V0.0.2_512.safetensors"),
    ("vinthony/SadTalker", "mapping_00109-model.pth.tar",
     paths.SADTALKER_CKPT / "mapping_00109-model.pth.tar"),
    ("vinthony/SadTalker", "mapping_00229-model.pth.tar",
     paths.SADTALKER_CKPT / "mapping_00229-model.pth.tar"),
]


def _hf_download(repo_id: str, filename: str, dest: Path) -> None:
    from huggingface_hub import hf_hub_download

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  ↓ {repo_id}/{filename}")
    try:
        cached = hf_hub_download(repo_id=repo_id, filename=filename)
        # copia para destino determinístico do projeto
        import shutil

        shutil.copy2(cached, dest)
        print(f"    → {dest}")
    except Exception as e:  # noqa: BLE001
        print(f"    ! falhou ({e}). Baixe manualmente e coloque em {dest}")


def _clone(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  = já existe: {dest}")
        return
    print(f"  ⧉ git clone {url}")
    try:
        subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"    ! clone falhou ({e}). Clone manualmente em {dest}")


def setup_rvc() -> None:
    print("[RVC] features base + extrator de F0")
    for repo, fname, dest in RVC_FILES:
        _hf_download(repo, fname, dest)


def setup_vocoder() -> None:
    print("[NSF-HiFiGAN] vocoder")
    for repo, fname, dest in (VOCODER_HF, VOCODER_CONFIG):
        _hf_download(repo, fname, dest)


def setup_diffsinger() -> None:
    print("[DiffSinger] checkpoint acústico EN")
    _hf_download(*DIFFSINGER_EN)


def setup_repos() -> None:
    print("[repos] clonando DiffSinger, RVC e SadTalker")
    for url, dest in REPOS:
        _clone(url, dest)


def setup_avatar() -> None:
    print("[SadTalker] checkpoints do avatar")
    for repo, fname, dest in SADTALKER_FILES:
        _hf_download(repo, fname, dest)


COMPONENTS = {
    "repos": setup_repos,
    "rvc": setup_rvc,
    "vocoder": setup_vocoder,
    "diffsinger": setup_diffsinger,
    "avatar": setup_avatar,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=list(COMPONENTS), help="baixar apenas um componente")
    args = ap.parse_args()

    paths.ensure_dirs()
    targets = [args.only] if args.only else list(COMPONENTS)
    for name in targets:
        COMPONENTS[name]()
    print("\nConcluído. Verifique avisos acima para downloads que precisem de ação manual.")


if __name__ == "__main__":
    main()
