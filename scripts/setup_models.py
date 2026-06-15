"""Baixa os pesos pré-treinados necessários para o pipeline.

Componentes:
  1. RVC base    — HuBERT/ContentVec (features de conteúdo) + RMVPE (extração de F0).
  2. NSF-HiFiGAN — vocoder neural (mel -> forma de onda).
  3. DiffSinger  — modelo acústico SVS em inglês (clonado como repositório + checkpoint).

Os repositórios DiffSinger e RVC não são pacotes PyPI limpos, então clonamos via git e
baixamos os checkpoints do Hugging Face Hub. Os IDs abaixo podem ser ajustados se você
preferir outra fonte/versão.

Uso:
    python scripts/setup_models.py             # só ÁUDIO (seguro no Colab grátis)
    python scripts/setup_models.py --with-avatar  # inclui SadTalker (NÃO no Colab grátis)
    python scripts/setup_models.py --only rvc  # baixa só um componente
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

# NSF-HiFiGAN (vocoder) — release do OpenVPI (zip do GitHub, extraído em models/nsf_hifigan).
VOCODER_ZIP_URL = ("https://github.com/openvpi/vocoders/releases/download/"
                   "nsf-hifigan-44.1k-hop512-128bin-2024.02/"
                   "nsf_hifigan_44.1k_hop512_128bin_2024.02.zip")

# Repositórios para clonar
# Repos de ÁUDIO (permitidos no Colab grátis). O SadTalker (avatar) NÃO entra aqui de
# propósito: o Colab grátis proíbe ferramentas de animação facial/deepfake e encerra a
# sessão ao clonar/rodar o SadTalker. Ele só é baixado pelo componente "avatar" (opt-in),
# que deve rodar localmente ou num host que permita (ver GUIDE.md).
REPOS = [
    ("https://github.com/openvpi/DiffSinger.git", paths.DIFFSINGER_REPO),
    ("https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git", paths.RVC_REPO),
]
SADTALKER_REPO_URL = "https://github.com/OpenTalker/SadTalker.git"

# Checkpoint acústico do DiffSinger em inglês (modelo da comunidade / OpenUtau).
# Ajuste o repo_id para o modelo EN que você escolher usar.
DIFFSINGER_EN = ("AnnaWegmann/diffsinger-en-placeholder", "model.ckpt",
                 paths.DIFFSINGER_DIR / "model.ckpt")

# Checkpoints do SadTalker (avatar) — repo HF correto: vinthony/SadTalker-V002rc.
# Os pesos do GFPGAN/face-detection são baixados automaticamente pelo SadTalker em runtime.
_SADTALKER_REPO_ID = "vinthony/SadTalker-V002rc"
SADTALKER_FILES = [
    (_SADTALKER_REPO_ID, "SadTalker_V0.0.2_256.safetensors",
     paths.SADTALKER_CKPT / "SadTalker_V0.0.2_256.safetensors"),
    (_SADTALKER_REPO_ID, "SadTalker_V0.0.2_512.safetensors",
     paths.SADTALKER_CKPT / "SadTalker_V0.0.2_512.safetensors"),
    (_SADTALKER_REPO_ID, "mapping_00109-model.pth.tar",
     paths.SADTALKER_CKPT / "mapping_00109-model.pth.tar"),
    (_SADTALKER_REPO_ID, "mapping_00229-model.pth.tar",
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


def _download_and_extract_zip(url: str, dest_dir: Path) -> None:
    import io
    import zipfile

    import requests

    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ↓ {url}")
    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            z.extractall(dest_dir)
        print(f"    → extraído em {dest_dir}")
    except Exception as e:  # noqa: BLE001
        print(f"    ! falhou ({e}). Baixe manualmente e extraia em {dest_dir}")


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
    print("[NSF-HiFiGAN] vocoder (zip do GitHub release)")
    _download_and_extract_zip(VOCODER_ZIP_URL, paths.VOCODER_DIR)


def setup_diffsinger() -> None:
    print("[DiffSinger] checkpoint acústico EN")
    _hf_download(*DIFFSINGER_EN)


def setup_repos() -> None:
    print("[repos] clonando DiffSinger e RVC (áudio)")
    for url, dest in REPOS:
        _clone(url, dest)


def setup_avatar() -> None:
    print("[SadTalker] AVISO: NÃO rode isto no Colab grátis (proíbe face-animation).")
    print("[SadTalker] clonando repo + checkpoints do avatar")
    _clone(SADTALKER_REPO_URL, paths.SADTALKER_REPO)
    for repo, fname, dest in SADTALKER_FILES:
        _hf_download(repo, fname, dest)


COMPONENTS = {
    "repos": setup_repos,
    "rvc": setup_rvc,
    "vocoder": setup_vocoder,
    "diffsinger": setup_diffsinger,
    "avatar": setup_avatar,
}
# Componentes do caminho PADRÃO (motor de voz-guia = DSP embutido): só RVC.
# DiffSinger/NSF-HiFiGAN só são necessários para o motor "diffsinger" (opt-in), e o
# avatar (SadTalker) é proibido no Colab grátis — ambos ficam de fora por padrão.
AUDIO_COMPONENTS = ["repos", "rvc"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=list(COMPONENTS), help="baixar apenas um componente")
    ap.add_argument("--with-diffsinger", action="store_true",
                    help="incluir vocoder + checkpoint DiffSinger EN (motor de maior "
                         "qualidade; precisa de voicebank EN e GPU). Padrão usa o motor DSP.")
    ap.add_argument("--with-avatar", action="store_true",
                    help="incluir o SadTalker (avatar). NÃO use no Colab grátis — "
                         "ele proíbe face-animation e encerra a sessão. Rode local/host permitido.")
    args = ap.parse_args()

    paths.ensure_dirs()
    if args.only:
        targets = [args.only]
    else:
        targets = list(AUDIO_COMPONENTS)
        if args.with_diffsinger:
            targets += ["vocoder", "diffsinger"]
        if args.with_avatar:
            targets.append("avatar")
    for name in targets:
        COMPONENTS[name]()
    print("\nConcluído. O motor PADRÃO (DSP) já funciona só com o RVC baixado.")
    if "diffsinger" not in targets:
        print("DiffSinger NÃO baixado (motor opcional). Use --with-diffsinger se quiser tentá-lo.")
    if "avatar" not in targets:
        print("Avatar (SadTalker) NÃO baixado. Rode com --with-avatar FORA do Colab grátis.")


if __name__ == "__main__":
    main()
