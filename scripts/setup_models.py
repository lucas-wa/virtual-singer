"""Baixa os pesos pré-treinados necessários para o pipeline.

Componentes:
  1. RVC base    — HuBERT/ContentVec (features de conteúdo) + RMVPE (extração de F0).
  2. NSF-HiFiGAN — vocoder neural (mel -> forma de onda).
  3. DiffSinger  — modelo acústico SVS em inglês (clonado como repositório + checkpoint).

Os repositórios DiffSinger e RVC não são pacotes PyPI limpos, então clonamos via git e
baixamos os checkpoints do Hugging Face Hub. Os IDs abaixo podem ser ajustados se você
preferir outra fonte/versão.

Uso:
    python scripts/setup_models.py               # PADRÃO: Seed-VC (timbre zero-shot)
    python scripts/setup_models.py --with-rvc    # inclui RVC (treino por voz; py3.10)
    python scripts/setup_models.py --with-diffsinger  # inclui SVS DiffSinger (maior qualidade)
    python scripts/setup_models.py --with-avatar # inclui SadTalker (NÃO no Colab grátis)
    python scripts/setup_models.py --only seedvc # baixa só um componente
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
# Repo PADRÃO de conversão de timbre: Seed-VC (zero-shot, sem treino, sem fairseq).
SEEDVC_REPO_URL = "https://github.com/Plachtaa/seed-vc.git"
# Repos OPCIONAIS (motores avançados). RVC = timbre por treino; DiffSinger = SVS de maior
# qualidade. SadTalker (avatar) é proibido no Colab grátis (face-animation), opt-in.
RVC_REPO_URL = "https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git"
DIFFSINGER_REPO_URL = "https://github.com/openvpi/DiffSinger.git"
SADTALKER_REPO_URL = "https://github.com/OpenTalker/SadTalker.git"
YINGMUSIC_REPO_URL = "https://github.com/GiantAILab/YingMusic-SVC.git"
# Checkpoints do YingMusic-SVC (HF). Modelo final RL + separador de acompanhamento.
YINGMUSIC_FILES = [
    ("GiantAILab/YingMusic-SVC", "YingMusic-SVC-full.pt",
     paths.YINGMUSIC_CKPT / "YingMusic-SVC-full.pt"),
    ("GiantAILab/YingMusic-SVC", "bs_roformer.ckpt",
     paths.YINGMUSIC_CKPT / "bs_roformer.ckpt"),
]

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


def setup_seedvc() -> None:
    print("[Seed-VC] clonando repo (conversão de timbre zero-shot — padrão)")
    _clone(SEEDVC_REPO_URL, paths.SEEDVC_REPO)
    print("[Seed-VC] checkpoints baixam sozinhos na 1ª inferência")


def setup_rvc() -> None:
    print("[RVC] repo + features base (opcional, treino por voz)")
    _clone(RVC_REPO_URL, paths.RVC_REPO)
    for repo, fname, dest in RVC_FILES:
        _hf_download(repo, fname, dest)


def setup_vocoder() -> None:
    print("[NSF-HiFiGAN] vocoder (zip do GitHub release)")
    _download_and_extract_zip(VOCODER_ZIP_URL, paths.VOCODER_DIR)


def setup_diffsinger() -> None:
    print("[DiffSinger] repo + checkpoint acústico EN (opcional)")
    _clone(DIFFSINGER_REPO_URL, paths.DIFFSINGER_REPO)
    _hf_download(*DIFFSINGER_EN)


def setup_yingmusic() -> None:
    print("[YingMusic-SVC] clonando repo + checkpoints (SVC SOTA; precisa py3.10)")
    _clone(YINGMUSIC_REPO_URL, paths.YINGMUSIC_REPO)
    for repo, fname, dest in YINGMUSIC_FILES:
        _hf_download(repo, fname, dest)


def setup_avatar() -> None:
    print("[SadTalker] AVISO: NÃO rode isto no Colab grátis (proíbe face-animation).")
    print("[SadTalker] clonando repo + checkpoints do avatar")
    _clone(SADTALKER_REPO_URL, paths.SADTALKER_REPO)
    for repo, fname, dest in SADTALKER_FILES:
        _hf_download(repo, fname, dest)


COMPONENTS = {
    "seedvc": setup_seedvc,
    "yingmusic": setup_yingmusic,
    "rvc": setup_rvc,
    "vocoder": setup_vocoder,
    "diffsinger": setup_diffsinger,
    "avatar": setup_avatar,
}
# PADRÃO: só Seed-VC (timbre zero-shot) — o motor de voz-guia é o DSP embutido.
# RVC/DiffSinger/avatar são opt-in.
AUDIO_COMPONENTS = ["seedvc"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=list(COMPONENTS), help="baixar apenas um componente")
    ap.add_argument("--with-yingmusic", action="store_true",
                    help="incluir o YingMusic-SVC (SVC SOTA, zero-shot; precisa de Python 3.10).")
    ap.add_argument("--with-rvc", action="store_true",
                    help="incluir RVC (timbre por treino; precisa de Python 3.10 + fairseq).")
    ap.add_argument("--with-diffsinger", action="store_true",
                    help="incluir vocoder + checkpoint DiffSinger EN (SVS de maior qualidade).")
    ap.add_argument("--with-avatar", action="store_true",
                    help="incluir o SadTalker (avatar). NÃO use no Colab grátis — "
                         "ele proíbe face-animation e encerra a sessão. Rode local/host permitido.")
    args = ap.parse_args()

    paths.ensure_dirs()
    if args.only:
        targets = [args.only]
    else:
        targets = list(AUDIO_COMPONENTS)
        if args.with_yingmusic:
            targets.append("yingmusic")
        if args.with_rvc:
            targets.append("rvc")
        if args.with_diffsinger:
            targets += ["vocoder", "diffsinger"]
        if args.with_avatar:
            targets.append("avatar")
    for name in targets:
        COMPONENTS[name]()
    print("\nConcluído. PADRÃO = voz-guia DSP + timbre Seed-VC (zero-shot, sem treino).")
    if "diffsinger" not in targets:
        print("DiffSinger NÃO baixado (motor opcional de maior qualidade). Use --with-diffsinger.")
    if "avatar" not in targets:
        print("Avatar (SadTalker) NÃO baixado. Rode com --with-avatar FORA do Colab grátis.")


if __name__ == "__main__":
    main()
