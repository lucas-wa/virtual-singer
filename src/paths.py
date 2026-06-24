"""Caminhos canônicos do projeto, resolvidos a partir da raiz do repositório."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
SONGS = DATA / "songs"
VOICES = DATA / "voices"
FACES = DATA / "faces"        # rostos (sintéticos/próprios/consentidos) para o avatar

MODELS = ROOT / "models"
DIFFSINGER_DIR = MODELS / "diffsinger"
VOCODER_DIR = MODELS / "nsf_hifigan"
RVC_BASE_DIR = MODELS / "rvc_base"
SADTALKER_CKPT = MODELS / "sadtalker"   # checkpoints do SadTalker
YINGMUSIC_CKPT = MODELS / "yingmusic"   # checkpoints do YingMusic-SVC
TCSINGER_CKPT = MODELS / "tcsinger"     # checkpoints do TCSinger (não usado; vão no repo)

THIRD_PARTY = ROOT / "third_party"
DIFFSINGER_REPO = THIRD_PARTY / "DiffSinger"
RVC_REPO = THIRD_PARTY / "rvc"
SEEDVC_REPO = THIRD_PARTY / "seed-vc"       # conversão de timbre zero-shot (padrão)
YINGMUSIC_REPO = THIRD_PARTY / "YingMusic-SVC"   # SVC zero-shot SOTA (motor opcional, py3.10)
TCSINGER_REPO = THIRD_PARTY / "TCSinger"    # transferência de estilo de canto (py3.10)
SADTALKER_REPO = THIRD_PARTY / "SadTalker"

OUT = ROOT / "out"


def ensure_dirs() -> None:
    for d in (DATA, SONGS, VOICES, FACES, MODELS, THIRD_PARTY, OUT):
        d.mkdir(parents=True, exist_ok=True)
