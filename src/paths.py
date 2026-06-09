"""Caminhos canônicos do projeto, resolvidos a partir da raiz do repositório."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
SONGS = DATA / "songs"
VOICES = DATA / "voices"

MODELS = ROOT / "models"
DIFFSINGER_DIR = MODELS / "diffsinger"
VOCODER_DIR = MODELS / "nsf_hifigan"
RVC_BASE_DIR = MODELS / "rvc_base"

THIRD_PARTY = ROOT / "third_party"
DIFFSINGER_REPO = THIRD_PARTY / "DiffSinger"
RVC_REPO = THIRD_PARTY / "rvc"

OUT = ROOT / "out"


def ensure_dirs() -> None:
    for d in (DATA, SONGS, VOICES, MODELS, THIRD_PARTY, OUT):
        d.mkdir(parents=True, exist_ok=True)
