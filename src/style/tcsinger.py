"""Transferência de estilo de canto via TCSinger (EMNLP 2024, AaronZ345/TCSinger).

Recebe a partitura ALVO (nova melodia + letra, como `Performance`) e uma REFERÊNCIA de
estilo (gravação de vocês + sua partitura), e gera o vocal cantando a nova melodia com o
estilo da referência (técnica/emoção/ritmo/ornamentação). O timbre é ajustado depois pelo
Seed-VC/YingMusic (cascade).

Como o TCSinger consome os dados (confirmado lendo o repo):
  - `inference/style_transfer.py` lê `{processed_data_dir}/metadata.json` e seleciona dois
    itens por nome: `gen_name` (alvo) e `ref_name` (estilo).
  - Cada item: ph (fonemas ARPAbet — COMPATÍVEL com nosso g2p_en), ep_pitches (nota por
    fonema), ep_notedurs (duração por fonema), ep_types (1=rest, 2=lyric, 3=slur), wav_fn.
  - Fonemas EN do TCSinger = ARPAbet padrão (AA1, T, IH1...); rest = "<SP>".

================================ CALIBRAÇÃO NO CLUSTER ================================
3 costuras a validar contra o repo real (não deu para fixar 100% sem um metadata.json de
exemplo do GTSinger):
  (1) Formato de ep_pitches/ep_notedurs/ep_types — aqui: nota por nome ("C4"/"rest"),
      duração em segundos, tipos 1/2/3, listas do MESMO tamanho de `ph`. Conferir com um
      item real do GTSinger/metadata.json.
  (2) ref_name/gen_name são HARDCODED no style_transfer.py -> use scripts/patch_tcsinger_env.py
      para fazê-los lerem TCS_GEN_NAME/TCS_REF_NAME do ambiente.
  (3) processed_data_dir é passado via `--hparams "processed_data_dir=<dir>"` (convenção
      NATSpeech). Se a versão não aceitar, apontar o config para o nosso dir.
======================================================================================
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from .. import paths
from ..hardware import RuntimeProfile, detect_profile
from ..score.model import Performance
from ..svs.ds_format import midi_to_note_name

REF_NAME = "vsinger_ref_style"
GEN_NAME = "vsinger_gen_target"


def performance_to_item(perf: Performance, item_name: str, wav_fn: str = "") -> dict:
    """Converte uma `Performance` num item do TCSinger (listas por fonema)."""
    ph, ep_pitches, ep_notedurs, ep_types, ph_durs = [], [], [], [], []
    for note in perf.notes:
        pitch_name = "rest" if note.is_rest else midi_to_note_name(note.midi_pitch)
        for p in note.phonemes:
            sym = "<SP>" if p.symbol in ("SP", "AP", "") else p.symbol
            ph.append(sym)
            ep_pitches.append(pitch_name)               # nota por fonema
            ep_notedurs.append(float(round(note.duration, 4)))  # duração da NOTA por fonema (s)
            ep_types.append(1 if note.is_rest else 2)    # 1=rest, 2=lyric (3=slur: futuro)
            ph_durs.append(float(round(p.duration, 4)))  # duração do fonema (s) — float nativo p/ JSON
    return {
        "item_name": item_name, "ph": ph, "ep_pitches": ep_pitches,
        "ep_notedurs": ep_notedurs, "ep_types": ep_types, "ph_durs": ph_durs,
        "wav_fn": str(wav_fn),
    }


def _newest(folder: Path, pattern: str, since: float) -> Path | None:
    hits = [p for p in folder.rglob(pattern) if p.stat().st_mtime >= since]
    return max(hits, key=lambda p: p.stat().st_mtime) if hits else None


def style_transfer(target: Performance, style_ref: Performance, style_wav: str | Path,
                   out_wav: str | Path, profile: RuntimeProfile | None = None) -> Path:
    """Sintetiza `target` (nova melodia+letra) com o estilo de `style_ref`+`style_wav`.

    style_wav = gravação de vocês (referência de estilo), idealmente 48 kHz.
    """
    profile = profile or detect_profile()
    out_wav = Path(out_wav).resolve()
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    style_wav = Path(style_wav).resolve()

    infer = paths.TCSINGER_REPO / "inference" / "style_transfer.py"
    if not infer.exists():
        raise RuntimeError(
            f"TCSinger não encontrado em {infer}. "
            "Rode: python scripts/setup_models.py --with-tcsinger"
        )

    # 1. metadata.json com os dois itens (gen=alvo, ref=estilo).
    work = (out_wav.parent / "_tcsinger_work").resolve()
    work.mkdir(parents=True, exist_ok=True)
    items = [
        performance_to_item(target, GEN_NAME, wav_fn=""),
        performance_to_item(style_ref, REF_NAME, wav_fn=str(style_wav)),
    ]
    (work / "metadata.json").write_text(json.dumps(items, ensure_ascii=False, indent=2),
                                        encoding="utf-8")

    # 2. roda o style_transfer.py do TCSinger (cwd=repo). ref/gen via env (ver patch).
    import os
    env = dict(os.environ, TCS_GEN_NAME=GEN_NAME, TCS_REF_NAME=REF_NAME)
    since = time.time()
    cmd = [
        sys.executable, "inference/style_transfer.py",
        "--config", "egs/sdlm.yaml",
        "--exp_name", "checkpoints/SDLM",
        "--hparams", f"processed_data_dir={work}",
    ]
    print(f"[tcsinger] {profile.summary()}")
    print(f"[tcsinger] $ TCS_GEN_NAME={GEN_NAME} TCS_REF_NAME={REF_NAME} {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, cwd=str(paths.TCSINGER_REPO), env=env)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"TCSinger falhou (exit {e.returncode}). Verifique: (1) formato dos campos "
            "ep_* no metadata.json vs o esperado pelo repo; (2) patch_tcsinger_env aplicado "
            "(ref/gen via env); (3) --hparams processed_data_dir aceito. Veja o log acima."
        ) from e

    # 3. saída padrão do TCSinger: infer_out/transfer.wav
    produced = (paths.TCSINGER_REPO / "infer_out" / "transfer.wav")
    if not produced.exists():
        produced = _newest(paths.TCSINGER_REPO / "infer_out", "*.wav", since) \
            or _newest(paths.TCSINGER_REPO, "*.wav", since)
    if not produced or not produced.exists():
        raise RuntimeError(f"TCSinger não gerou .wav (esperado em {paths.TCSINGER_REPO}/infer_out)")
    produced.replace(out_wav)
    print(f"[tcsinger] vocal estilizado -> {out_wav}")
    return out_wav
