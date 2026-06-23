"""Conversão de timbre ZERO-SHOT via Seed-VC (sem treino).

Em vez de treinar um modelo por voz (RVC), o Seed-VC clona o timbre a partir de um
clipe de referência de 1-30 s. Vantagens: nada de fairseq, roda em Python 3.12, e
trocar de voz é só apontar para outro clipe de referência.

Encapsula o `inference.py` do repositório Plachtaa/seed-vc (vendored em third_party/seed-vc).
Os checkpoints são baixados automaticamente na primeira execução.

Ref.: https://github.com/Plachtaa/seed-vc
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .. import paths
from ..hardware import RuntimeProfile, detect_profile

_AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")


def resolve_reference(voice_name: str, max_seconds: float = 30.0) -> Path:
    """Acha um clipe de referência para a voz `voice_name`.

    Aceita tanto uma pasta data/voices/<nome>/ (pega o 1º áudio) quanto um caminho
    direto para um arquivo. Recorta para <= max_seconds (Seed-VC pede 1-30 s).
    """
    cand = Path(voice_name)
    if cand.is_file():
        clip = cand
    else:
        folder = cand if cand.is_dir() else (paths.VOICES / voice_name)
        if not folder.is_dir():
            raise FileNotFoundError(f"voz/referência não encontrada: {voice_name}")
        clips = sorted(p for p in folder.iterdir() if p.suffix.lower() in _AUDIO_EXTS)
        if not clips:
            raise FileNotFoundError(f"nenhum áudio de referência em {folder}")
        clip = clips[0]

    return _trim(clip, max_seconds)


def _trim(clip: Path, max_seconds: float) -> Path:
    """Garante referência <= max_seconds; se maior, salva um recorte em out/."""
    try:
        import soundfile as sf

        info = sf.info(str(clip))
        if info.frames / info.samplerate <= max_seconds:
            return clip
        audio, sr = sf.read(str(clip), frames=int(max_seconds * info.samplerate))
        out = paths.OUT / f"_ref_{clip.stem}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), audio, sr)
        return out
    except Exception:  # noqa: BLE001
        return clip  # se algo falhar, usa o clipe original


def _newest_wav(folder: Path) -> Path | None:
    wavs = list(folder.rglob("*.wav"))
    return max(wavs, key=lambda p: p.stat().st_mtime) if wavs else None


def convert(source_wav: str | Path, reference: str | Path, out_wav: str | Path,
            profile: RuntimeProfile | None = None, diffusion_steps: int = 30,
            semitone_shift: int = 0, sing: bool = True) -> Path:
    """Converte `source_wav` (voz-guia) para o timbre de `reference` (clipe curto).

    sing=True ativa o condicionamento de F0 (modo canto). semitone_shift transpõe.
    """
    profile = profile or detect_profile()
    # ABSOLUTOS: o Seed-VC roda com cwd=repo, então caminhos relativos quebrariam.
    source_wav = Path(source_wav).resolve()
    reference = Path(reference).resolve()
    out_wav = Path(out_wav).resolve()
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    infer = paths.SEEDVC_REPO / "inference.py"
    if not infer.exists():
        raise RuntimeError(
            f"Seed-VC não encontrado em {infer}. "
            "Rode: python scripts/setup_models.py --only seedvc"
        )

    work = (out_wav.parent / "_seedvc_work").resolve()
    work.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(infer),
        "--source", str(source_wav),
        "--target", str(reference),
        "--output", str(work),
        "--diffusion-steps", str(diffusion_steps),
        "--f0-condition", "True" if sing else "False",
        "--semi-tone-shift", str(semitone_shift),
        "--fp16", "True" if profile.use_fp16 else "False",
    ]
    print(f"[seed-vc] {profile.summary()}")
    print(f"[seed-vc] $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(paths.SEEDVC_REPO))

    produced = _newest_wav(work)
    if produced is None:
        raise RuntimeError(f"Seed-VC não gerou áudio em {work}")
    produced.replace(out_wav)
    print(f"[seed-vc] timbre convertido -> {out_wav}")
    return out_wav
