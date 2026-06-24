"""Geração de acompanhamento por gênero via MusicGen-melody (AudioCraft/Meta).

Recebe um áudio de **melodia** (o vocal sintetizado ou a melodia renderizada) + um
**prompt de texto de gênero** e gera um acompanhamento instrumental seguindo a melodia
no estilo descrito. É o eixo de "gênero/arranjo" do pipeline de estilo.

É lib pip (`audiocraft`), não subprocess. Os pesos (`facebook/musicgen-melody`) baixam
na 1ª execução. Use prompt de texto (gênero) — não precisa de áudio de terceiros.

Ref.: https://github.com/facebookresearch/audiocraft
"""
from __future__ import annotations

from pathlib import Path

from ..hardware import RuntimeProfile, detect_profile

# MusicGen gera no máximo ~30s por vez; músicas maiores precisam de geração em blocos.
_MAX_DURATION = 30.0


def generate_accompaniment(melody_wav: str | Path, out_wav: str | Path, prompt: str,
                           profile: RuntimeProfile | None = None,
                           duration: float | None = None,
                           model_name: str = "facebook/musicgen-melody") -> Path:
    """Gera acompanhamento no gênero `prompt`, seguindo a melodia de `melody_wav`.

    `prompt` = descrição do gênero/arranjo (ex.: "bossa nova, soft acoustic guitar and
    light percussion"). Devolve o caminho do .wav gerado.
    """
    profile = profile or detect_profile()
    import torch
    import torchaudio
    from audiocraft.data.audio import audio_write
    from audiocraft.models import MusicGen

    melody_wav = Path(melody_wav)
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    melody, sr = torchaudio.load(str(melody_wav))
    if duration is None:
        duration = melody.shape[-1] / sr
    duration = min(float(duration), _MAX_DURATION)

    print(f"[musicgen] {profile.summary()} | dur={duration:.1f}s | prompt='{prompt}'")
    model = MusicGen.get_pretrained(model_name)
    model.set_generation_params(duration=duration)

    device = "cuda" if profile.device == "cuda" else "cpu"
    wav = model.generate_with_chroma([prompt], melody[None].to(device), sr)

    # audio_write acrescenta .wav e normaliza loudness.
    stem = out_wav.with_suffix("")
    audio_write(str(stem), wav[0].cpu(), model.sample_rate, strategy="loudness")
    produced = stem.with_suffix(".wav")
    print(f"[musicgen] acompanhamento -> {produced}")
    return produced
