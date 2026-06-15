"""Sintetizador de voz-guia por DSP (fonte-filtro), em numpy puro — sem modelos externos.

Gera uma melodia cantada a partir da `Performance`:
  - fonte glotal = soma de harmônicos em f0 (com leve vibrato);
  - filtro = realce de formantes da vogal de cada nota (modela o trato vocal);
  - envelope ADSR por nota para evitar cliques.

Não é "realista", mas produz pitch e ritmo corretos e roda em qualquer lugar (até CPU).
Serve de voz-guia: o RVC depois impõe o timbre do usuário sobre este sinal.

Conteúdo clássico de processamento de voz (síntese fonte-filtro / formantes).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..score.model import Performance

# Formantes médios (F1, F2, F3) em Hz por vogal ARPAbet (aprox. inglês americano).
_VOWEL_FORMANTS = {
    "AA": (700, 1220, 2600), "AE": (660, 1700, 2400), "AH": (600, 1190, 2390),
    "AO": (590, 880, 2540), "AW": (640, 1230, 2550), "AY": (660, 1700, 2400),
    "EH": (530, 1840, 2480), "ER": (490, 1350, 1690), "EY": (530, 1840, 2480),
    "IH": (390, 1990, 2550), "IY": (270, 2290, 3010), "OW": (430, 1100, 2540),
    "OY": (550, 960, 2540), "UH": (440, 1020, 2240), "UW": (300, 870, 2240),
}
_DEFAULT_FORMANTS = _VOWEL_FORMANTS["AH"]  # schwa para consoantes/sem vogal
_FORMANT_BW = 110.0  # largura (Hz) de cada ressonância


def _note_vowel(phoneme_symbols: list[str]) -> tuple[float, float, float]:
    """Acha a vogal da nota (fonema com dígito de stress) e devolve seus formantes."""
    for sym in phoneme_symbols:
        base = sym.rstrip("012")
        if base in _VOWEL_FORMANTS and sym[-1:].isdigit():
            return _VOWEL_FORMANTS[base]
    # tenta qualquer vogal sem stress
    for sym in phoneme_symbols:
        base = sym.rstrip("012")
        if base in _VOWEL_FORMANTS:
            return _VOWEL_FORMANTS[base]
    return _DEFAULT_FORMANTS


def _formant_gain(freqs: np.ndarray, formants: tuple[float, float, float]) -> np.ndarray:
    """Ganho espectral (envelope de formantes) avaliado em `freqs` (Hz)."""
    gain = np.full_like(freqs, 0.03)  # piso
    for i, f in enumerate(formants):
        weight = 1.0 / (1 + 0.5 * i)  # F1 mais forte que F2/F3
        gain += weight * np.exp(-0.5 * ((freqs - f) / _FORMANT_BW) ** 2)
    return gain


def _adsr(n: int, sr: int) -> np.ndarray:
    """Envelope simples: ataque/curva/release curtos para não estourar nas bordas."""
    env = np.ones(n)
    a = min(int(0.02 * sr), n // 4)   # ataque 20 ms
    r = min(int(0.04 * sr), n // 4)   # release 40 ms
    if a > 0:
        env[:a] = np.linspace(0, 1, a)
    if r > 0:
        env[-r:] = np.linspace(1, 0, r)
    return env


def _midi_to_hz(midi: int) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def _synth_note(f0: float, dur: float, sr: int,
                formants: tuple[float, float, float]) -> np.ndarray:
    n = max(int(dur * sr), 1)
    t = np.arange(n) / sr

    # vibrato suave (~5.5 Hz, ±~1.5%) entrando após 0.15 s
    onset = np.clip(t / 0.15, 0, 1)
    vibrato = 1.0 + 0.015 * onset * np.sin(2 * np.pi * 5.5 * t)
    f0_contour = f0 * vibrato

    # fase instantânea (integra o f0 variável)
    phase = 2 * np.pi * np.cumsum(f0_contour) / sr

    sig = np.zeros(n)
    k = 1
    nyq = sr / 2 * 0.95
    while k * f0 < nyq:
        fk = k * f0
        src_amp = 1.0 / k                              # espectro fonte (-6 dB/oitava)
        form_amp = _formant_gain(np.array([fk]), formants)[0]
        sig += src_amp * form_amp * np.sin(k * phase)
        k += 1

    sig *= _adsr(n, sr)
    return sig


def synthesize_dsp(perf: Performance, out_wav: str | Path, sr: int = 44100) -> Path:
    """Sintetiza a voz-guia da performance e salva em `out_wav`."""
    from .. import audio

    out_wav = Path(out_wav)
    chunks = []
    for note in perf.notes:
        if note.is_rest:
            chunks.append(np.zeros(max(int(note.duration * sr), 1)))
            continue
        formants = _note_vowel([p.symbol for p in note.phonemes])
        chunks.append(_synth_note(_midi_to_hz(note.midi_pitch), note.duration, sr, formants))

    out = np.concatenate(chunks) if chunks else np.zeros(sr)
    peak = float(np.max(np.abs(out))) or 1.0
    out = 0.9 * out / peak
    audio.save_wav(out_wav, out.astype(np.float32), sr)
    print(f"[svs-dsp] voz-guia ({perf.duration:.1f}s, {len(perf.notes)} notas) -> {out_wav}")
    return out_wav


if __name__ == "__main__":
    import sys

    from ..score import build_performance

    song = sys.argv[1] if len(sys.argv) > 1 else "data/songs/demo"
    out = sys.argv[2] if len(sys.argv) > 2 else "out/guide_dsp.wav"
    synthesize_dsp(build_performance(song), out)
