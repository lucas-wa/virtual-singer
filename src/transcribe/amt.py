"""Núcleo da transcrição: f0 (pitch) -> notas MIDI.

Pipeline clássico de processamento de áudio:
  1. estimar f0 quadro a quadro (pYIN, via librosa — robusto e sem GPU);
  2. converter Hz -> semitom MIDI, suavizar com filtro de mediana (remove jitter/oitavas);
  3. segmentar em notas: quadros consecutivos de mesma altura viram uma nota; trechos
     não-vozeados ou curtos demais viram pausa.

Para pitch ainda mais preciso, CREPE (rede neural) pode substituir o pYIN — mas exige
torch/GPU. O pYIN cobre bem o caso de uma melodia cantada/assobiada.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..score.model import Note

# Faixa de busca de f0 padrão: C2 (~65 Hz) a C6 (~1047 Hz).
DEFAULT_FMIN = 65.0
DEFAULT_FMAX = 1047.0


def extract_f0(audio: np.ndarray, sr: int, fmin: float = DEFAULT_FMIN,
               fmax: float = DEFAULT_FMAX, hop_length: int = 512
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estima f0 com pYIN. Retorna (tempos, f0_hz, vozeado)."""
    import librosa

    f0, voiced_flag, _ = librosa.pyin(
        audio, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    return times, f0, voiced_flag


def extract_onsets(audio: np.ndarray, sr: int, hop_length: int = 512) -> set:
    """Detecta onsets (ataques de nota) e devolve os índices de quadro.

    Necessário para separar notas REPETIDAS de mesma altura — a estimativa de f0
    sozinha não as distingue, pois o pitch não muda entre elas.
    """
    import librosa

    frames = librosa.onset.onset_detect(
        y=audio, sr=sr, hop_length=hop_length, units="frames", backtrack=True
    )
    return set(int(f) for f in frames)


def _hz_to_midi_int(f0: np.ndarray, voiced: np.ndarray) -> np.ndarray:
    """Hz -> semitom MIDI inteiro; 0 onde não-vozeado/sem pitch."""
    midi = np.zeros(len(f0), dtype=int)
    for i, (hz, v) in enumerate(zip(f0, voiced)):
        if v and hz and not np.isnan(hz) and hz > 0:
            midi[i] = int(round(69 + 12 * np.log2(hz / 440.0)))
    return midi


def _median_smooth(seq: np.ndarray, kernel: int = 5) -> np.ndarray:
    """Filtro de mediana para remover jitter e saltos de oitava espúrios."""
    try:
        from scipy.signal import medfilt

        if kernel % 2 == 0:
            kernel += 1
        return medfilt(seq, kernel_size=kernel).astype(int)
    except Exception:  # noqa: BLE001
        return seq


def f0_to_notes(times: np.ndarray, f0: np.ndarray, voiced: np.ndarray,
                hop_s: float, min_note_s: float = 0.08,
                smooth_kernel: int = 5, onset_frames: set | None = None) -> List[Note]:
    """Segmenta a trilha de f0 em notas musicais.

    Uma nota termina quando (a) a altura muda, (b) o sinal fica não-vozeado, ou
    (c) há um onset (ataque) detectado — este último separa notas repetidas.
    """
    midi = _median_smooth(_hz_to_midi_int(f0, voiced), smooth_kernel)
    onset_frames = onset_frames or set()
    notes: List[Note] = []
    n = len(midi)
    i = 0
    while i < n:
        pitch = int(midi[i])
        j = i + 1
        # estende enquanto a altura for a mesma E não houver um novo onset
        while j < n and int(midi[j]) == pitch and j not in onset_frames:
            j += 1
        start = float(times[i])
        dur = (j - i) * hop_s
        if pitch > 0 and dur >= min_note_s:
            notes.append(Note(start=start, duration=dur, midi_pitch=pitch, lyric=""))
        # trechos de silêncio ou curtos viram pausa implícita (não emitimos Note de rest)
        i = j
    return notes


def notes_to_midi(notes: List[Note], path, tempo_bpm: float = 120.0) -> None:
    """Escreve as notas como um arquivo MIDI monofônico."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo_bpm)
    inst = pretty_midi.Instrument(program=0)
    for nt in notes:
        if nt.midi_pitch <= 0:
            continue
        inst.notes.append(pretty_midi.Note(
            velocity=90, pitch=int(nt.midi_pitch),
            start=float(nt.start), end=float(nt.end),
        ))
    pm.instruments.append(inst)
    pm.write(str(path))


def transcribe_audio(audio_path, fmin: float = DEFAULT_FMIN, fmax: float = DEFAULT_FMAX,
                     hop_length: int = 512, min_note_s: float = 0.08) -> List[Note]:
    """Conveniência: carrega o áudio e devolve as notas transcritas."""
    from ..audio import load_wav

    audio, sr = load_wav(audio_path)
    times, f0, voiced = extract_f0(audio, sr, fmin, fmax, hop_length)
    onsets = extract_onsets(audio, sr, hop_length)
    return f0_to_notes(times, f0, voiced, hop_s=hop_length / sr,
                       min_note_s=min_note_s, onset_frames=onsets)
