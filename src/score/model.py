"""Estruturas de dados da partitura e da performance.

Uma `Performance` é a ponte entre o front-end de partitura e o motor SVS:
contém a sequência temporal de notas, cada uma com sua sílaba/letra e os fonemas
correspondentes, com durações em segundos.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Phoneme:
    """Um fonema com sua duração (segundos) dentro da nota a que pertence."""
    symbol: str          # fonema em ARPAbet (ex.: "AH0", "T") ou "SP"/"AP" (pausa/respiração)
    duration: float      # segundos


@dataclass
class Note:
    """Uma nota musical com a sílaba cantada e seus fonemas."""
    start: float                 # início em segundos
    duration: float              # duração em segundos
    midi_pitch: int              # 0-127; 0 = silêncio/descanso
    lyric: str                   # sílaba/palavra associada (vazio se descanso)
    phonemes: List[Phoneme] = field(default_factory=list)

    @property
    def end(self) -> float:
        return self.start + self.duration

    @property
    def is_rest(self) -> bool:
        return self.midi_pitch <= 0 or self.lyric.strip() == ""


@dataclass
class Performance:
    """Sequência completa de notas que o motor SVS vai sintetizar."""
    notes: List[Note]
    tempo_bpm: float = 120.0
    language: str = "en"
    title: str = ""

    @property
    def duration(self) -> float:
        return max((n.end for n in self.notes), default=0.0)

    def all_phonemes(self) -> List[Phoneme]:
        out: List[Phoneme] = []
        for n in self.notes:
            out.extend(n.phonemes)
        return out

    def summary(self) -> str:
        n_sung = sum(1 for n in self.notes if not n.is_rest)
        return (
            f"'{self.title or 'sem título'}' | {self.language} | "
            f"{len(self.notes)} notas ({n_sung} cantadas) | "
            f"{self.duration:.1f}s | {len(self.all_phonemes())} fonemas"
        )
