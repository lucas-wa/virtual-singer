"""Validação de uma `Performance` importada.

Aponta problemas comuns que quebram a síntese: melodia polifônica, notas sem letra,
extensão fora do cantável, ausência de fonemas, etc. Devolve listas de erros e avisos.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .model import Performance

# Faixa cantável aproximada (MIDI): C2 (36) a C6 (84).
_MIN_PITCH, _MAX_PITCH = 36, 84


@dataclass
class ValidationReport:
    errors: List[str] = field(default_factory=list)     # impedem a síntese
    warnings: List[str] = field(default_factory=list)   # síntese roda, mas qualidade cai

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        lines = []
        for e in self.errors:
            lines.append(f"  [ERRO]  {e}")
        for w in self.warnings:
            lines.append(f"  [aviso] {w}")
        if not lines:
            lines.append("  [OK] partitura valida, sem problemas")
        return "\n".join(lines)


def _has_overlaps(perf: Performance) -> bool:
    """Detecta polifonia: notas cantadas que se sobrepõem no tempo."""
    sung = sorted((n for n in perf.notes if not n.is_rest), key=lambda n: n.start)
    for a, b in zip(sung, sung[1:]):
        if b.start < a.end - 1e-4:
            return True
    return False


def validate(perf: Performance) -> ValidationReport:
    rep = ValidationReport()

    sung = [n for n in perf.notes if not n.is_rest]
    if not sung:
        rep.errors.append("nenhuma nota cantada (todas são pausas ou sem letra)")
    if perf.duration <= 0:
        rep.errors.append("duração total é zero")

    if _has_overlaps(perf):
        rep.errors.append(
            "melodia polifônica (notas sobrepostas) — o SVS exige melodia monofônica; "
            "exporte só a linha de melodia/voz"
        )

    # notas com pitch mas sem letra (ex.: MIDI sem lyrics.txt)
    no_lyric = sum(1 for n in perf.notes if n.midi_pitch > 0 and not n.lyric.strip())
    if no_lyric:
        rep.warnings.append(
            f"{no_lyric} nota(s) com altura mas sem sílaba — serão cantadas como vogal "
            "neutra; forneça a letra (lyrics.txt ou letra no MusicXML)"
        )

    # extensão fora do cantável
    out_of_range = [n.midi_pitch for n in sung
                    if not (_MIN_PITCH <= n.midi_pitch <= _MAX_PITCH)]
    if out_of_range:
        rep.warnings.append(
            f"{len(out_of_range)} nota(s) fora da faixa cantável "
            f"(MIDI {_MIN_PITCH}-{_MAX_PITCH}); considere transpor"
        )

    # fonemas ausentes
    if not perf.all_phonemes():
        rep.errors.append("nenhum fonema gerado — verifique o G2P/idioma e a letra")

    return rep
