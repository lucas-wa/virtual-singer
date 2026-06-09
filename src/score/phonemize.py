"""G2P (grafema->fonema) em inglês e distribuição de fonemas dentro de cada nota.

Em canto, a vogal sustenta a nota e as consoantes são curtas. Por isso damos às
consoantes uma duração fixa pequena e a vogal fica com o tempo restante da nota.
"""
from __future__ import annotations

from typing import List

from .model import Note, Phoneme

# Duração fixa (segundos) atribuída a cada consoante; a vogal recebe o resto.
_CONSONANT_DUR = 0.06
# Símbolos de "vogal" no ARPAbet terminam com dígito de stress (0/1/2).
_VOWEL_SUFFIXES = ("0", "1", "2")


class EnglishG2P:
    """Wrapper preguiçoso sobre g2p_en (carrega o modelo só uma vez)."""

    def __init__(self) -> None:
        self._g2p = None

    def _ensure(self):
        if self._g2p is None:
            from g2p_en import G2p

            self._g2p = G2p()
        return self._g2p

    def word_to_phonemes(self, word: str) -> List[str]:
        g2p = self._ensure()
        # g2p_en devolve fonemas ARPAbet e pontuação; filtramos não-fonemas.
        return [p for p in g2p(word) if p not in (" ", "", "'") and not p.isspace()]


def _is_vowel(symbol: str) -> bool:
    return symbol.endswith(_VOWEL_SUFFIXES)


def _distribute(symbols: List[str], total: float) -> List[Phoneme]:
    """Distribui os fonemas de uma sílaba ao longo de `total` segundos."""
    if not symbols:
        return [Phoneme("SP", total)]  # silêncio

    vowel_idx = next((i for i, s in enumerate(symbols) if _is_vowel(s)), None)
    n = len(symbols)

    if vowel_idx is None:
        # sem vogal (raro): divide igualmente
        each = total / n
        return [Phoneme(s, each) for s in symbols]

    # consoantes curtas, vogal sustenta o resto
    cons_count = n - 1
    cons_total = min(_CONSONANT_DUR * cons_count, total * 0.5)
    cons_each = cons_total / cons_count if cons_count else 0.0
    vowel_dur = max(total - cons_total, 0.02)

    out: List[Phoneme] = []
    for i, s in enumerate(symbols):
        out.append(Phoneme(s, vowel_dur if i == vowel_idx else cons_each))
    return out


def phonemize_notes(notes: List[Note], g2p: EnglishG2P | None = None) -> List[Note]:
    """Preenche `note.phonemes` para cada nota cantada (mutação in-place + retorno)."""
    g2p = g2p or EnglishG2P()
    for note in notes:
        if note.is_rest:
            note.phonemes = [Phoneme("SP", note.duration)]
            continue
        # remove o hífen de junção de sílabas antes do G2P
        word = note.lyric.replace("-", "").strip()
        symbols = g2p.word_to_phonemes(word) if word else []
        note.phonemes = _distribute(symbols, note.duration)
    return notes
