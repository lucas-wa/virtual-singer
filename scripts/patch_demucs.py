"""Corrige o bug do Demucs com PyTorch novo: operações in-place num tensor de memória
compartilhada (mono expandido p/ estéreo) disparam
    RuntimeError: ... more than one element of the written-to tensor refers to a single
    memory location.

Troca as operações in-place sobre `wav` por versões out-of-place no demucs/separate.py.
Idempotente (não faz nada se já corrigido).

Uso: python scripts/patch_demucs.py
"""
from __future__ import annotations

import importlib.util
import pathlib

# (trecho antigo, trecho novo) — só sobre `wav` (o tensor que compartilha memória).
REPLACEMENTS = [
    ("wav -= ref.mean()", "wav = wav - ref.mean()"),
    ("wav /= ref.std()", "wav = wav / ref.std()"),
    ("wav -= ref.mean(0)", "wav = wav - ref.mean(0)"),
]


def main() -> None:
    spec = importlib.util.find_spec("demucs")
    if spec is None or not spec.submodule_search_locations:
        print("[patch-demucs] demucs não encontrado; nada a fazer")
        return
    root = pathlib.Path(list(spec.submodule_search_locations)[0])

    n = 0
    for f in root.rglob("*.py"):
        try:
            text = f.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            continue
        orig = text
        for old, new in REPLACEMENTS:
            if old in text:
                text = text.replace(old, new)
        if text != orig:
            f.write_text(text, encoding="utf-8")
            print(f"[patch-demucs] corrigido: {f}")
            n += 1
    print(f"[patch-demucs] {n} arquivo(s) ajustado(s) (in-place -> out-of-place)")


if __name__ == "__main__":
    main()
