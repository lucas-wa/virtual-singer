"""Corrige incompatibilidades do SadTalker/basicsr com torch/torchvision/numpy novos.

Dois patches, ambos idempotentes:
  1. basicsr: `torchvision.transforms.functional_tensor` -> `...functional`
     (functional_tensor foi removido no torchvision>=0.17).
  2. SadTalker (e basicsr): aliases removidos do numpy>=1.24 — `np.float`, `np.int`,
     `np.bool`, `np.object`, `np.complex`, `np.str` -> builtins. Usa \b para NÃO tocar
     em `np.float64`, `np.int32`, `np.bool_`, etc.

Uso: python scripts/patch_sadtalker_env.py
"""
from __future__ import annotations

import importlib.util
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src import paths  # noqa: E402

TV_OLD = "torchvision.transforms.functional_tensor"
TV_NEW = "torchvision.transforms.functional"

# np.<alias> -> builtin, só quando NÃO seguido de letra/dígito/_ (\b protege np.float64 etc.)
NP_ALIASES = {"float": "float", "int": "int", "bool": "bool",
              "object": "object", "complex": "complex", "str": "str"}
NP_PATTERNS = [(re.compile(rf"\bnp\.{a}\b"), b) for a, b in NP_ALIASES.items()]

# Correções literais (bugs do SadTalker com numpy novo): montar array misturando escalar
# com array de 1 elemento agora dá "inhomogeneous shape" -> extrair o escalar com [0].
LITERAL_FIXES = {
    "np.array([w0, h0, s, t[0], t[1]])": "np.array([w0, h0, s, t[0][0], t[1][0]])",
}


def _patch_file(f: pathlib.Path) -> bool:
    try:
        text = f.read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        return False
    orig = text
    if TV_OLD in text:
        text = text.replace(TV_OLD, TV_NEW)
    for old, new in LITERAL_FIXES.items():
        if old in text:
            text = text.replace(old, new)
    for pat, repl in NP_PATTERNS:
        text = pat.sub(repl, text)
    if text != orig:
        f.write_text(text, encoding="utf-8")
        return True
    return False


def _basicsr_root() -> pathlib.Path | None:
    # find_spec NÃO executa o __init__ do basicsr (que falharia com o import quebrado).
    spec = importlib.util.find_spec("basicsr")
    return pathlib.Path(spec.origin).parent if spec and spec.origin else None


def main() -> None:
    roots = []
    br = _basicsr_root()
    if br:
        roots.append(br)
    if paths.SADTALKER_REPO.exists():
        roots.append(paths.SADTALKER_REPO)

    if not roots:
        print("[patch] basicsr e SadTalker não encontrados; nada a fazer")
        return

    n = 0
    for root in roots:
        for f in root.rglob("*.py"):
            if _patch_file(f):
                print(f"[patch] corrigido: {f}")
                n += 1
    print(f"[patch] {n} arquivo(s) ajustado(s) (functional_tensor + aliases np.*)")


if __name__ == "__main__":
    main()
