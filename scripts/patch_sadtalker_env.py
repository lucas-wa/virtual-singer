"""Corrige a incompatibilidade do basicsr (usado por GFPGAN/SadTalker) com torchvision novo.

O basicsr importa `torchvision.transforms.functional_tensor`, removido no torchvision>=0.17.
A função `rgb_to_grayscale` agora vive em `torchvision.transforms.functional`. Reescrevemos
o import nos arquivos do basicsr instalado. Idempotente (não faz nada se já corrigido).

Uso: python scripts/patch_sadtalker_env.py
"""
from __future__ import annotations

import importlib.util
import pathlib

OLD = "torchvision.transforms.functional_tensor"
NEW = "torchvision.transforms.functional"


def main() -> None:
    # find_spec NÃO executa o __init__ do basicsr (que falharia com o import quebrado).
    spec = importlib.util.find_spec("basicsr")
    if spec is None or not spec.origin:
        print("[patch] basicsr não encontrado; nada a fazer")
        return
    root = pathlib.Path(spec.origin).parent

    n = 0
    for f in root.rglob("*.py"):
        try:
            text = f.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            continue
        if OLD in text:
            f.write_text(text.replace(OLD, NEW), encoding="utf-8")
            print(f"[patch] corrigido: {f}")
            n += 1
    print(f"[patch] {n} arquivo(s) do basicsr ajustado(s) (functional_tensor -> functional)")


if __name__ == "__main__":
    main()
