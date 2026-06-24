"""Patch idempotente no TCSinger: faz o inference/style_transfer.py ler os nomes dos itens
(gen/ref) das variáveis de ambiente TCS_GEN_NAME / TCS_REF_NAME, em vez dos valores
hardcoded. Assim o nosso wrapper escolhe quais itens do metadata.json usar.

Uso: python scripts/patch_tcsinger_env.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import paths  # noqa: E402


def main() -> None:
    f = paths.TCSINGER_REPO / "inference" / "style_transfer.py"
    if not f.exists():
        print(f"[patch-tcsinger] não encontrado: {f}")
        return
    text = f.read_text(encoding="utf-8")
    orig = text

    if "TCS_REF_NAME" not in text:
        # 'ref_name': "..."  ->  'ref_name': os.environ.get('TCS_REF_NAME', "...")
        text = re.sub(r"(['\"]ref_name['\"]\s*:\s*)(\"[^\"]*\"|'[^']*')",
                      r"\1os.environ.get('TCS_REF_NAME', \2)", text, count=1)
        text = re.sub(r"(['\"]gen_name['\"]\s*:\s*)(\"[^\"]*\"|'[^']*')",
                      r"\1os.environ.get('TCS_GEN_NAME', \2)", text, count=1)

    if "import os" not in text:
        text = "import os\n" + text

    if text != orig:
        f.write_text(text, encoding="utf-8")
        print(f"[patch-tcsinger] corrigido: {f}")
    else:
        print("[patch-tcsinger] já estava patcheado (ou padrão não encontrado — confira manualmente)")


if __name__ == "__main__":
    main()
