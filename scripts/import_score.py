"""Importa e valida uma partitura para uso no pipeline.

Aceita MusicXML (.musicxml/.xml/.mxl) ou MIDI (.mid/.midi). Copia o arquivo para
data/songs/<nome>/, gera a Performance, roda a validação e mostra um relatório.

Para MIDI, forneça também a letra com --lyrics (uma sílaba por nota; veja README).

Uso:
    python scripts/import_score.py minha_musica.musicxml
    python scripts/import_score.py melodia.mid --name drivers_license --lyrics letra.txt

Origem do arquivo: use material que você tenha direito de usar — uma partitura que
VOCÊ exportou (ex.: do MuseScore), domínio público, ou licença aberta. Veja CONSENT.md.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import paths  # noqa: E402
from src.score import build_performance, validate  # noqa: E402

_SCORE_EXT = {".musicxml", ".xml", ".mxl", ".mid", ".midi"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", help="arquivo .musicxml/.mxl/.xml ou .mid/.midi")
    ap.add_argument("--name", default=None, help="nome da música (default: nome do arquivo)")
    ap.add_argument("--lyrics", default=None, help="arquivo de letra (para MIDI sem letra)")
    args = ap.parse_args()

    src = Path(args.file)
    if not src.exists():
        raise SystemExit(f"arquivo não encontrado: {src}")
    if src.suffix.lower() not in _SCORE_EXT:
        raise SystemExit(f"extensão não suportada: {src.suffix} (use {sorted(_SCORE_EXT)})")

    name = args.name or src.stem
    dest_dir = paths.SONGS / name
    dest_dir.mkdir(parents=True, exist_ok=True)

    # MusicXML vira 'score.<ext>'; MIDI vira 'score.mid'
    dest = dest_dir / ("score" + (".mid" if src.suffix.lower() in {".mid", ".midi"}
                                  else src.suffix.lower()))
    shutil.copy2(src, dest)
    if args.lyrics:
        shutil.copy2(args.lyrics, dest_dir / "lyrics.txt")
    print(f"[import] copiado para {dest}")

    perf = build_performance(dest_dir)
    print(f"[import] {perf.summary()}")
    rep = validate(perf)
    print("[validação]")
    print(rep.render())

    if not rep.ok:
        print("\n-> corrija os ERROS acima antes de usar no pipeline.")
        raise SystemExit(1)
    print(f"\n[OK] pronto para usar:  python -m src.pipeline --song {dest_dir} --voice <nome>")


if __name__ == "__main__":
    main()
