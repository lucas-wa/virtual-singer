"""Baixa uma voz pronta da VocalSet (CC BY 4.0) para testar a pipeline sem gravar a sua.

Em vez de baixar os ~2-6 GB do dataset inteiro, usa `remotezip` para pegar SÓ os
arquivos do cantor escolhido (dezenas de MB) via HTTP range, e os coloca (achatados)
em data/voices/<nome>/ — exatamente no mesmo formato da sua própria voz.

Assim você ALTERNA entre as vozes só trocando o --voice no pipeline:
    python -m src.pipeline --song data/songs/demo --voice vocalset_female1 ...
    python -m src.pipeline --song data/songs/demo --voice minha_voz ...

Fonte: VocalSet — Wilkins et al., CC BY 4.0 — https://zenodo.org/record/1442513

Uso:
    python scripts/get_sample_voice.py --list                 # lista os cantores
    python scripts/get_sample_voice.py --singer female1       # baixa o cantor female1
    python scripts/get_sample_voice.py --singer male1 --name voz_teste --max-files 60
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import paths  # noqa: E402

# VocalSet11.zip (~2 GB) tem a árvore FULL/<cantor>/.../*.wav e é menor que o 1-2.zip.
ZENODO_URL = "https://zenodo.org/api/records/1442513/files/VocalSet11.zip/content"
ROOT_PREFIX = "FULL/"


def _open_remote():
    try:
        from remotezip import RemoteZip
    except ImportError:
        raise SystemExit(
            "Falta a dependência 'remotezip'. Instale com:  pip install remotezip"
        )
    return RemoteZip(ZENODO_URL)


def _singer_of(name: str) -> str | None:
    # ex.: "FULL/female9/excerpts/vibrato/f9_caro_vibrato.wav" -> "female9"
    parts = name.split("/")
    if len(parts) > 2 and parts[0] == "FULL" and parts[1]:
        return parts[1]
    return None


def list_singers() -> None:
    print("[vocalset] lendo índice remoto (sem baixar o dataset inteiro)...")
    with _open_remote() as z:
        singers = sorted({s for s in (_singer_of(n) for n in z.namelist()) if s})
    print(f"[vocalset] {len(singers)} cantores disponíveis:")
    print("  " + ", ".join(singers))


def fetch_singer(singer: str, name: str, max_files: int | None) -> Path:
    dest = paths.VOICES / name
    dest.mkdir(parents=True, exist_ok=True)

    print(f"[vocalset] baixando os .wav de '{singer}' (apenas este cantor)...")
    prefix = f"{ROOT_PREFIX}{singer}/"
    total_bytes = 0
    saved = 0
    with _open_remote() as z:
        members = [n for n in z.namelist()
                   if n.startswith(prefix) and n.lower().endswith(".wav")]
        if not members:
            raise SystemExit(
                f"nenhum .wav encontrado para '{singer}'. Use --list para ver os nomes."
            )
        if max_files:
            members = members[:max_files]
        for m in members:
            # achata: FULL/female9/excerpts/vibrato/x.wav -> excerpts_vibrato_x.wav
            rel = m[len(prefix):].replace("/", "_")
            out = dest / rel
            data = z.read(m)
            out.write_bytes(data)
            total_bytes += len(data)
            saved += 1
    print(f"[vocalset] {saved} arquivos salvos em {dest} ({total_bytes/1e6:.1f} MB)")
    _report_duration(dest)
    return dest


def _report_duration(folder: Path) -> None:
    try:
        import soundfile as sf
    except ImportError:
        return
    secs = 0.0
    for w in folder.glob("*.wav"):
        try:
            info = sf.info(str(w))
            secs += info.frames / info.samplerate
        except Exception:  # noqa: BLE001
            pass
    print(f"[vocalset] duração total: ~{secs/60:.1f} min "
          f"(RVC: 5-15 min costuma bastar; use --max-files para limitar)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="listar cantores e sair")
    ap.add_argument("--singer", default="female1", help="cantor (ex.: female1, male1)")
    ap.add_argument("--name", default=None, help="nome da voz (default: vocalset_<singer>)")
    ap.add_argument("--max-files", type=int, default=None,
                    help="limitar o número de arquivos baixados (menos minutos)")
    args = ap.parse_args()

    paths.ensure_dirs()
    if args.list:
        list_singers()
        return

    name = args.name or f"vocalset_{args.singer}"
    dest = fetch_singer(args.singer, name, args.max_files)
    print(f"\n[OK] voz de teste pronta: {dest}")
    print(f"Treine com:   python scripts/train_voice.py --voice {dest}")
    print(f"Depois alterne com --voice {name} (ou --voice <sua_voz>) no pipeline.")


if __name__ == "__main__":
    main()
