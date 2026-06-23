#!/usr/bin/env bash
# Runner headless de ÁUDIO do Cantor Virtual (SLURM/Singularity ou local).
# Faz: (partitura) -> [voz pronta opcional] -> síntese (DSP + Seed-VC zero-shot) -> .wav
# O AVATAR é um job/etapa SEPARADA (scripts/make_avatar.py / slurm/run_avatar.sbatch),
# porque o SadTalker exige Python 3.10. Aqui é só áudio.
set -euo pipefail

# ---- parâmetros (sobrescreva via env) --------------------------------------
SONG="${SONG:-demo}"                       # pasta em data/songs/
VOICE_NAME="${VOICE_NAME:-vocalset_female1}"
VOICE_SOURCE="${VOICE_SOURCE:-vocalset:female1}"  # vocalset:<singer> | path:<dir> | existing
ENGINE="${ENGINE:-dsp}"                    # dsp (padrão) | diffsinger
MAX_FILES="${MAX_FILES:-80}"               # limite de arquivos da VocalSet
TRANSPOSE="${TRANSPOSE:-0}"
OUT="${OUT:-out/${VOICE_NAME}_${SONG}.wav}"

echo "=== run_all | song=$SONG voice=$VOICE_NAME source=$VOICE_SOURCE engine=$ENGINE ==="

# ---- 1. partitura de demo, se necessário -----------------------------------
if [ "$SONG" = "demo" ] && [ ! -f data/songs/demo/score.mid ]; then
    python3 scripts/make_demo_song.py
fi

# ---- 2. dados do G2P (nltk) -------------------------------------------------
python3 - <<'PY'
import nltk
for p in ["averaged_perceptron_tagger_eng", "cmudict", "punkt", "punkt_tab"]:
    nltk.download(p, quiet=True)
print("[nltk] dados do G2P prontos")
PY

# ---- 3. preparar a voz ------------------------------------------------------
case "$VOICE_SOURCE" in
    vocalset:*)
        SINGER="${VOICE_SOURCE#vocalset:}"
        python3 scripts/get_sample_voice.py --singer "$SINGER" --name "$VOICE_NAME" --max-files "$MAX_FILES"
        ;;
    path:*)
        SRC="${VOICE_SOURCE#path:}"
        mkdir -p "data/voices/$VOICE_NAME"
        cp -r "$SRC"/. "data/voices/$VOICE_NAME/"
        ;;
    existing)
        echo "[voz] usando data/voices/$VOICE_NAME (já presente)"
        ;;
    *)
        echo "VOICE_SOURCE inválido: $VOICE_SOURCE" >&2; exit 1 ;;
esac

# ---- 4. (sem treino) — Seed-VC é zero-shot: usa os clipes de referência direto ----
echo "[seed-vc] timbre zero-shot — sem etapa de treino; referência em data/voices/$VOICE_NAME"

# ---- 5. síntese (áudio) -----------------------------------------------------
python3 -m src.pipeline \
    --song "data/songs/$SONG" \
    --voice "$VOICE_NAME" \
    --engine "$ENGINE" \
    --transpose "$TRANSPOSE" \
    --out "$OUT"

echo "=== run_all concluído -> $OUT ==="
echo "Para o avatar (rosto cantando): rode o job slurm/run_avatar.sbatch (container py3.10)."
