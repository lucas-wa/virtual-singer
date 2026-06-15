#!/usr/bin/env bash
# Runner headless da pipeline do Cantor Virtual (para SLURM/Singularity ou local).
# Faz: (partitura) -> [voz pronta opcional] -> treino RVC -> síntese -> [avatar].
# Tudo parametrizado por variáveis de ambiente (com defaults sensatos).
set -euo pipefail

# ---- parâmetros (sobrescreva via env) --------------------------------------
SONG="${SONG:-demo}"                       # pasta em data/songs/
VOICE_NAME="${VOICE_NAME:-vocalset_female1}"
VOICE_SOURCE="${VOICE_SOURCE:-vocalset:female1}"  # vocalset:<singer> | path:<dir> | existing
ENGINE="${ENGINE:-dsp}"                    # dsp (padrão) | diffsinger
EPOCHS="${EPOCHS:-100}"                    # épocas de treino do RVC
MAX_FILES="${MAX_FILES:-80}"               # limite de arquivos da VocalSet
AVATAR_IMAGE="${AVATAR_IMAGE:-}"           # caminho de imagem | "synthetic" | vazio
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

# ---- 4. treinar o RVC (pula se já houver modelo) ----------------------------
if [ -f "models/voices/$VOICE_NAME/model.pth" ]; then
    echo "[rvc] modelo '$VOICE_NAME' já existe — pulando treino"
else
    python3 scripts/train_voice.py --voice "data/voices/$VOICE_NAME" --name "$VOICE_NAME" --epochs "$EPOCHS"
fi

# ---- 5. avatar opcional -----------------------------------------------------
AVATAR_ARGS=()
if [ "$AVATAR_IMAGE" = "synthetic" ]; then
    python3 scripts/get_face.py --name avatar_synthetic
    AVATAR_ARGS=(--avatar-image data/faces/avatar_synthetic.jpg)
elif [ -n "$AVATAR_IMAGE" ]; then
    AVATAR_ARGS=(--avatar-image "$AVATAR_IMAGE")
fi

# ---- 6. síntese final -------------------------------------------------------
python3 -m src.pipeline \
    --song "data/songs/$SONG" \
    --voice "$VOICE_NAME" \
    --engine "$ENGINE" \
    --transpose "$TRANSPOSE" \
    --out "$OUT" \
    "${AVATAR_ARGS[@]}"

echo "=== run_all concluído -> $OUT ==="
