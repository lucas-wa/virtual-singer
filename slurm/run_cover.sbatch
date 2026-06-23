#!/bin/bash
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --ntasks=1
#SBATCH --time=UNLIMITED
#SBATCH --nodelist=dgx-H100-02
#SBATCH --job-name=vsinger-cover
#SBATCH --output=logs/cover_%j.out
#SBATCH --error=logs/cover_%j.err

# ============================================================================
# Cover (conversão de timbre): pega as músicas de um cantor FONTE e as converte para a
# voz de um cantor ALVO, via Seed-VC zero-shot. Container py3.12 (mesmo do áudio).
#
# Estrutura esperada (abstraída por SOURCE_SINGER/TARGET_SINGER):
#   $CANTORES_ROOT/<source_singer>/*.mp3   (músicas a converter)
#   $CANTORES_ROOT/<target_singer>/*.mp3   (voz cujo timbre queremos)
#
# Defina no slurm/.env:
#   SOURCE_SINGER=priscila   TARGET_SINGER=lucas   (troque à vontade)
# ============================================================================

CONTAINER_IMAGE="${CONTAINER_IMAGE:-/raid/user_lucasalves/images/vllm-openai_nightly-2e36cdbe2b4bff4370a02105fdfbdd015d6ec4e8.sif}"
HOST_RAID="${HOST_RAID:-/raid/user_lucasalves}"
PROJECT_SUBDIR="${PROJECT_SUBDIR:-ufg/pav/virtual-singer}"

mkdir -p logs
echo "=== Cover job $SLURM_JOB_ID em $(hostname) | $(date) ==="

ENV_FILE="${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}/slurm/.env"
[ -f "$ENV_FILE" ] || ENV_FILE="$(dirname "$0")/.env"
[ -f "$ENV_FILE" ] && { echo "Carregando $ENV_FILE"; set -a; source "$ENV_FILE"; set +a; }

srun singularity exec \
    --nv \
    --no-home \
    --bind "${HOST_RAID}:/work" \
    "$CONTAINER_IMAGE" \
    bash -c '
        set -euo pipefail

        export HOME=/work/.home
        export PIP_CACHE_DIR=/work/.cache/pip
        export HF_HOME=/work/.cache/huggingface
        export NLTK_DATA=/work/.home/nltk_data
        mkdir -p "$HOME" "$PIP_CACHE_DIR" "$HF_HOME" "$NLTK_DATA"
        export PATH="/work/.home/.local/bin:${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export PIP_RETRIES=10
        export PIP_DEFAULT_TIMEOUT=120

        cd "/work/'"$PROJECT_SUBDIR"'"
        git pull --ff-only 2>/dev/null || true

        # ---- PARÂMETROS (defina no slurm/.env) ----
        export SOURCE_SINGER="${SOURCE_SINGER:-priscila}"   # de quem são as músicas
        export TARGET_SINGER="${TARGET_SINGER:-lucas}"      # voz que vai cantar
        export CANTORES_ROOT="${CANTORES_ROOT:-cantores}"   # raiz das pastas de cantores
        export COVER_LIMIT="${COVER_LIMIT:-0}"              # nº máx. de músicas (0=todas)
        export COVER_SEMITONE="${COVER_SEMITONE:-0}"        # transposição
        export COVER_SEPARATE="${COVER_SEPARATE:-1}"        # 1=separar vocais (Demucs) | 0=não
        export COVER_REMIX="${COVER_REMIX:-1}"              # 1=remixar c/ instrumental | 0=só vocal
        # -------------------------------------------

        # Deps: mesmo esquema do job de áudio (Seed-VC + Demucs, py3.12) — 1x e cacheia.
        # marker v2: o demucs foi ADICIONADO aos requirements -> força reinstalar uma vez.
        DEPS_MARKER="/work/.home/.deps_ok_v2"
        [ -n "${REINSTALL:-}" ] && rm -f "$DEPS_MARKER"
        if [ ! -f "$DEPS_MARKER" ]; then
            python3 -m pip install -U pip setuptools wheel
            python3 -m pip install --no-build-isolation --prefer-binary -r requirements.txt
        fi
        python3 scripts/setup_models.py            # garante o repo do Seed-VC (idempotente)
        if [ ! -f "$DEPS_MARKER" ]; then
            [ -f third_party/seed-vc/requirements.txt ] && \
                python3 -m pip install --no-build-isolation --prefer-binary \
                    -r third_party/seed-vc/requirements.txt
            touch "$DEPS_MARKER"
        fi

        # Expõe o ffmpeg empacotado (imageio-ffmpeg) como "ffmpeg" no PATH, p/ Demucs/librosa
        # lerem MP3 sem ffmpeg do sistema. Idempotente.
        FFMPEG_BIN="$(python3 -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" 2>/dev/null || true)"
        if [ -n "$FFMPEG_BIN" ] && [ -x "$FFMPEG_BIN" ]; then
            ln -sf "$FFMPEG_BIN" /work/.home/.local/bin/ffmpeg
            echo "[cover] ffmpeg -> $FFMPEG_BIN"
        fi

        COVER_ARGS=""
        [ "$COVER_SEPARATE" = "0" ] && COVER_ARGS="$COVER_ARGS --no-separate"
        [ "$COVER_REMIX" = "0" ]    && COVER_ARGS="$COVER_ARGS --no-remix"

        echo "[cover] $SOURCE_SINGER -> $TARGET_SINGER (raiz: $CANTORES_ROOT) sep=$COVER_SEPARATE remix=$COVER_REMIX"
        python3 scripts/cover.py \
            --songs-dir "$CANTORES_ROOT/$SOURCE_SINGER" \
            --voice-dir "$CANTORES_ROOT/$TARGET_SINGER" \
            --out-dir "out/covers_${TARGET_SINGER}" \
            --limit "$COVER_LIMIT" \
            --semitone "$COVER_SEMITONE" \
            $COVER_ARGS
    '

echo "=== Concluído: $(date) ==="
