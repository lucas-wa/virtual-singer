#!/bin/bash
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --ntasks=1
#SBATCH --time=UNLIMITED
#SBATCH --nodelist=dgx-H100-02
#SBATCH --job-name=vsinger-finetune
#SBATCH --output=logs/finetune/ft_%j.out
#SBATCH --error=logs/finetune/ft_%j.err

# ============================================================================
# Fine-tune do Seed-VC na voz de um cantor (melhora a fidelidade do timbre).
# Container py3.12 (mesmo do cover). Treina com clipes de 1-30s de VOZ LIMPA.
#
# Defina no slurm/.env:
#   FT_SINGER=lucas          # de quem é a voz a treinar (pasta $CANTORES_ROOT/<FT_SINGER>)
#   FT_MAX_STEPS=1000        # passos de treino (100 já dá um começo; 1000+ melhor)
#   FT_SEPARATE=1            # 1=Demucs p/ limpar a voz antes | 0=já é voz limpa
# Depois aponte o cover para o checkpoint (o log imprime COVER_CHECKPOINT/COVER_CONFIG).
# ============================================================================

CONTAINER_IMAGE="${CONTAINER_IMAGE:-/raid/user_lucasalves/images/vllm-openai_nightly-2e36cdbe2b4bff4370a02105fdfbdd015d6ec4e8.sif}"
HOST_RAID="${HOST_RAID:-/raid/user_lucasalves}"
PROJECT_SUBDIR="${PROJECT_SUBDIR:-ufg/pav/virtual-singer}"

mkdir -p logs/finetune
echo "=== Finetune job $SLURM_JOB_ID em $(hostname) | $(date) ==="

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
        export FT_SINGER="${FT_SINGER:-lucas}"
        export CANTORES_ROOT="${CANTORES_ROOT:-singers}"
        export FT_MAX_STEPS="${FT_MAX_STEPS:-1000}"
        export FT_SEPARATE="${FT_SEPARATE:-1}"
        # -------------------------------------------

        # Deps (mesmo marker do cover; Seed-VC + Demucs).
        DEPS_MARKER="/work/.home/.deps_ok_v2"
        [ -n "${REINSTALL:-}" ] && rm -f "$DEPS_MARKER"
        if [ ! -f "$DEPS_MARKER" ]; then
            python3 -m pip install -U pip setuptools wheel
            python3 -m pip install --no-build-isolation --prefer-binary -r requirements.txt
        fi
        python3 scripts/setup_models.py
        if [ ! -f "$DEPS_MARKER" ]; then
            [ -f third_party/seed-vc/requirements.txt ] && \
                python3 -m pip install --no-build-isolation --prefer-binary \
                    -r third_party/seed-vc/requirements.txt
            touch "$DEPS_MARKER"
        fi

        # ffmpeg empacotado no PATH (ler MP3).
        FF="$(python3 -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" 2>/dev/null || true)"
        [ -n "$FF" ] && [ -x "$FF" ] && ln -sf "$FF" /work/.home/.local/bin/ffmpeg

        # Corrige o bug in-place do Demucs com PyTorch novo (idempotente).
        python3 scripts/patch_demucs.py

        FT_ARGS=""
        [ "$FT_SEPARATE" = "0" ] && FT_ARGS="--no-separate"

        echo "[ft] treinando voz de $FT_SINGER ($FT_MAX_STEPS passos)"
        python3 scripts/finetune_voice.py \
            --voice-dir "$CANTORES_ROOT/$FT_SINGER" \
            --run-name "$FT_SINGER" \
            --max-steps "$FT_MAX_STEPS" \
            $FT_ARGS
    '

echo "=== Concluído: $(date) ==="
