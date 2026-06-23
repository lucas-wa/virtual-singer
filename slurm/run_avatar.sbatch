#!/bin/bash
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --ntasks=1
#SBATCH --time=UNLIMITED
#SBATCH --nodelist=dgx-H100-02
#SBATCH --job-name=vsinger-avatar
#SBATCH --output=logs/avatar_%j.out
#SBATCH --error=logs/avatar_%j.err

# ============================================================================
# Avatar (SadTalker): foto + áudio JÁ gerado -> vídeo .mp4.
#
# Roda SEPARADO do job de áudio porque o SadTalker exige Python 3.10 (no py3.12 do
# container vllm ele nem instala). Crie uma imagem py3.10 uma vez:
#   singularity pull /raid/user_lucasalves/images/pytorch_2.2_cu121_py310.sif \
#       docker://pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
#
# Pré-requisitos:
#   1. o áudio já existe (rode antes o run_virtual_singer.sbatch);
#   2. sua foto está em data/faces/<arquivo> (suba via scp — veja o chat).
# ESCOPO: rosto próprio/sintético/consentido — nunca pessoa real sem consentimento.
# ============================================================================

CONTAINER_IMAGE="${AVATAR_CONTAINER_IMAGE:-/raid/user_lucasalves/images/pytorch_2.2_cu121_py310.sif}"
HOST_RAID="${HOST_RAID:-/raid/user_lucasalves}"
PROJECT_SUBDIR="${PROJECT_SUBDIR:-ufg/pav/virtual-singer}"

mkdir -p logs
echo "=== Avatar job $SLURM_JOB_ID em $(hostname) | $(date) ==="
echo "Container (py3.10): $CONTAINER_IMAGE"

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
        export PIP_RETRIES=10
        export PIP_DEFAULT_TIMEOUT=120

        cd "/work/'"$PROJECT_SUBDIR"'"
        git pull --ff-only 2>/dev/null || true

        # ---- PARÂMETROS (defina no slurm/.env) ----
        export AVATAR_IMAGE="${AVATAR_IMAGE:?defina AVATAR_IMAGE no slurm/.env (foto do rosto)}"
        export AVATAR_AUDIO="${AVATAR_AUDIO:-out/vocalset_female1_demo.wav}"
        export AVATAR_OUT="${AVATAR_OUT:-out/avatar.mp4}"
        # -------------------------------------------

        # Markers próprios do ambiente do avatar (py3.10), separado do job de áudio.
        AV_MARKER="/work/.home/.avatar310_deps_ok_v1"
        [ -n "${REINSTALL:-}" ] && rm -f "$AV_MARKER"

        # Instala as deps UMA vez (persistem em /work/.home/.local; wheels em PIP_CACHE_DIR).
        # Marker evita reinstalar a cada run. Para forçar: REINSTALL=1 no slurm/.env.
        if [ ! -f "$AV_MARKER" ]; then
            echo "[avatar] >>> instalando deps (1ª vez; cache em /work)..."
            python3 -m pip install -U pip setuptools wheel
            python3 -m pip install --prefer-binary huggingface_hub requests
        else
            echo "[avatar] deps já instaladas (marker presente) — pulando pip. REINSTALL=1 força."
        fi

        # Clona o SadTalker + baixa os checkpoints (idempotente: pula se já existem; HF cacheado).
        python3 scripts/setup_models.py --with-avatar

        # Requirements do SadTalker — só na 1ª vez (no py3.10 instalam normalmente).
        if [ ! -f "$AV_MARKER" ]; then
            echo "[avatar] >>> instalando requirements do SadTalker..."
            python3 -m pip install --prefer-binary -r third_party/SadTalker/requirements.txt
            touch "$AV_MARKER"
        fi

        # SEMPRE garante (fora do marker, idempotente):
        #  - OpenCV HEADLESS (container sem libs gráficas libgthread/libGL);
        #  - numpy<2: o SadTalker (scipy 1.10.1) e o torch do container foram compilados
        #    p/ numpy 1.x; o opencv-headless NOVO exige numpy>=2 -> fixamos a 4.9 (numpy 1.x).
        if ! python3 -c "import cv2, numpy, sys; sys.exit(0 if int(numpy.__version__.split('.')[0])<2 else 1)" 2>/dev/null; then
            echo "[avatar] >>> ajustando OpenCV headless + numpy<2..."
            python3 -m pip uninstall -y opencv-python opencv-contrib-python 2>/dev/null || true
            python3 -m pip install --prefer-binary "opencv-python-headless==4.9.0.80" "numpy<2"
        fi

        echo "[avatar] gerando vídeo: $AVATAR_IMAGE + $AVATAR_AUDIO -> $AVATAR_OUT"
        python3 scripts/make_avatar.py --image "$AVATAR_IMAGE" \
            --audio "$AVATAR_AUDIO" --out "$AVATAR_OUT"
    '

echo "=== Concluído: $(date) ==="
