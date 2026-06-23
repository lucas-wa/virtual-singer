#!/bin/bash
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --ntasks=1
#SBATCH --time=UNLIMITED
#SBATCH --nodelist=dgx-H100-02
#SBATCH --job-name=vsinger-cover-ying
#SBATCH --output=logs/cover_ying/cy_%j.out
#SBATCH --error=logs/cover_ying/cy_%j.err

# ============================================================================
# Cover com YingMusic-SVC (SVC zero-shot SOTA). EXIGE Python 3.10 -> usa o MESMO
# container do avatar (pytorch py3.10), NÃO o vllm py3.12.
#
# Fluxo: Demucs separa (vocal+instrumental) -> YingMusic converte no timbre do alvo
# (recebe vocal + acompanhamento e devolve a mix convertida).
#
# Defina no slurm/.env (igual ao cover):  SOURCE_SINGER, TARGET_SINGER, CANTORES_ROOT
# ============================================================================

# YingMusic compila deps em C -> precisa de gcc. A imagem -devel (não -runtime) tem.
# Crie uma vez:
#   singularity pull /raid/user_lucasalves/images/pytorch_2.2_cu121_py310_devel.sif \
#       docker://pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel
# e aponte YING_CONTAINER_IMAGE para ela (no slurm/.env). Default cai na py3.10 que você já tem.
CONTAINER_IMAGE="${YING_CONTAINER_IMAGE:-${AVATAR_CONTAINER_IMAGE:-/raid/user_lucasalves/images/pytorch_2.2_cu121_py310.sif}}"
HOST_RAID="${HOST_RAID:-/raid/user_lucasalves}"
PROJECT_SUBDIR="${PROJECT_SUBDIR:-ufg/pav/virtual-singer}"

mkdir -p logs/cover_ying
echo "=== Cover-YingMusic job $SLURM_JOB_ID em $(hostname) | $(date) ==="
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
        mkdir -p "$HOME" "$PIP_CACHE_DIR" "$HF_HOME"
        export PATH="/work/.home/.local/bin:${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export PIP_RETRIES=10
        export PIP_DEFAULT_TIMEOUT=120

        cd "/work/'"$PROJECT_SUBDIR"'"
        git pull --ff-only 2>/dev/null || true

        # ---- PARÂMETROS (slurm/.env) ----
        export SOURCE_SINGER="${SOURCE_SINGER:-priscila}"
        export TARGET_SINGER="${TARGET_SINGER:-lucas}"
        export CANTORES_ROOT="${CANTORES_ROOT:-singers}"
        export COVER_LIMIT="${COVER_LIMIT:-0}"
        export COVER_STEPS="${COVER_STEPS:-100}"
        export COVER_REMIX="${COVER_REMIX:-1}"
        # ----------------------------------

        # Deps do ambiente py3.10 do YingMusic (marker próprio).
        YM_MARKER="/work/.home/.ying310_deps_ok_v1"
        [ -n "${REINSTALL:-}" ] && rm -f "$YM_MARKER"
        if [ ! -f "$YM_MARKER" ]; then
            python3 -m pip install -U pip setuptools wheel
            python3 -m pip install --prefer-binary -r requirements.txt
        fi
        # Clona YingMusic + baixa checkpoints (idempotente).
        python3 scripts/setup_models.py --with-yingmusic
        # Deps do REPO YingMusic — SEMPRE (fora do marker): repo de pesquisa com
        # requirements às vezes incompleto, e um marker antigo pulava esta etapa.
        # Idempotente (pip não reinstala o que já está satisfeito).
        if [ -f third_party/YingMusic-SVC/requirements.txt ]; then
            # O container py3.10 (runtime) NÃO tem gcc -> deps que compilam C falham.
            # webrtcvad compila; trocamos pelo wheel pré-compilado webrtcvad-wheels.
            grep -ivE "^webrtcvad([=<>! ]|$)" third_party/YingMusic-SVC/requirements.txt \
                > /tmp/ying_reqs.txt || cp third_party/YingMusic-SVC/requirements.txt /tmp/ying_reqs.txt
            python3 -m pip install --prefer-binary -r /tmp/ying_reqs.txt
            python3 -m pip install --prefer-binary webrtcvad-wheels
        fi
        # deps comuns que faltam em requirements de pesquisa:
        python3 -m pip install --prefer-binary munch
        touch "$YM_MARKER"

        # ffmpeg empacotado no PATH + patch do Demucs (in-place fix).
        FF="$(python3 -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" 2>/dev/null || true)"
        [ -n "$FF" ] && [ -x "$FF" ] && ln -sf "$FF" /work/.home/.local/bin/ffmpeg
        python3 scripts/patch_demucs.py

        COVER_ARGS=""
        [ "$COVER_REMIX" = "0" ] && COVER_ARGS="--no-remix"

        echo "[cover-ying] $SOURCE_SINGER -> $TARGET_SINGER | steps=$COVER_STEPS"
        python3 scripts/cover.py \
            --engine yingmusic \
            --songs-dir "$CANTORES_ROOT/$SOURCE_SINGER" \
            --voice-dir "$CANTORES_ROOT/$TARGET_SINGER" \
            --out-dir "out/covers_ying_${TARGET_SINGER}" \
            --limit "$COVER_LIMIT" \
            --diffusion-steps "$COVER_STEPS" \
            $COVER_ARGS
    '

echo "=== Concluído: $(date) ==="
