#!/bin/bash
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --ntasks=1
#SBATCH --time=UNLIMITED
#SBATCH --nodelist=dgx-H100-02
#SBATCH --job-name=virtual-singer
#SBATCH --output=logs/vsinger_%j.out
#SBATCH --error=logs/vsinger_%j.err

# ============================================================================
# Cantor Virtual — treino RVC + síntese (+ avatar opcional) numa H100.
#
# Imagem do container: precisa de PyTorch + CUDA 12 + Python 3.10 (fairseq exige 3.10).
# Crie uma vez (exemplo):
#   singularity pull /raid/user_lucasalves/images/pytorch_2.2_cu121_py310.sif \
#       docker://pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
#
# Coloque o repositório em $HOST_RAID/$PROJECT_SUBDIR (ex.: /raid/user_lucasalves/virtual-singer).
# Rode:  sbatch slurm/run_virtual_singer.sbatch
# ============================================================================

CONTAINER_IMAGE="${CONTAINER_IMAGE:-/raid/user_lucasalves/images/vllm-openai_nightly-2e36cdbe2b4bff4370a02105fdfbdd015d6ec4e8.sif}"
HOST_RAID="${HOST_RAID:-/raid/user_lucasalves}"
PROJECT_SUBDIR="${PROJECT_SUBDIR:-ufg/pav/virtual-singer}"

mkdir -p logs

echo "=== Job $SLURM_JOB_ID em $(hostname) | $(date) ==="
echo "Container: $CONTAINER_IMAGE"
echo "Projeto:   $HOST_RAID/$PROJECT_SUBDIR"

# Carrega slurm/.env se existir (nunca commitar esse arquivo).
# No SLURM, $0 pode ser uma cópia no spool — preferimos $SLURM_SUBMIT_DIR.
ENV_FILE="${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}/slurm/.env"
[ -f "$ENV_FILE" ] || ENV_FILE="$(dirname "$0")/.env"
if [ -f "$ENV_FILE" ]; then
    echo "Carregando $ENV_FILE"
    set -a; source "$ENV_FILE"; set +a   # set -a => toda variável do .env é exportada
fi

srun singularity exec \
    --nv \
    --no-home \
    --bind "${HOST_RAID}:/work" \
    "$CONTAINER_IMAGE" \
    bash -c '
        set -euo pipefail

        # --no-home torna /home read-only; redireciona HOME/caches para /work
        export HOME=/work/.home
        export PIP_CACHE_DIR=/work/.cache/pip
        export HF_HOME=/work/.cache/huggingface
        export NLTK_DATA=/work/.home/nltk_data
        mkdir -p "$HOME" "$PIP_CACHE_DIR" "$HF_HOME" "$NLTK_DATA"
        export PATH="/work/.home/.local/bin:${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"

        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

        cd "/work/'"$PROJECT_SUBDIR"'"
        git pull --ff-only 2>/dev/null || true

        # ---- PARÂMETROS DO EXPERIMENTO (defina no slurm/.env; defaults abaixo) ----
        # Vêm do .env (sourced no shell externo e propagado pelo Singularity).
        export SONG="${SONG:-demo}"                          # pasta em data/songs/
        export VOICE_NAME="${VOICE_NAME:-vocalset_female1}"
        export VOICE_SOURCE="${VOICE_SOURCE:-vocalset:female1}"  # vocalset:<singer> | path:<dir> | existing
        export ENGINE="${ENGINE:-dsp}"                       # dsp (sempre roda) | diffsinger
        export AVATAR_IMAGE="${AVATAR_IMAGE:-}"              # vazio | "synthetic" | caminho de imagem
        export OUT="${OUT:-out/${VOICE_NAME}_${SONG}.wav}"
        # --------------------------------------------------------------------------

        # Dependências: o container já traz torch+CUDA (não reinstalar).
        # PADRÃO usa Seed-VC (timbre zero-shot, SEM fairseq) -> roda em Python 3.12.
        python3 -m pip install --quiet -r requirements.txt

        # Baixa o que o experimento precisa (avatar permitido aqui — não é Colab).
        SETUP_FLAGS=""
        [ "$ENGINE" = "diffsinger" ] && SETUP_FLAGS="$SETUP_FLAGS --with-diffsinger"
        [ -n "$AVATAR_IMAGE" ]       && SETUP_FLAGS="$SETUP_FLAGS --with-avatar"
        python3 scripts/setup_models.py $SETUP_FLAGS

        # Instala as deps do Seed-VC (do próprio repo clonado).
        [ -f third_party/seed-vc/requirements.txt ] && \
            python3 -m pip install --quiet -r third_party/seed-vc/requirements.txt

        # Pipeline ponta a ponta: voz-guia (DSP) -> timbre (Seed-VC) -> [avatar].
        bash scripts/run_all.sh
    '

echo "=== Concluído: $(date) ==="
