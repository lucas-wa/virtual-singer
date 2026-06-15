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

CONTAINER_IMAGE="${CONTAINER_IMAGE:-/raid/user_lucasalves/images/pytorch_2.2_cu121_py310.sif}"
HOST_RAID="${HOST_RAID:-/raid/user_lucasalves}"
PROJECT_SUBDIR="${PROJECT_SUBDIR:-virtual-singer}"

mkdir -p logs

echo "=== Job $SLURM_JOB_ID em $(hostname) | $(date) ==="
echo "Container: $CONTAINER_IMAGE"
echo "Projeto:   $HOST_RAID/$PROJECT_SUBDIR"

# Carrega slurm/.env se existir (HF_TOKEN é opcional; nunca commitar esse arquivo)
ENV_FILE="$(dirname "$0")/.env"
if [ -f "$ENV_FILE" ]; then
    echo "Carregando $ENV_FILE"
    set -a; source "$ENV_FILE"; set +a
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

        # ---------------- PARÂMETROS DO EXPERIMENTO (edite à vontade) -----------
        export SONG=demo                       # pasta em data/songs/
        export VOICE_NAME=vocalset_female1
        export VOICE_SOURCE=vocalset:female1   # vocalset:<singer> | path:<dir> | existing
        export ENGINE=dsp                      # dsp (sempre roda) | diffsinger
        export EPOCHS=100
        export AVATAR_IMAGE=                    # vazio=sem vídeo | "synthetic" | caminho de imagem
        export OUT=out/${VOICE_NAME}_${SONG}.wav
        # ------------------------------------------------------------------------

        # Dependências: o container já traz torch+CUDA (não reinstalar). fairseq exige
        # pip<24.1 (metadata do omegaconf<2.1). faiss/fairseq só p/ o RVC.
        python3 -m pip install --quiet "pip<24.1"
        python3 -m pip install --quiet -r requirements.txt
        python3 -m pip install --quiet fairseq==0.12.2 faiss-cpu

        # Baixa modelos conforme o experimento (avatar permitido aqui — não é Colab).
        SETUP_FLAGS=""
        [ "$ENGINE" = "diffsinger" ] && SETUP_FLAGS="$SETUP_FLAGS --with-diffsinger"
        [ -n "$AVATAR_IMAGE" ]       && SETUP_FLAGS="$SETUP_FLAGS --with-avatar"
        python3 scripts/setup_models.py $SETUP_FLAGS

        # Pipeline ponta a ponta (treino RVC + síntese + avatar opcional).
        bash scripts/run_all.sh
    '

echo "=== Concluído: $(date) ==="
