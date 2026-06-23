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
# Cantor Virtual — síntese de canto (+ avatar opcional) numa H100.
#
# Caminho PADRÃO: voz-guia DSP + timbre Seed-VC (zero-shot, SEM treino, SEM fairseq).
# Roda em container PyTorch+CUDA com Python 3.12 (ex.: a imagem vllm). Só o caminho RVC
# opcional (--with-rvc) exigiria Python 3.10.
#
# Coloque o repositório em $HOST_RAID/$PROJECT_SUBDIR e rode:
#   sbatch slurm/run_virtual_singer.sbatch
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

        # Rede do nó de cómputo pode cair em downloads grandes -> pip resiliente.
        export PIP_RETRIES=10
        export PIP_DEFAULT_TIMEOUT=120

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

        # ---- Dependências: instaladas UMA vez; persistem no volume -----------------
        # Com --no-home o pip instala em $HOME/.local = /work/.home/.local (no /raid,
        # persistente). Wheels ficam em PIP_CACHE_DIR. Um "marker" evita reinstalar a
        # cada run. Para forçar reinstalação: submeta com REINSTALL=1 no slurm/.env.
        DEPS_MARKER="/work/.home/.deps_ok_v1"
        AVATAR_DEPS_MARKER="/work/.home/.avatar_deps_ok_v1"
        [ -n "${REINSTALL:-}" ] && rm -f "$DEPS_MARKER" "$AVATAR_DEPS_MARKER"

        # requirements.txt precisa estar instalado ANTES do setup_models (usa huggingface_hub).
        if [ ! -f "$DEPS_MARKER" ]; then
            python3 -m pip install -r requirements.txt
        else
            echo "[deps] já instaladas (marker presente) — pulando pip. REINSTALL=1 força."
        fi

        # Baixa repos/modelos (idempotente: git clone e downloads HF pulam se já existem).
        SETUP_FLAGS="--with-avatar --with-diffsinger"
        [ "$ENGINE" = "diffsinger" ] && SETUP_FLAGS="$SETUP_FLAGS --with-diffsinger"
        [ -n "$AVATAR_IMAGE" ]       && SETUP_FLAGS="$SETUP_FLAGS --with-avatar"
        python3 scripts/setup_models.py $SETUP_FLAGS

        # Deps do Seed-VC (do repo clonado) — mesma marca; finaliza o marker ao concluir.
        if [ ! -f "$DEPS_MARKER" ]; then
            [ -f third_party/seed-vc/requirements.txt ] && \
                python3 -m pip install -r third_party/seed-vc/requirements.txt
            touch "$DEPS_MARKER"
        fi

        # Deps do SadTalker (só se avatar ligado), com marker próprio.
        if [ -n "$AVATAR_IMAGE" ] && [ ! -f "$AVATAR_DEPS_MARKER" ] && \
           [ -f third_party/SadTalker/requirements.txt ]; then
            python3 -m pip install -r third_party/SadTalker/requirements.txt
            touch "$AVATAR_DEPS_MARKER"
        fi

        # Pipeline ponta a ponta: voz-guia (DSP) -> timbre (Seed-VC) -> [avatar].
        bash scripts/run_all.sh
    '

echo "=== Concluído: $(date) ==="
