#!/bin/bash
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --ntasks=1
#SBATCH --time=UNLIMITED
#SBATCH --nodelist=dgx-H100-02
#SBATCH --job-name=vsinger-style
#SBATCH --output=logs/style/style_%j.out
#SBATCH --error=logs/style/style_%j.err

# ============================================================================
# Pipeline de ESTILO: nova melodia + estilo de canto (TCSinger) + timbre (Seed-VC)
# + gênero/arranjo (MusicGen). EXIGE Python 3.10 com COMPILADOR (imagem -devel).
#   singularity pull /raid/user_lucasalves/images/pytorch_2.2_cu121_py310_devel.sif \
#       docker://pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel
#
# Tudo sobre material legítimo: melodia/letra suas, referência de estilo = gravação de
# vocês, timbre = voz consentida, gênero = prompt de texto.
# ============================================================================

CONTAINER_IMAGE="${YING_CONTAINER_IMAGE:-/raid/user_lucasalves/images/pytorch_2.2_cu121_py310_devel.sif}"
HOST_RAID="${HOST_RAID:-/raid/user_lucasalves}"
PROJECT_SUBDIR="${PROJECT_SUBDIR:-ufg/pav/virtual-singer}"

mkdir -p logs/style
echo "=== Style job $SLURM_JOB_ID em $(hostname) | $(date) ==="

ENV_FILE="${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}/slurm/.env"
[ -f "$ENV_FILE" ] || ENV_FILE="$(dirname "$0")/.env"
[ -f "$ENV_FILE" ] && { echo "Carregando $ENV_FILE"; set -a; source "$ENV_FILE"; set +a; }

srun singularity exec \
    --nv --no-home --bind "${HOST_RAID}:/work" \
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
        command -v gcc >/dev/null 2>&1 && echo "[env] gcc OK" || echo "[env] AVISO: sem gcc -> use imagem -devel"

        # ---- PARÂMETROS (slurm/.env) ----
        export STYLE_TARGET="${STYLE_TARGET:-data/songs/demo}"   # partitura alvo (melodia+letra)
        export STYLE_VOICE="${STYLE_VOICE:-singers/lucas}"       # voz de timbre (clip de vocês)
        export STYLE_REF_DIR="${STYLE_REF_DIR:-}"                # partitura da ref. de estilo
        export STYLE_WAV="${STYLE_WAV:-}"                        # gravação de vocês (estilo)
        export STYLE_GENRE="${STYLE_GENRE:-acoustic pop, soft guitar}"
        export STYLE_FLAGS="${STYLE_FLAGS:-}"                    # ex.: "--no-genre" p/ isolar
        # ----------------------------------

        # Deps (marker próprio do ambiente de estilo py3.10).
        M="/work/.home/.style310_deps_ok_v1"
        [ -n "${REINSTALL:-}" ] && rm -f "$M"
        if [ ! -f "$M" ]; then
            python3 -m pip install -U pip setuptools wheel
            python3 -m pip install --prefer-binary -r requirements.txt
        fi
        # Clona Seed-VC (timbre) + TCSinger (estilo) e baixa checkpoints.
        python3 scripts/setup_models.py --with-tcsinger
        if [ ! -f "$M" ]; then
            [ -f third_party/seed-vc/requirements.txt ] && \
                python3 -m pip install --prefer-binary -r third_party/seed-vc/requirements.txt
            [ -f third_party/TCSinger/requirements.txt ] && \
                python3 -m pip install --prefer-binary -r third_party/TCSinger/requirements.txt
            python3 -m pip install --prefer-binary audiocraft   # MusicGen (gênero)
            touch "$M"
        fi
        python3 - <<PY
import nltk
for p in ["averaged_perceptron_tagger_eng","cmudict","punkt","punkt_tab"]: nltk.download(p, quiet=True)
PY

        # ffmpeg no PATH + patches (TCSinger lê nomes via env; Demucs in-place fix).
        FF="$(python3 -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" 2>/dev/null || true)"
        [ -n "$FF" ] && [ -x "$FF" ] && ln -sf "$FF" /work/.home/.local/bin/ffmpeg
        python3 scripts/patch_tcsinger_env.py
        python3 scripts/patch_demucs.py || true

        STYLE_ARGS=""
        [ -n "$STYLE_REF_DIR" ] && STYLE_ARGS="$STYLE_ARGS --style-ref $STYLE_REF_DIR"
        [ -n "$STYLE_WAV" ]     && STYLE_ARGS="$STYLE_ARGS --style-wav $STYLE_WAV"
        [ -n "$STYLE_GENRE" ]   && STYLE_ARGS="$STYLE_ARGS --genre \"$STYLE_GENRE\""

        echo "[style] alvo=$STYLE_TARGET voz=$STYLE_VOICE genero=$STYLE_GENRE"
        eval python3 -m src.pipeline_style \
            --target "$STYLE_TARGET" \
            --voice "$STYLE_VOICE" \
            --out "out/style_$(basename $STYLE_TARGET).wav" \
            $STYLE_ARGS $STYLE_FLAGS
    '

echo "=== Concluído: $(date) ==="
