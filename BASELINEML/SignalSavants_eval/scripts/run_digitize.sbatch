#!/bin/bash
#SBATCH --job-name=sigsav
#SBATCH --account=jhu159
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
set -e
: "${TAG:?set TAG}"
MODEL="${MODEL:-M1}"
FOLD="${FOLD:-0}"
source "$SLURM_SUBMIT_DIR/paths.env"
module purge
module load gpu
source "$CONDA_SH"
conda activate "$ENV"
export PYTHONUNBUFFERED=1
export NNUNET_FOLD="$FOLD"
export PYTHONPATH="$CODE"
RUN="$WORK/$TAG"
OUT="$RUN/out"
mkdir -p "$OUT" "$RUN/cwd"
# own working dir per run: digitize.py writes data/temp_nnUNet_* relative to cwd,
# so two runs can no longer overwrite each other's masks
cd "$RUN/cwd"
echo "tag=$TAG model=$MODEL fold=$FOLD"
git -C "$REPO" rev-parse HEAD
git -C "$REPO" status --porcelain BASELINEML/SignalSavants/src
nvidia-smi | head -15
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.__version__)"
python -m src.run.digitize -d "$RUN/images" -m "$WEIGHTS/$MODEL" -o "$OUT" -v --allow_failures