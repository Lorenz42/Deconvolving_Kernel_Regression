#!/bin/bash
# Submit one SLURM array task per (function, sample size) combination:
#
#   sbatch scripts/start_experiment.sh
#
# Each task runs plot_comparison_square_vs_rectangular_support.py for a single
# combination, writing its figure to figures/ and its raw data to the scratch
# data folder. All combinations run in parallel as an array.

#SBATCH --job-name=deconv_experiment
#SBATCH --output=/scratch/gpfs/GILLES/lh9809/my_experiments/deconv_%A_%a.out
#SBATCH --error=/scratch/gpfs/GILLES/lh9809/my_experiments/deconv_%A_%a.err
#SBATCH --array=0-7                 # MUST equal (#FUNCTIONS * #SAMPLE_SIZES) - 1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --mail-type=begin,end,fail  # receive email notifications
#SBATCH --mail-user=lh9809@princeton.edu

set -euo pipefail

# --- Parameter grid -------------------------------------------------------
# IMPORTANT: keep the --array range above in sync with this grid.
# total = (#FUNCTIONS * #SAMPLE_SIZES); --array must be 0..total-1.
FUNCTIONS=(product_of_cos_function absolute_value_function)
SAMPLE_SIZES=(500 1000 5000 10000)

n_sizes=${#SAMPLE_SIZES[@]}
total=$(( ${#FUNCTIONS[@]} * n_sizes ))

task_id=${SLURM_ARRAY_TASK_ID}
if (( task_id >= total )); then
    echo "task id ${task_id} out of range; set --array=0-$(( total - 1 ))" >&2
    exit 1
fi

# Map the flat array index to a (function, sample size) pair.
FUNCTION=${FUNCTIONS[$(( task_id / n_sizes ))]}
N_SAMPLES=${SAMPLE_SIZES[$(( task_id % n_sizes ))]}

echo "Array task ${task_id}/${total}: function=${FUNCTION}, n_samples=${N_SAMPLES}"

# --- Environment ----------------------------------------------------------
export MPLBACKEND=Agg               # headless matplotlib backend

# matplotlib usetex calls latex; some compute nodes have a stale/minimal TeX
# index that fails to surface standard packages (e.g. type1cm). Search the home
# TeX tree first, then disk-scan the full system tree to bypass any stale ls-R.
export TEXINPUTS=${HOME}/texmf//:/usr/share/texlive/texmf-dist//:

# Keep BLAS/OpenMP from oversubscribing the allocated cores.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

PROJECT_ROOT=/home/lh9809/deconvolving_kernel_package
# Make the package importable (or `pip install -e .` once and drop this line).
export PYTHONPATH=${PROJECT_ROOT}/src:${PYTHONPATH:-}

# --- Run ------------------------------------------------------------------
/usr/bin/python "${PROJECT_ROOT}/scripts/plot_comparison_square_vs_rectangular_support.py" \
    --function "${FUNCTION}" \
    --n_samples "${N_SAMPLES}"
