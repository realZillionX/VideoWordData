#!/bin/bash
#SBATCH --job-name=videoworddata
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=120
#SBATCH --time=48:00:00
#SBATCH --partition=cpu

# =============================================================================
# VideoWordData Slurm Job Script
# 
# This script is designed to be used with job arrays for multi-node processing.
# Each array task processes a different slice of the dataset.
#
# Usage:
#   sbatch --array=0-19 slurm_job.sh gsm8k inference
#   sbatch --array=0-19 slurm_job.sh tinystories rendering
# =============================================================================

# Configuration
DATASET=${1:-"gsm8k"}           # Dataset name
TASK_TYPE=${2:-"inference"}     # inference or rendering
BASE_DIR=${3:-"/inspire/hdd/project/embodied-multimodality/public/textcentric"}
SAMPLES_PER_NODE=${4:-10000}    # Number of samples each node processes

# Calculate start index for this node
NODE_ID=${SLURM_ARRAY_TASK_ID:-0}
START_IDX=$((NODE_ID * SAMPLES_PER_NODE))

# Number of workers (use all available CPUs)
NUM_WORKERS=${SLURM_CPUS_PER_TASK:-120}

# Project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Create logs directory
mkdir -p logs

# Activate conda environment if needed
# source /path/to/conda/bin/activate your_env

echo "=============================================="
echo "VideoWordData Processing Job"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "CPUs: ${NUM_WORKERS}"
echo "----------------------------------------------"
echo "Dataset: ${DATASET}"
echo "Task Type: ${TASK_TYPE}"
echo "Start Index: ${START_IDX}"
echo "Samples: ${SAMPLES_PER_NODE}"
echo "Base Dir: ${BASE_DIR}"
echo "=============================================="

# Construct the script path
SCRIPT_PATH="${PROJECT_DIR}/${TASK_TYPE}/${DATASET}.py"

# Check if script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Script not found: $SCRIPT_PATH"
    echo "Available scripts in ${TASK_TYPE}/:"
    ls -la "${PROJECT_DIR}/${TASK_TYPE}/"
    exit 1
fi

# Run the Python script
echo "Running: python $SCRIPT_PATH"
echo "  --base_dir $BASE_DIR"
echo "  --start_idx $START_IDX"
echo "  --num_samples $SAMPLES_PER_NODE"
echo "  --num_workers $NUM_WORKERS"
echo ""

python "$SCRIPT_PATH" \
    --base_dir "$BASE_DIR" \
    --start_idx "$START_IDX" \
    --num_samples "$SAMPLES_PER_NODE" \
    --num_workers "$NUM_WORKERS"

exit_code=$?

echo ""
echo "=============================================="
echo "Job completed with exit code: $exit_code"
echo "=============================================="

exit $exit_code
