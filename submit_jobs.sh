#!/bin/bash
# =============================================================================
# VideoWordData Multi-Node Submission Script
#
# Submits Slurm job arrays to process datasets across multiple nodes.
# Each node processes a different slice of the dataset in parallel.
#
# Usage:
#   ./submit_jobs.sh                    # Submit all datasets, all tasks
#   ./submit_jobs.sh gsm8k              # Submit specific dataset
#   ./submit_jobs.sh gsm8k inference    # Submit specific dataset and task
#   ./submit_jobs.sh --dry-run          # Show commands without executing
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================

# Number of nodes to use
NUM_NODES=${NUM_NODES:-20}

# CPUs per node
CPUS_PER_NODE=${CPUS_PER_NODE:-120}

# Samples per node (adjust based on dataset size)
SAMPLES_PER_NODE=${SAMPLES_PER_NODE:-10000}

# Base directory for output
BASE_DIR=${BASE_DIR:-"/inspire/hdd/project/embodied-multimodality/public/textcentric"}

# Slurm partition
PARTITION=${PARTITION:-"cpu"}

# Time limit (HH:MM:SS)
TIME_LIMIT=${TIME_LIMIT:-"48:00:00"}

# Project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =============================================================================
# Dataset Configurations (samples per dataset for optimal node distribution)
# =============================================================================

declare -A DATASET_SAMPLES
DATASET_SAMPLES["gsm8k"]=7500
DATASET_SAMPLES["openmath2_gsm8k"]=200000
DATASET_SAMPLES["belle_school_math"]=250000
DATASET_SAMPLES["gsm8k_chinese"]=8800
DATASET_SAMPLES["tinystories"]=2000000

# All available datasets
ALL_DATASETS=("gsm8k" "openmath2_gsm8k" "belle_school_math" "gsm8k_chinese" "tinystories")

# All task types
ALL_TASKS=("inference" "rendering")

# =============================================================================
# Helper Functions
# =============================================================================

print_usage() {
    echo "Usage: $0 [OPTIONS] [DATASET] [TASK_TYPE]"
    echo ""
    echo "Arguments:"
    echo "  DATASET      Dataset to process (gsm8k, openmath2_gsm8k, belle_school_math,"
    echo "               gsm8k_chinese, tinystories, or 'all')"
    echo "  TASK_TYPE    Task type (inference, rendering, or 'all')"
    echo ""
    echo "Options:"
    echo "  --dry-run    Show commands without executing"
    echo "  --status     Show job status"
    echo "  --cancel     Cancel all VideoWordData jobs"
    echo "  -h, --help   Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  NUM_NODES         Number of nodes (default: 20)"
    echo "  CPUS_PER_NODE     CPUs per node (default: 120)"
    echo "  SAMPLES_PER_NODE  Samples per node (default: 10000)"
    echo "  BASE_DIR          Output base directory"
    echo "  PARTITION         Slurm partition (default: cpu)"
    echo "  TIME_LIMIT        Time limit (default: 48:00:00)"
    echo ""
    echo "Examples:"
    echo "  $0                           # Submit all datasets and tasks"
    echo "  $0 gsm8k                     # Submit gsm8k (both inference and rendering)"
    echo "  $0 gsm8k inference           # Submit gsm8k inference only"
    echo "  $0 all rendering             # Submit all datasets, rendering only"
    echo "  $0 --dry-run tinystories     # Show commands for tinystories"
    echo ""
}

calculate_nodes_needed() {
    local dataset=$1
    local total_samples=${DATASET_SAMPLES[$dataset]:-100000}
    local nodes_needed=$(( (total_samples + SAMPLES_PER_NODE - 1) / SAMPLES_PER_NODE ))
    
    # Cap at NUM_NODES
    if [ $nodes_needed -gt $NUM_NODES ]; then
        nodes_needed=$NUM_NODES
    fi
    
    echo $nodes_needed
}

submit_job() {
    local dataset=$1
    local task_type=$2
    local dry_run=$3
    
    local nodes_needed=$(calculate_nodes_needed "$dataset")
    local array_end=$((nodes_needed - 1))
    
    echo "----------------------------------------------"
    echo "Dataset: $dataset"
    echo "Task: $task_type"
    echo "Nodes: $nodes_needed (array 0-$array_end)"
    echo "Samples/Node: $SAMPLES_PER_NODE"
    echo "Total Coverage: $((nodes_needed * SAMPLES_PER_NODE)) samples"
    echo "----------------------------------------------"
    
    local cmd="sbatch \
        --job-name=vwd_${dataset}_${task_type} \
        --array=0-${array_end} \
        --partition=${PARTITION} \
        --cpus-per-task=${CPUS_PER_NODE} \
        --time=${TIME_LIMIT} \
        ${PROJECT_DIR}/slurm_job.sh ${dataset} ${task_type} ${BASE_DIR} ${SAMPLES_PER_NODE}"
    
    echo "Command: $cmd"
    echo ""
    
    if [ "$dry_run" != "true" ]; then
        eval "$cmd"
    else
        echo "(Dry run - not actually submitting)"
    fi
    echo ""
}

show_status() {
    echo "Current VideoWordData Jobs:"
    echo "========================================"
    squeue -u "$USER" -n "vwd_*" -o "%.18i %.20j %.8T %.10M %.9l %.6D %R" 2>/dev/null || \
    squeue -u "$USER" | grep -E "(JOBID|vwd_)" || \
    echo "No jobs found or squeue not available"
}

cancel_jobs() {
    echo "Cancelling all VideoWordData jobs..."
    scancel -u "$USER" -n "vwd_*" 2>/dev/null || \
    scancel -u "$USER" --name="vwd_*" 2>/dev/null || \
    echo "No jobs to cancel or scancel not available"
}

# =============================================================================
# Main Logic
# =============================================================================

# Parse arguments
DRY_RUN=false
DATASET_ARG=""
TASK_ARG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --status)
            show_status
            exit 0
            ;;
        --cancel)
            cancel_jobs
            exit 0
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            if [ -z "$DATASET_ARG" ]; then
                DATASET_ARG=$1
            elif [ -z "$TASK_ARG" ]; then
                TASK_ARG=$1
            fi
            shift
            ;;
    esac
done

# Default to all
DATASET_ARG=${DATASET_ARG:-"all"}
TASK_ARG=${TASK_ARG:-"all"}

# Create logs directory
mkdir -p "${PROJECT_DIR}/logs"

# Header
echo "=============================================="
echo "VideoWordData Multi-Node Job Submission"
echo "=============================================="
echo "Project Dir: $PROJECT_DIR"
echo "Output Dir: $BASE_DIR"
echo "Nodes: $NUM_NODES"
echo "CPUs/Node: $CPUS_PER_NODE"
echo "Partition: $PARTITION"
echo "Time Limit: $TIME_LIMIT"
echo "Dry Run: $DRY_RUN"
echo "=============================================="
echo ""

# Determine datasets to process
if [ "$DATASET_ARG" == "all" ]; then
    DATASETS=("${ALL_DATASETS[@]}")
else
    DATASETS=("$DATASET_ARG")
fi

# Determine tasks to process
if [ "$TASK_ARG" == "all" ]; then
    TASKS=("${ALL_TASKS[@]}")
else
    TASKS=("$TASK_ARG")
fi

# Submit jobs
for dataset in "${DATASETS[@]}"; do
    for task in "${TASKS[@]}"; do
        submit_job "$dataset" "$task" "$DRY_RUN"
    done
done

echo "=============================================="
if [ "$DRY_RUN" == "true" ]; then
    echo "Dry run complete. No jobs were submitted."
else
    echo "All jobs submitted!"
    echo "Use 'squeue -u $USER' to monitor progress"
    echo "Use '$0 --status' to show job status"
    echo "Use '$0 --cancel' to cancel all jobs"
fi
echo "=============================================="
