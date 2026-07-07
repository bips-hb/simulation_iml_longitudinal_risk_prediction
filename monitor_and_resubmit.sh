#!/bin/bash
#===============================================================================
# monitor_and_resubmit.sh
#
# Two utilities:
#
#   bash monitor_and_resubmit.sh status
#     -- counts completed / missing / failed tasks by checking for manifest.json
#
#   bash monitor_and_resubmit.sh resubmit
#     -- builds a reduced task index of only the missing/failed tasks and
#        submits a new SLURM job array covering just those
#
# Run from the same directory as submit_array.sh and task_index.csv.
#===============================================================================

RESULT_ROOT="/home/hoepler/eICU_Real_World_Example/simulation_results_repeated"
SCENARIOS=("baseline" "high_noise" "high_correlation" "short_history")
N_REPS=100

# ── Status check ─────────────────────────────────────────────────────────────
check_status() {
    DONE=0
    MISSING=0
    MISSING_LIST=""

    for REP in $(seq 1 $N_REPS); do
        REP_TAG=$(printf "rep_%03d" $REP)
        for SC in "${SCENARIOS[@]}"; do
            MANIFEST="${RESULT_ROOT}/${REP_TAG}/${SC}/manifest.json"
            if [ -f "$MANIFEST" ]; then
                DONE=$(( DONE + 1 ))
            else
                MISSING=$(( MISSING + 1 ))
                MISSING_LIST="${MISSING_LIST} ${REP_TAG}/${SC}"
            fi
        done
    done

    TOTAL=$(( N_REPS * ${#SCENARIOS[@]} ))
    echo "Progress: ${DONE} / ${TOTAL} tasks complete"
    echo "Missing : ${MISSING}"

    if [ $MISSING -gt 0 ] && [ "$2" == "--verbose" ]; then
        echo ""
        echo "Missing tasks:"
        for t in $MISSING_LIST; do echo "  $t"; done
    fi
}

# ── Resubmit missing tasks ────────────────────────────────────────────────────
resubmit_missing() {
    RETRY_INDEX="task_index_retry.csv"
    echo "task_id,rep,scenario" > $RETRY_INDEX
    TASK=1

    for REP in $(seq 1 $N_REPS); do
        REP_TAG=$(printf "rep_%03d" $REP)
        for SC in "${SCENARIOS[@]}"; do
            MANIFEST="${RESULT_ROOT}/${REP_TAG}/${SC}/manifest.json"
            if [ ! -f "$MANIFEST" ]; then
                echo "${TASK},${REP},${SC}" >> $RETRY_INDEX
                TASK=$(( TASK + 1 ))
            fi
        done
    done

    N_RETRY=$(( TASK - 1 ))
    if [ $N_RETRY -eq 0 ]; then
        echo "No missing tasks found. Nothing to resubmit."
        return
    fi

    echo "Resubmitting ${N_RETRY} missing tasks..."

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=xai_retry
#SBATCH --array=1-${N_RETRY}%20
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/retry_%A_%a.out
#SBATCH --error=logs/retry_%A_%a.err

source /home/hoepler/miniconda3/etc/profile.d/conda.sh
conda activate xai_env

TASK_LINE=\$(awk -F',' -v tid="\$SLURM_ARRAY_TASK_ID" 'NR>1 && \$1==tid {print}' ${RETRY_INDEX})
REP=\$(echo "\$TASK_LINE" | cut -d',' -f2)
SCENARIO=\$(echo "\$TASK_LINE" | cut -d',' -f3)

echo "Retry task \$SLURM_ARRAY_TASK_ID  ->  rep=\$REP  scenario=\$SCENARIO"

python compute_one_task.py --rep "\$REP" --scenario "\$SCENARIO"
EOF
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
case "$1" in
    status)    check_status "$@" ;;
    resubmit)  resubmit_missing ;;
    *)
        echo "Usage: bash monitor_and_resubmit.sh [status|resubmit] [--verbose]"
        exit 1
        ;;
esac
