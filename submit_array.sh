#!/bin/bash
#===============================================================================
# submit_array.sh
#
# Submits a SLURM job array covering all (repetition, scenario) combinations.
# Each array task trains all models and computes all attribution methods for
# one repetition x one scenario. Tasks are fully independent and can run in
# parallel up to the --array limit.
#
# Usage:
#   bash submit_array.sh
#
# Adjust the #SBATCH directives for your cluster (partition, memory, time).
#===============================================================================

N_REPS=100
SCENARIOS=("baseline" "high_noise" "high_correlation" "short_history")
N_SCENARIOS=${#SCENARIOS[@]}

# Total tasks = N_REPS x N_SCENARIOS = 100 x 4 = 400
N_TASKS=$(( N_REPS * N_SCENARIOS ))

# Write the (rep, scenario) index table so each task can look up its own pair
INDEX_FILE="task_index.csv"
echo "task_id,rep,scenario" > $INDEX_FILE
TASK=1
for REP in $(seq 1 $N_REPS); do
    for SC in "${SCENARIOS[@]}"; do
        echo "${TASK},${REP},${SC}" >> $INDEX_FILE
        TASK=$(( TASK + 1 ))
    done
done
echo "Index file written: $INDEX_FILE  ($N_TASKS tasks)"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=xai_sim
#SBATCH --array=1-${N_TASKS}%50        # run up to 50 tasks concurrently
#SBATCH --cpus-per-task=4              # adjust to your cluster
#SBATCH --mem=32G                      # adjust; TimeSHAP is memory-hungry
#SBATCH --time=04:00:00               # per-task wall time; adjust if needed
#SBATCH --partition=gpu                # change to your GPU partition name
#SBATCH --gres=gpu:1                   # 1 GPU per task for model training
#SBATCH --output=logs/xai_%A_%a.out
#SBATCH --error=logs/xai_%A_%a.err

mkdir -p logs

# ── Activate your Python environment ──────────────────────────────────────
source /home/hoepler/miniconda3/etc/profile.d/conda.sh
conda activate xai_env

# ── Resolve (rep, scenario) from SLURM_ARRAY_TASK_ID ──────────────────────
TASK_LINE=\$(awk -F',' -v tid="\$SLURM_ARRAY_TASK_ID" 'NR>1 && \$1==tid {print}' task_index.csv)
REP=\$(echo "\$TASK_LINE" | cut -d',' -f2)
SCENARIO=\$(echo "\$TASK_LINE" | cut -d',' -f3)

echo "Task \$SLURM_ARRAY_TASK_ID  ->  rep=\$REP  scenario=\$SCENARIO"
echo "Node: \$(hostname)  GPU: \$CUDA_VISIBLE_DEVICES"

# ── Run the compute script ─────────────────────────────────────────────────
python compute_one_task.py \\
    --rep     "\$REP" \\
    --scenario "\$SCENARIO"

EXIT_CODE=\$?
if [ \$EXIT_CODE -ne 0 ]; then
    echo "FAILED: task \$SLURM_ARRAY_TASK_ID (rep=\$REP, scenario=\$SCENARIO)"
    exit \$EXIT_CODE
fi
echo "Done: task \$SLURM_ARRAY_TASK_ID"
EOF
