#!/bin/bash
#SBATCH --job-name=__JOBNAME__
#SBATCH --output=__JOBNAME__.out
#SBATCH --error=__JOBNAME__.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=__NPROC__
#SBATCH --mem=__MEM__
#SBATCH --time=__TIME__
#SBATCH --partition=__PARTITION__
# Uncomment if your cluster requires an account:
##SBATCH --account=__ACCOUNT__

# ── Environment ────────────────────────────────────────────────────────────
# Adjust module names to match your HPC environment.
module purge
module load gaussian/16          # or: gaussian/g16.C.01, etc.

export GAUSS_SCRDIR=${TMPDIR:-/tmp}/$SLURM_JOB_ID
mkdir -p "$GAUSS_SCRDIR"

# ── Run ────────────────────────────────────────────────────────────────────
INPUT=__INPUT__
OUTPUT=${INPUT%.com}.log

echo "Job      : $SLURM_JOB_ID"
echo "Node     : $SLURM_NODELIST"
echo "Input    : $INPUT"
echo "Scratch  : $GAUSS_SCRDIR"
echo "Start    : $(date)"

g16 < "$INPUT" > "$OUTPUT"

echo "End      : $(date)"
echo "Exit code: $?"

# ── Cleanup ────────────────────────────────────────────────────────────────
# Copy checkpoint file back if present
CHK=${INPUT%.com}.chk
if [[ -f "$GAUSS_SCRDIR/$CHK" ]]; then
    cp "$GAUSS_SCRDIR/$CHK" .
fi
rm -rf "$GAUSS_SCRDIR"
