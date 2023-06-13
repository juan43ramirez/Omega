export SRC_DIR=$HOME/repos/Omega
export LOG_DIR=$HOME/slurm_logs

export WANDB_PROJECT=omega
export WANDB_DIR=/repos/Omega/wandb

# Automatically set SLURM_TMPDIR when running on interactive sessions
if [ "$SLURM_TMPDIR" = "" ]; then
    export SLURM_TMPDIR=/Tmp/slurm.${SLURM_JOB_ID}.0
fi
