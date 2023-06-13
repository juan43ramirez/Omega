#!/bin/bash

# fix for cn-e002 nodes see: https://mila-umontreal.slack.com/archives/CFAS8455H/p1659965170663569?thread_ts=1659964343.092399&cid=CFAS8455H
module load libffi

# fix for job array see: https://mila-umontreal.slack.com/archives/CFAS8455H/p1652821882612849?thread_ts=1652815432.200809&cid=CFAS8455H
if [[ -n "${SLURM_ARRAY_TASK_COUNT}" ]]; then
  touch "$SLURM_SUBMIT_DIR/.no_report"
fi

# Populate {CHECKPOINT, DATA, ENV, LOG}_DIR
source scripts/env_vars.sh

# Source the virtual environment
module load anaconda/3
conda activate omega

# Start training
python main.py "$@"
