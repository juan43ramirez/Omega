#!/bin/bash

# -----------------------------------------------------------------------------
#                               TO BE CUSTOMIZED
# -----------------------------------------------------------------------------
# Directory containing the source code
source scripts/env_vars.sh


main_bash_script="scripts/job.sh"

# SLURM options
slurm_log_dir="$HOME/slurm_logs"
notify_email="" # Leave empty ("") for no email
partition="long"

NUM_GPUS=1

# Default arguments
seed=0
num_iters=5000
num_samples=100
make_normal_B=True
tag="quadratic_linear_x_comparison"

declare -a _batch_sizes=(1)

# Conditioning of the problem
declare -a _L_B=(1)
declare -a _mu_B=(1) # 0.1)
declare -a _L=(1)
declare -a _mu=(1) # 0.1)

declare -a _x_optim=(SGD)
declare -a _momentum=(0.0)
declare -a _x_lr=(0.02)

# Omega, OMD and SGD on the Linear player
declare -a _y_optim=(OMEGA OMD SGD)
declare -a _y_lr=(0.01)

# -----------------------------------------------------------------------------

# Resources, given number of GPUs requested
if [ "${partition}" = "main" ];
then
  # Set the maximum allowed on main
  mem=10
  cpus=2
  time=00:05:00
else
  mem=$(( $NUM_GPUS * 10 ))
  cpus=2
  time=00:05:00
fi

# The parameter of this function is the python arguments
submit_sbatch () {
    sbatch --job-name=omega-linear_quadratic-slurm-%j.out \
        --time=$time \
        --cpus-per-task $cpus \
        --mem="$mem"G \
        --gres=gpu:$NUM_GPUS \
        --partition=$partition \
        --output=$slurm_log_dir/omega-slurm-%j.out \
        --mail-type=ALL --mail-user=$notify_email \
        $main_bash_script $1
}

export NUM_GPUS

num_iters_arg="--config.train.num_iters=$num_iters"
num_samples_arg="--config.game.num_samples=$num_samples"
dim_args="--config.game.dim=100"
make_normal_B_arg="--config.game.make_normal_B=$make_normal_B"

for batch_sizes in ${_batch_sizes[@]}; do
    batch_size_arg="--config.train.batch_size=$batch_sizes"
    for L_B in ${_L_B[@]}; do
        L_B_args="--config.game.L_B=$L_B"
        for mu_B in ${_mu_B[@]}; do
            mu_B_args="--config.game.mu_B=$mu_B"
            for L in ${_L[@]}; do
                L_args="--config.game.L=$L"
                for mu in ${_mu[@]}; do
                    mu_args="--config.game.mu=$mu"
                    for x_optim in ${_x_optim[@]}; do
                        for x_lr in ${_x_lr[@]}; do
                            for y_optim in ${_y_optim[@]}; do
                                for y_lr in ${_y_lr[@]}; do
                                    config_file_args="--config=configs/main.py:quadratic_linear-sim_gda-$(echo "$x_optim" | tr '[:upper:]' '[:lower:]')-$(echo "$y_optim" | tr '[:upper:]' '[:lower:]') --config.game.make_normal_B=$make_normal_B"
                                    optim_args="--config.train.x.optimizer=$x_optim --config.train.y.optimizer=$y_optim --config.train.x.lr=$x_lr --config.train.y.lr=$y_lr "
                                    if [ "$y_optim" = "OMEGA" ]; then
                                        optim_args="${optim_args} --config.train.y.kwargs.ema_beta=0.9"
                                    fi

                                    if [ "$x_optim" = "OMEGA" ]; then
                                        extra_args="--config.train.x.kwargs.ema_beta=0.9"
                                        python_args="${config_file_args} ${seed_arg} ${num_samples_arg} ${batch_size_arg} ${num_iters_arg} ${dim_args} ${L_B_args} ${mu_B_args} ${L_args} ${mu_args} ${optim_args} ${extra_args} --config.tag=$tag"
                                        submit_sbatch "$python_args"
                                    elif [ "$x_optim" = "OMEGAM" ]; then
                                        extra_args="--config.train.x.kwargs.ema_beta=0.9"
                                        python_args="${config_file_args} ${seed_arg} ${num_samples_arg} ${batch_size_arg} ${num_iters_arg} ${dim_args} ${L_B_args} ${mu_B_args} ${L_args} ${mu_args} ${optim_args} ${extra_args} --config.tag=$tag"
                                        submit_sbatch "$python_args"
                                    elif [ "$x_optim" = "OMD" ]; then
                                        python_args="${config_file_args} ${seed_arg} ${num_samples_arg} ${batch_size_arg} ${num_iters_arg} ${dim_args} ${L_B_args} ${mu_B_args} ${L_args} ${mu_args} ${optim_args} --config.tag=$tag"
                                        submit_sbatch "$python_args"
                                    else
                                        for momentum in ${_momentum[@]}; do
                                            extra_args="--config.train.x.kwargs.momentum=$momentum --config.train.x.kwargs.dampening=$momentum"
                                            python_args="${config_file_args} ${seed_arg} ${num_samples_arg} ${batch_size_arg} ${num_iters_arg} ${dim_args} ${L_B_args} ${mu_B_args} ${L_args} ${mu_args} ${optim_args} ${extra_args} --config.tag=$tag"
                                            submit_sbatch "$python_args"
                                        done
                                    fi
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
