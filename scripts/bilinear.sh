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
declare -a _seeds=(0)
declare -a _num_samples=(100)
declare -a _batch_sizes=(1)
declare -a _num_iters=(5000)
declare -a _dims=(100)
declare -a _L_B=(1)
declare -a _mu_B=(1)
declare -a _bias=(True)
declare -a _optim=(OMEGA)
declare -a _ema_beta=(0.9)
declare -a _momentum=(0.0)
declare -a _lr=(0.02)
use_lr_scheduler=False
make_normal_B=True

# # All optimizers
# declare -a _optim=(OMEGA OMEGAM SGD OMD)
# declare -a _mu_B=(1 0.1 0.01)
# tag="bilinear_omega"

# # BS sweep
# declare -a _batch_sizes=(1 5 10 20 100)
# declare -a _optim=(OMEGA OMEGAM SGD OMD)
# declare -a _mu_B=(1 0.1)
# tag="bilinear_batch_size"

# # EMA Sweep
# declare -a _optim=(OMEGA)
# declare -a _ema_beta=(0.99 0.9 0.7 0.5 0.3 0.1 0.0)
# declare -a _mu_B=(1 0.1)
# tag="bilinear_ema"

# -----------------------------------------------------------------------------

# Resources, given number of GPUs requested
if [ "${partition}" = "main" ];
then
  # Set the maximum allowed on mainSGD
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
    sbatch --job-name=omega-bilinear-sim_gda-omd-slurm-%j.out \
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


for seed in ${_seeds[@]}; do
    seed_arg="--config.game.data_seed=$seed --config.game.sample_seed=$seed"
    for num_samples in ${_num_samples[@]}; do
        num_samples_arg="--config.game.num_samples=$num_samples"
        for batch_sizes in ${_batch_sizes[@]}; do
            batch_size_arg="--config.train.batch_size=$batch_sizes"
            for num_iters in ${_num_iters[@]}; do
                num_iters_arg="--config.train.num_iters=$num_iters"
                for dim in ${_dims[@]}; do
                    dim_args="--config.game.dim=$dim"
                    for L_B in ${_L_B[@]}; do
                        L_B_args="--config.game.L_B=$L_B"
                        for mu_B in ${_mu_B[@]}; do
                            mu_B_args="--config.game.mu_B=$mu_B"
                            for lr in ${_lr[@]}; do
                                lr_args="--config.train.x.lr=$lr --config.train.y.lr=$lr"
                                for bias in ${_bias[@]}; do
                                    bias_args="--config.game.bias=$bias"
                                    for optim in ${_optim[@]}; do
                                        config_file_args="--config=configs/main.py:bilinear-sim_gda-$(echo "$optim" | tr '[:upper:]' '[:lower:]')-$(echo "$optim" | tr '[:upper:]' '[:lower:]')"
                                        optim_args="--config.train.x.optimizer=$optim --config.train.y.optimizer=$optim --config.train.use_lr_scheduler=$use_lr_scheduler --config.game.make_normal_B=$make_normal_B --config.tag=$tag"

                                        if [ "$optim" = "OMEGA" ]; then
                                            for ema_beta in ${_ema_beta[@]}; do
                                                ema_args="--config.train.x.kwargs.ema_beta=$ema_beta --config.train.y.kwargs.ema_beta=$ema_beta"
                                                python_args="${config_file_args} ${seed_arg} ${num_samples_arg} ${batch_size_arg} ${num_iters_arg} ${dim_args} ${L_B_args} ${mu_B_args} ${lr_args} ${bias_args} ${optim_args} ${ema_args}"
                                                submit_sbatch "$python_args"
                                            done
                                        elif [ "$optim" = "SGD" ]; then
                                            for momentum in ${_momentum[@]}; do
                                                momentum_args="--config.train.x.kwargs.momentum=$momentum --config.train.y.kwargs.momentum=$momentum --config.train.x.kwargs.dampening=$momentum --config.train.y.kwargs.dampening=$momentum"
                                                python_args="${config_file_args} ${seed_arg} ${num_samples_arg} ${batch_size_arg} ${num_iters_arg} ${dim_args} ${L_B_args} ${mu_B_args} ${lr_args} ${bias_args} ${optim_args} ${momentum_args}"
                                                submit_sbatch "$python_args"
                                            done
                                        elif [ "$optim" = "OMEGAM" ]; then
                                            # Hardcode ema_beta to 0.9
                                            momentum_args="--config.train.x.kwargs.ema_beta=0.9 --config.train.y.kwargs.ema_beta=0.9"
                                            python_args="${config_file_args} ${seed_arg} ${num_samples_arg} ${batch_size_arg} ${num_iters_arg} ${dim_args} ${L_B_args} ${mu_B_args} ${lr_args} ${bias_args} ${optim_args} ${momentum_args}"
                                            submit_sbatch "$python_args"
                                        else
                                            # OMD
                                            python_args="${config_file_args} ${seed_arg} ${num_samples_arg} ${batch_size_arg} ${num_iters_arg} ${dim_args} ${L_B_args} ${mu_B_args} ${lr_args} ${bias_args} ${optim_args}"
                                            submit_sbatch "$python_args"
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
done
