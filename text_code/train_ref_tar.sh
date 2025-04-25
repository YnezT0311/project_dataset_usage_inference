#!/bin/bash

# Set CUDA_VISIBLE_DEVICES environment variable
export CUDA_VISIBLE_DEVICES=1

mkdir -p logs/
mkdir -p logs/tar_models
############################################################################################################
# Prepare evaluation models

for expid in {1..2}
do
nohup python -u ${PWD}/train_gpt2_ref.py --id $expid --epochs 3 --max_length 256 --ref_dir ./ &> logs/log_ref"$expid"
done

 for expid in {1..4}
do
    nohup python -u ${PWD}/train_gpt2_tar.py --id $expid --epochs 3 --max_length 256 --sampling_type sequential --tar_dir ./ &> logs/tar_models/log_tar"$expid"
done

nohup python -u ${PWD}/compute_gpt2.py --save_dir ./ --sampling_type sequential --max_length 256 &> logs/log_compute_scores

