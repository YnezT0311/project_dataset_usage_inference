#!/bin/bash

# Set CUDA_VISIBLE_DEVICES environment variable
export CUDA_VISIBLE_DEVICES=1

mkdir -p logs/

# Run duci
python evaluation_gpt2.py --save_dir ./ --dataset BookMIA-in-sentences-25 --sampling_type sequential &> logs/duci.log

