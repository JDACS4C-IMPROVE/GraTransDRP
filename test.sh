#!/bin/bash --login

# Runs training for 2 epochs and dums predictions and scores into ap_res
python training.py --model 0 --train_batch 1024 --val_batch 1024 --test_batch 1024 --lr 0.0001 --num_epoch 2 --log_interval 20 --cuda_name "cuda:0"
