#!/bin/bash

# Assuming you've set up HfArgumentParser to parse command line arguments in train.py

python train.py \
  --model_path "allenai/longformer-base-4096" \
  --train_csv "../data/train/train_chatgpt.csv" \
  --load_best_model_at_end True \
  --dev_csv "../data/dev/dev.csv" \
  --test_csvs ../data/dev/dev.csv \
  --metric_for_best_model "eval_mean_absolute_diff" \
  --greater_is_better False \
  --do_train True \
  --do_predict True \
  --seed 55 \
  --output_dir "./runs/exp_5" \
  --logging_dir "./runs/exp_5/logs" \
  --num_train_epochs 10 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --auto_find_batch_size True \
  --logging_steps 10 \
  --load_best_model_at_end True \
  --evaluation_strategy "epoch" \
  --save_strategy "epoch" \
  --save_total_limit 2
