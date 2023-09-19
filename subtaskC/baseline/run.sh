#!/bin/bash
exp_name="exp_1"
seed_value=42
python transformer_baseline.py \
  --model_path "allenai/longformer-base-4096" \
  --train_file "../data/subtaskC_train.jsonl" \
  --load_best_model_at_end True \
  --dev_file "../data/subtaskC_dev.jsonl" \
  --test_files ../data/subtaskC_dev.jsonl \
  --metric_for_best_model "eval_mean_absolute_diff" \
  --greater_is_better False \
  --do_train True \
  --do_predict True \
  --seed $seed_value \
  --output_dir "./runs/$exp_name" \
  --logging_dir "./runs/$exp_name/logs" \
  --num_train_epochs 10 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --auto_find_batch_size True \
  --logging_steps 10 \
  --load_best_model_at_end True \
  --evaluation_strategy "epoch" \
  --save_strategy "epoch" \
  --save_total_limit 2
