#!/bin/bash
export PYTHONPATH=$PYTHONPATH:`realpath .`
export CUDA_VISIBLE_DEVICES=0

model_name_or_path="output/mllm_sft_training/checkpoint-60"
data_path=data/test.json
save_path=data/mllm_sft_training_test_answer_greedy.jsonl

python ./eval/sft_model_eval.py \
    --model-name-or-path $model_name_or_path \
    --question-file $data_path \
    --answers-file $save_path \
    --num-beams 4
    # --sampling
