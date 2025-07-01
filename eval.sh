#!/bin/bash
export PYTHONPATH=$PYTHONPATH:`realpath .`
export CUDA_VISIBLE_DEVICES=0

model_name_or_path="output/mllm_preference_training"
data_path=data/objhal_bench.jsonl
save_path=objhal_benc_mllm_preference_training_answer_greedy.jsonl

python ./eval/model_eval.py \
--model-name-or-path $model_name_or_path \
--question-file $data_path \
--answers-file $save_path \
# --sampling
