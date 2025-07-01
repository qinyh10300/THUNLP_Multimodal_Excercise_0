#!/bin/bash
export PYTHONPATH=$PYTHONPATH:`realpath .`
export CUDA_VISIBLE_DEVICES=0

ground_truth=data/test.json
model_answer=data/mllm_sft_training_test_answer_greedy.jsonl
output_errors=data/mllm_sft_training_test_answer_greedy_errors.jsonl

python ./eval/eval_sft_accuracy.py \
    --ground-truth $ground_truth \
    --model-answer $model_answer \
    --output-errors $output_errors \
