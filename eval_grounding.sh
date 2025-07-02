#!/bin/bash
export PYTHONPATH=$PYTHONPATH:`realpath .`
export CUDA_VISIBLE_DEVICES=4

model_name_or_path="output/train_output/mllm_sft_grounding_training_rec/checkpoint-5000"
# question_file=data/grouding_data/REC_refcoco_unc_val.jsonl
# question_file=data/grouding_data/REC_refcoco_unc_testA.jsonl
# question_file=data/grouding_data/REC_refcoco_unc_testB.jsonl
# question_file=data/grouding_data/REC_refcoco+_unc_testA.jsonl
question_file=data/grouding_data/REC_refcoco+_unc_testB.jsonl
# question_file=data/grouding_data/REC_refcoco+_unc_val.jsonl
# question_file=data/grouding_data/REC_refcocog_umd_test.jsonl
# question_file=data/grouding_data/REC_refcocog_umd_val.jsonl
image_dir=data/images/train2014
save_path=eval_output/grounding_eval_REC_refcoco+_unc_testB
# output_error_path=eval_output/sft_ckpt_90_error_cases_fixed.json

python ./eval/grounding_eval.py \
    --model-name-or-path $model_name_or_path \
    --question-file $question_file \
    --image-dir $image_dir \
    --save-dir $save_path \
    --sampling \
    --vis-nums 6
