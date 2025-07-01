#!/bin/bash

### ==> TODO: 编写Visual Grounding训练流程脚本

export PYTHONPATH=$PYTHONPATH:`realpath .`
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6000

MODEL="HaoyeZhang/MLLM_Excercise_Model"
DATA="data/train_minicpmv_grounding_9000.json"
EVAL_DATA="data/val_minicpmv_grounding_1244.json"
MODEL_MAX_Length=2048 # if conduct multi-images sft, please set MODEL_MAX_Length=4096


DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
torchrun $DISTRIBUTED_ARGS mllm/finetune.py  \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --eval_data_path $EVAL_DATA \
    --remove_unused_columns false \
    --label_names "labels" \
    --prediction_loss_only false \
    --bf16 true \
    --bf16_full_eval true \
    --fp16 false \
    --fp16_full_eval false \
    --do_train \
    --do_eval \
    --tune_vision true \
    --tune_llm true \
    --model_max_length $MODEL_MAX_Length \
    --max_slice_nums 9 \
    --max_steps 20000 \
    --eval_steps 200 \
    --output_dir output/train_output/mllm_sft_grounding_training_rec \
    --logging_dir output/train_output/mllm_sft_grounding_training_rec/log \
    --logging_strategy "steps" \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --max_grad_norm 1.0 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 10 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --deepspeed mllm/ds_config_zero2.json \
    --save_only_model true \
    --report_to "tensorboard" \
    --task Grounding
### <===