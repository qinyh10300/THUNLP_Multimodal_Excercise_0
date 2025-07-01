#!/bin/bash

# realpath . 会返回当前目录的完整绝对路径
export PYTHONPATH=$PYTHONPATH:`realpath .`
export CUDA_VISIBLE_DEVICES=4,5,6,7
# 不然会保存global_step60/ 文件夹，太大了
# 这是一个训练框架生成的内部子目录，可能包括这一步训练的中间缓存、优化器状态、学习率调度器状态等。
export DEEPSPEED_SAVE_OPTIMIZER_STATES=false
export DEEPSPEED_SAVE_LR_SCHEDULER_STATES=false

GPUS_PER_NODE=4
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

MODEL="HaoyeZhang/MLLM_Excercise_Model"
DATA="data/train.json"
EVAL_DATA="data/test.json"
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
    --max_steps 200 \
    --eval_steps 30 \
    --output_dir output/mllm_sft_training \
    --logging_dir output/mllm_sft_training/log \
    --logging_strategy "steps" \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --save_steps 60 \
    --save_total_limit 3 \
    --learning_rate 5e-6 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --deepspeed mllm/ds_config_zero2.json \
    --save_only_model true \
    --report_to "tensorboard" \
    --task LM
    # 指定任务为语言建模（Language Modeling）
    # --save_total_limit 3  最多保留3个checkpoint