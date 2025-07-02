MODEL_NAME="objhal_bench_answer.jsonl"

echo "========================="
echo "MODEL: $MODEL_NAME"
echo "========================="

python eval/chair.py \
    --coco_path data/annotations \
    --cache data/chair_300.pkl \
    --cap_file objhal_benc_mllm_preference_training_answer_greedy.jsonl \
    --save_path eval-chair-300_answer.json \
    --caption_key answer

    # --cap_file $MODEL_NAME/objhal_bench_answer.jsonl \
    # --save_path $MODEL_NAME/eval-chair-300_answer.json \
