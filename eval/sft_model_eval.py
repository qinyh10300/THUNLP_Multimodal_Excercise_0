from eval.model_eval import MLLMEvalModel
import torch
from transformers import AutoTokenizer
from mllm.model.image_processing import ModelImageProcessor
from mllm.model.processing import ModelProcessor
from utils.file_io import read_json
import json
import argparse
from PIL import Image
import io
import base64
from tqdm import tqdm

def eval_finetuned_model(args):
    model = MLLMEvalModel.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )

    img_processor_config = read_json('mllm/model/mllm_preprocessor_config.json')
    image_processor = ModelImageProcessor(**img_processor_config)
    processor = ModelProcessor(image_processor, tokenizer)

    model.eval().cuda()

    # 读取测试数据
    with open(args.question_file, 'r') as f:
        input_data = json.load(f)
    
    ans_file = open(args.answers_file, 'w')

    with torch.inference_mode():
        for item in tqdm(input_data):
            # 获取图片路径和用户问题
            image_path = item['image']
            # 获取用户的问题（conversations中的第一个user消息）
            user_question = None
            for conv in item['conversations']:
                if conv['role'] == 'user':
                    user_question = conv['content']
                    break
            
            if user_question is None:
                continue
                
            # 移除用户问题中的<image>标记，因为model.chat会自动处理
            user_question_clean = user_question.replace('<image>\n', '').replace('<image>', '')
            
            msgs = [{"role": "user", "content": user_question_clean}]

            # 加载图片
            try:
                # 如果是base64编码的图片
                if len(image_path) > 1000:
                    image = Image.open(io.BytesIO(base64.b64decode(image_path))).convert('RGB')
                else:
                    # 如果是文件路径
                    image = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue

            # 使用model.chat进行推理
            answer = model.chat(
                image=image,
                msgs=msgs,
                tokenizer=tokenizer,
                sampling=args.sampling,
                processor=processor,
                min_new_tokens=args.min_new_tokens,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams if not args.sampling else None,
                repetition_penalty=1.2,
            )

            # 准备答案字典
            answer_dict = {
                "id": item.get('id', 0),
                "question": user_question_clean,
                "answer": answer if answer else "",
                # "input_tokens": input_tokens,
                "model": args.model_name_or_path,
                "image_path": image_path
            }

            # 写入答案文件
            ans_file.write(json.dumps(answer_dict, ensure_ascii=False) + '\n')
            ans_file.flush()

    ans_file.close()
    print(f"评估完成，结果已保存到 {args.answers_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True, help="模型路径")
    parser.add_argument("--question-file", type=str, required=True, help="测试数据文件路径")
    parser.add_argument("--answers-file", type=str, required=True, help="答案输出文件路径")
    parser.add_argument("--sampling", action='store_true', help="是否使用采样生成")
    parser.add_argument("--num-beams", type=int, default=3, help="beam search的beam数")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="最大生成token数")
    parser.add_argument("--min-new-tokens", type=int, default=0, help="最小生成token数")
    args = parser.parse_args()

    eval_finetuned_model(args)