import re
import io
import os
import json
import base64
import argparse
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import torch
from transformers import AutoTokenizer

from mllm.model import MLLMModel
from mllm.model.processing import ModelProcessor
from mllm.model.image_processing import ModelImageProcessor
from utils.file_io import read_jsonlines, read_json

from torchvision.ops.boxes import box_area

from eval.model_eval import MLLMEvalModel

def vis_boxes(img, boxes, gt_boxes, expr, save_name='output.png'):
    ### ==> TODO: 可视化Visual Grounding结果，包括给定图像、针对图像中对象的描述和对应对象的坐标框
    img_draw = img.copy()  # img: Image.Image or numpy

    draw = ImageDraw.Draw(img_draw)
    box_width = 3
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)

    def draw_box(draw, box, color, text):
        x0, y0, x1, y1 = box
        draw.rectangle([x0, y0, x1, y1], outline=color, width=box_width)
        text_x = x0
        text_y = min(img_draw.height - 25, y1 + 5)
        text_bbox = draw.textbbox((text_x, text_y), text, font=font)
        # 获取文字绘制后所占据的矩形范围text_bbox
        draw.rectangle(text_bbox, fill=color, outline=color)
        draw.text((text_x, text_y), text, fill="white", font=font)

    draw_box(draw, gt_boxes, 'green', f"GT: {expr}")
    draw_box(draw, boxes, 'blue', f"Pred: {expr}")

    img_draw.save(save_name)
    print(f"可视化结果保存至：{save_name}")
    ### <===

def box_iou(boxes1, boxes2):
    # boxes1:  [N, 2]
    # boxes2:  [M, 2]
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])   # [N, M, 2]  表示交集框的左上角
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])   # [N, M, 2]  表示交集框的右下角

    wh = (rb - lt).clamp(min=0)   # [N, M, 2]  表示交集框的宽和高
    inter = wh[:, :, 0] * wh[:, :, 1]   # [N, M]  表示交集框的面积

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou

def eval_model(args):
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

    input_data = read_jsonlines(args.question_file)

    if args.index is not None:
        if args.index < 0 or args.index >= len(input_data):
            print(f"错误：索引 {args.index} 超出范围（数据集大小为 {len(input_data)}）")
            return
        input_data = [input_data[args.index]]

    with torch.no_grad():
        correct = total_cnt = 0
        no_match_cnt = 0

        for idx, item in enumerate(tqdm(input_data)):
            image_path = os.path.join(args.image_dir, item['img_path'])
            expr = item['expression']
            bbox = item['bbox']
            prompt = f"Where is {expr} in image? answer in [x0,y0,x1,y1] format."
            msgs = [{"role": "user", "content": prompt}]

            if len(image_path) > 1000:
                image = Image.open(io.BytesIO(base64.b64decode(image_path))).convert('RGB')
            else:
                image = Image.open(image_path).convert('RGB')

            answer = model.chat(
                image=image,
                msgs=msgs,
                context=None,
                tokenizer=tokenizer,
                sampling=args.sampling,
                processor=processor
            )

            # 提取预测框
            bbox_pattern = r'\[(\d+(?:\.\d+)?),(\d+(?:\.\d+)?),(\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\]|<box>(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)</box>'
            match = re.search(bbox_pattern, answer)
            if match:
                numbers = [float(x) for x in match.groups() if x is not None]
                answer_bbox = numbers
            else:
                no_match_cnt += 1
                print('No match found in answer, skipping.')
                continue

            iou = box_iou(torch.tensor(bbox).unsqueeze(0), torch.tensor(answer_bbox).unsqueeze(0))
            is_correct = iou > 0.5
            if is_correct:
                correct += 1
            total_cnt += 1

            # 可视化并保存
            vis_name = os.path.join(args.save_dir, f"vg_output_idx_{args.index if args.index is not None else idx}.png")
            vis_boxes(image, answer_bbox, bbox, expr, save_name=vis_name)

            result = {
                "index": args.index if args.index is not None else idx,
                "expression": expr,
                "ground_truth_bbox": bbox,
                "predicted_bbox": answer_bbox,
                "iou": float(iou.item()),
                "correct": bool(is_correct),
                "raw_answer": answer
            }

            # 保存结果到 JSON 文件
            save_json_path = os.path.join(args.save_dir, f"vg_result_idx_{args.index if args.index is not None else idx}.json")
            with open(save_json_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"已保存预测结果到: {save_json_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str)
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--image-dir", type=str)
    parser.add_argument("--sampling", action='store_true')
    parser.add_argument("--vis-nums", type=int, default=5)
    parser.add_argument("--save-dir", type=str, default='output')
    # 加 parser 参数
    parser.add_argument("--index", type=int, default=None, help="只评估指定索引的样本")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    eval_model(args)