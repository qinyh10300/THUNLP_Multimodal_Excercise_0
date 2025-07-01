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

    ### TODO: Implement inference loop
    with torch.no_grad():
        correct = total_cnt = 0
        num = args.vis_nums
        no_match_cnt = 0
        for item in tqdm(input_data):
            image = os.path.join(args.image_dir, item['img_path'])
            expr = item['expression']
            # image = os.path.join(args.image_dir, item['image'].split('/')[-1])
            # expr = item['sent']
            bbox = item['bbox']
            prompt = "Where is {} in image? answer in [x0,y0,x1,y1] format.".format(expr)
            
            msgs = [{"role": "user", "content": prompt}]

            if len(image) > 1000:
                image = Image.open(io.BytesIO(base64.b64decode(image))).convert('RGB')
            else:
                image = Image.open(image).convert('RGB')

            answer = model.chat(
                image=image,
                msgs=msgs,
                context=None,
                tokenizer=tokenizer,
                sampling=args.sampling,
                processor=processor
            )

            # Calculate acc
            ### ==> TODO: 实现Visual Grounding的结果准确率计算方法
            bbox_pattern = r'\[(\d+(?:\.\d+)?),(\d+(?:\.\d+)?),(\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\]|<box>(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)</box>'
            match = re.search(bbox_pattern, answer)
            if match:
                numbers = [float(x) for x in match.groups() if x is not None]
                print(numbers)
                answer_bbox = numbers
            else:
                no_match_cnt += 1
                print('no match and continue', no_match_cnt)
                continue

            iou = box_iou(torch.tensor(bbox).unsqueeze(0), torch.tensor(answer_bbox).unsqueeze(0))
            # IoU（Intersection over Union）衡量两个边界框之间的重叠程度（交集面积比上并集面积）
            # 1表示完全重合
            # 0表示完全不相交
            if iou > 0.5:
                correct += 1
            total_cnt += 1
            ### <===

            # Visualize VG results
            ### ==> TODO: 实现Visual Grounding结果的可视化
            if args.vis_nums > 0:
                vis_boxes(image, answer_bbox, bbox, expr, save_name=os.path.join(args.save_dir, 'output_{}.png'.format(num - args.vis_nums)))
                args.vis_nums -= 1
            ### <===

    print(f"Evaluating {args.question_file} ...")
    print(f'Precision @ 1: {correct / total_cnt} \n')
    print(f'No match cnt: {no_match_cnt}')

    with open(os.path.join(args.save_dir, 'eval_result.json'), 'w') as f:
        json.dump({'precision@1': correct / total_cnt, 'no_match_cnt': no_match_cnt, 'total_cnt': total_cnt}, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str)
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--image-dir", type=str)
    parser.add_argument("--sampling", action='store_true')
    parser.add_argument("--vis-nums", type=int, default=5)
    parser.add_argument("--save-dir", type=str, default='output')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    eval_model(args)