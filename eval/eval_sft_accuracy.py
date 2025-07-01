#!/usr/bin/env python3
"""
计算SFT模型答案准确率的脚本
比较模型生成的答案与test.json中的标准答案
"""

import json
import argparse
import os
from typing import Dict, List, Tuple
import re
import copy

def normalize_answer(answer: str) -> str:
    """
    规范化答案，用于比较
    """
    if not answer:
        return ""
    
    # 转换为小写
    answer = answer.lower().strip()
    
    # 移除标点符号和多余空格
    answer = re.sub(r'[^\w\s]', '', answer)
    answer = re.sub(r'\s+', ' ', answer)
    answer = answer.strip()
    
    # 特殊处理：如果答案以yes或no开头，优先考虑
    if answer.startswith('yes ') or answer == 'yes':
        return 'yes'
    elif answer.startswith('no ') or answer == 'no':
        return 'no'
    
    return answer

def load_ground_truth(test_json_path: str) -> Dict[int, str]:
    """
    从test.json加载标准答案
    返回：{id: 标准答案}的字典
    """
    with open(test_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    ground_truth = {}
    for item in data:
        item_id = item['id']
        # 找到assistant的回答
        for conv in item['conversations']:
            if conv['role'] == 'assistant':
                ground_truth[item_id] = conv['content']
                break
    
    return ground_truth

def load_model_answers(model_answer_path: str) -> Dict[int, str]:
    """
    从模型答案文件加载答案
    返回：{id: 模型答案}的字典
    """
    model_answers = {}
    with open(model_answer_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                item = {k: v for k, v in item.items() if k not in ["model", "image_path"]}
                model_answers[item['id']] = item['answer']
    
    return model_answers

def calculate_accuracy(ground_truth: Dict[int, str], model_answers: Dict[int, str]) -> Tuple[float, int, int, List[Dict]]:
    """
    计算准确率
    返回：(准确率, 正确数量, 总数量, 错误案例列表)
    """
    correct = 0
    total = 0
    error_cases = []
    
    # 找到共同的ID
    common_ids = set(ground_truth.keys()) & set(model_answers.keys())
    
    for item_id in common_ids:
        total += 1
        gt_answer = normalize_answer(ground_truth[item_id])
        model_answer = normalize_answer(model_answers[item_id])
        
        if gt_answer == model_answer:
            correct += 1
        else:
            error_cases.append({
                'id': item_id,
                'ground_truth': ground_truth[item_id],
                'model_answer': model_answers[item_id],
                'normalized_gt': gt_answer,
                'normalized_model': model_answer
            })
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total, error_cases

def main():
    parser = argparse.ArgumentParser(description='计算SFT模型答案准确率')
    parser.add_argument('--ground-truth', type=str, default='../data/test.json',
                       help='测试数据文件路径 (default: ../data/test.json)')
    parser.add_argument('--model-answer', type=str, default='../eval_output/sft_ckpt_90_answer_1.jsonl',
                       help='模型答案文件路径 (default: ../eval_output/sft_ckpt_90_answer_1.jsonl)')
    parser.add_argument('--output-errors', type=str, default=None,
                       help='输出错误案例到文件 (可选)')
    parser.add_argument('--show-errors', action='store_true',
                       help='显示错误案例')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.ground_truth):
        print(f"错误：测试数据文件不存在: {args.ground_truth}")
        return
    
    if not os.path.exists(args.model_answer):
        print(f"错误：模型答案文件不存在: {args.model_answer}")
        return
    
    print("正在加载数据...")
    
    # 加载标准答案
    ground_truth = load_ground_truth(args.ground_truth)
    print(f"加载了 {len(ground_truth)} 个标准答案")
    
    # 加载模型答案
    model_answers = load_model_answers(args.model_answer)
    print(f"加载了 {len(model_answers)} 个模型答案")
    
    # 计算准确率
    accuracy, correct, total, error_cases = calculate_accuracy(ground_truth, model_answers)
    
    # 显示结果
    print("\n" + "="*50)
    print("准确率计算结果")
    print("="*50)
    print(f"总问题数量: {total}")
    print(f"答对数量: {correct}")
    print(f"答错数量: {total - correct}")
    print(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("="*50)
    
    # 显示或保存错误案例
    if error_cases:
        if args.show_errors:
            print(f"\n错误案例 (共{len(error_cases)}个):")
            print("-" * 100)
            for i, case in enumerate(error_cases[:10]):  # 只显示前10个
                print(f"案例 {i+1} (ID: {case['id']}):")
                print(f"  标准答案: '{case['ground_truth']}'")
                print(f"  模型答案: '{case['model_answer']}'")
                print(f"  规范化标准答案: '{case['normalized_gt']}'")
                print(f"  规范化模型答案: '{case['normalized_model']}'")
                print("-" * 50)
            
            if len(error_cases) > 10:
                print(f"... 还有 {len(error_cases) - 10} 个错误案例")
        
        if args.output_errors:
            error_cases_to_write = copy.deepcopy(error_cases)
            accuracy_dict = {
                'accuracy': accuracy,
                'correct': correct,
                'total': total
            }
            error_cases_to_write.append(accuracy_dict)
            with open(args.output_errors, 'w', encoding='utf-8') as f:
                json.dump(error_cases_to_write, f, ensure_ascii=False, indent=2)

            print(f"\n错误案例已保存到: {args.output_errors}")
    
    # 分析错误类型
    if error_cases:
        print(f"\n错误分析:")
        yes_to_no = sum(1 for case in error_cases if case['normalized_gt'] == 'yes' and case['normalized_model'] == 'no')
        no_to_yes = sum(1 for case in error_cases if case['normalized_gt'] == 'no' and case['normalized_model'] == 'yes')
        other_errors = len(error_cases) - yes_to_no - no_to_yes
        
        print(f"  Yes -> No 错误: {yes_to_no}")
        print(f"  No -> Yes 错误: {no_to_yes}")
        print(f"  其他错误: {other_errors}")

if __name__ == '__main__':
    main() 