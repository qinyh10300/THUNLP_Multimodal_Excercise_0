import os
import json
import numpy as np
from numpy import random as nprdm
import random
import tqdm
import multiprocessing
import argparse
import threading

import re

random.seed(71)
nprdm.seed(71)


IMAGE_PLACEHOLDER = '<image>'
BOXES_PLACEHOLDER = '<boxes>'
EXPR_PLACEHOLDER = '<expr>'   # <expr> means expression like 'microphone'
OBJS_PLACEHOLDER = '<objs>'   # <objs> means bbox i think
QUESTION_PLACEHOLDER = '<question>'
POINTS_PLACEHOLDER = '<points>'
PHRASE_ST_PLACEHOLDER = '<ph_st>'
PHRASE_ED_PLACEHOLDER = '<ph_ed>'



class RECDataset():
    '''
    Question: In <image>, I need the bounding box coordinates of <expr>.
    Answer: boxes
    '''
    def __init__(self, filename="",
                 template_file="",
                 image_folders={
                    'images/VG_100K': '',
                    'images2/VG_100K_2':''
                 },
                 version = 'vg',
                 total = None,
                 ratio = None,
                 shuffle = False,
                ):

        self.datafile = filename
        self.templates = json.load(open(template_file))
        self.image_dirs = image_folders
        self.version = version

        self.total = total
        self.ratio = ratio
        self.shuffle = shuffle

    def get_template(self,):
        return nprdm.choice(self.templates, 1)[0]

    def build(self, return_dict=None, dict_key="key"):
        
        ### ==> TODO: 实现Referring Expression Comprehension数据集
        result = []
        with open(self.datafile, 'r') as f:
            data = [json.loads(line.strip()) for line in f]

        # 根据total、ratio、shuffle对数据进行处理
        if self.shuffle:
            random.shuffle(data)
        if self.ratio:
            data = data[:int(len(data) * self.ratio)]
        if self.total:
            data = data[:self.total]

        # 遍历数据，根据模板构建对话
        for item in tqdm.tqdm(data, desc='Processing RECDataset...'):
            img_path = item['img_path']
            # if self.version == 'vg':
            #     if 'images/VG_100K' in img_path:
            #         img_path = os.path.join(self.image_dirs.get('images/VG_100K', ''), img_path)
            #     elif 'images2/VG_100K_2' in img_path:
            #         img_path = os.path.join(self.image_dirs.get('images2/VG_100K_2', ''), img_path)
            # elif self.version == 'coco':
            #     if 'train2014' in img_path:
            #         img_path = os.path.join(self.image_dirs.get('train2014', ''), img_path)
            flag = 0
            for folder_key in self.image_dirs:
                if folder_key in img_path:
                    img_path = os.path.join(self.image_dirs[folder_key], img_path)
                    flag = 1
                    break

            if flag == 0:
                print(img_path)
                exit(0)

            expression = item['expression']
            bbox = item['bbox']
            bbox_str = f'[{bbox[0]:.2f},{bbox[1]:.2f},{bbox[2]:.2f},{bbox[3]:.2f}]'
        
            template = self.get_template()
            question = template.replace(EXPR_PLACEHOLDER, expression)

            conversation = [
                {
                    "role": "user",
                    "content": question,
                },
                {
                    "role": "assistant",
                    "content": bbox_str,
                },
            ]

            result.append({
                "if": len(result),
                "image": img_path,
                "conversations": conversation
            })

        if return_dict is not None:
            return_dict[dict_key] = result
        ### <===

        return result



class GCDataset():
    '''
    question: Can you give me a description of the region <objs> in image <image>?
    answer: description
    '''
    def __init__(self, filename="",
                 template_file="",
                 image_folders={
                    'images/VG_100K': '',
                    'images2/VG_100K_2':''
                 },
                 total = None,
                 ratio = None,
                 shuffle = False,
                ):

        self.datafile = filename
        self.templates = json.load(open(template_file))
        self.image_dirs = image_folders

        self.total = total
        self.ratio = ratio
        self.shuffle = shuffle

    def get_template(self,):
        return nprdm.choice(self.templates, 1)[0]

    def build(self, return_dict=None, dict_key="key"):
        ### ==> TODO: 实现Grounded Captioning数据集
        result = []
        with open(self.datafile, 'r') as f:
            data = [json.loads(line.strip()) for line in f]

        if self.shuffle:
            random.shuffle(data)
        if self.ratio:
            data = data[:int(len(data) * self.ratio)]
        if self.total:
            data = data[:self.total]

        for item in tqdm.tqdm(data, desc="Processing GC data"):
            template = self.get_template()

            img_path = item['img_path']
            flag = 0
            for folder_key in self.image_dirs:
                if folder_key in img_path:
                    img_path = os.path.join(self.image_dirs[folder_key], img_path)
                    flag = 1
                    break

            if flag == 0:
                print(img_path)
                exit(0)
            
            bbox = item['bbox']
            bbox_str = f"[{bbox[0]:.2f},{bbox[1]:.2f},{bbox[2]:.2f},{bbox[3]:.2f}]"

            question = template.replace(OBJS_PLACEHOLDER, bbox_str)
            answer = item['expression']

            conversation = [
                {
                    "role": "user",
                    "content": question,
                },
                {
                    "role": "assistant",
                    "content": answer,
                },
            ]

            result.append({
                "id": len(result),
                "image": img_path,
                "conversations": conversation,
            })

        if return_dict is not None:
            return_dict[dict_key] = result

        ### <===w

        return result
    
class REGDataset():
    '''
    question: For the given image <image>, can you provide a unique description of the area <objs>?
    answer: description
    '''
    def __init__(self, filename="",
                 template_file="",
                 image_folders={
                    'train2014': '',
                    'val2014':''
                 },
                 total = None,
                 ratio = None,
                 shuffle = False,
                ):

        self.datafile = filename
        self.templates = json.load(open(template_file))
        self.image_dirs = image_folders

        self.total = total
        self.ratio = ratio
        self.shuffle = shuffle

    def get_template(self,):
        return nprdm.choice(self.templates, 1)[0]

    def build(self, return_dict=None, dict_key="key"):
        ### ==> TODO: 实现Referring Expression Generation数据集
        result = []
        with open(self.datafile, 'r') as f:
            data = [json.loads(line.strip()) for line in f]

        if self.shuffle:
            random.shuffle(data)
        if self.ratio:
            data = data[:int(len(data) * self.ratio)]
        if self.total:
            data = data[:self.total]

        for item in tqdm.tqdm(data, desc="Processing REG data"):
            template = self.get_template()

            img_path = item['img_path']
            flag = 0
            for folder_key in self.image_dirs:
                if folder_key in img_path:
                    img_path = os.path.join(self.image_dirs[folder_key], img_path)
                    flag = 1
                    break

            if flag == 0:
                print(img_path)
                exit(0)

            bbox = item['bbox']
            bbox_str = f"[{bbox[0]:.2f},{bbox[1]:.2f},{bbox[2]:.2f},{bbox[3]:.2f}]"

            question = template.replace(OBJS_PLACEHOLDER, bbox_str)
            answer = item['expression']

            conversation = [
                {
                    "role": "user",
                    "content": question,
                },
                {
                    "role": "assistant",
                    "content": answer,
                },
            ]

            result.append({
                "id": len(result),
                "image": img_path,
                "conversations": conversation,
            })

        if return_dict is not None:
            return_dict[dict_key] = result
        ### <===
        return result



class FlickrDataset():
    '''
    question: Can you provide a description of the image <image> and include the coordinates [x0,y0,x1,y1] for each mentioned object?
    answer: description+boxes
    '''
    def __init__(self, filename="",
                 template_file="",
                 image_folders={
                    'flickr30k': '',
                 },
                 total = None,
                 ratio = None,
                 shuffle = False,
                ):

        self.datafile = filename
        self.templates = json.load(open(template_file))
        self.image_dirs = image_folders

        self.total = total
        self.ratio = ratio
        self.shuffle = shuffle

    def get_template(self,):
        return nprdm.choice(self.templates, 1)[0]

    def build(self, return_dict=None, dict_key="key"):
        ### ==> TODO: 实现Flik30K-entities数据集
        result = []

        with open(self.datafile, 'r') as f:
            data = [json.loads(line.strip()) for line in f]

        # 根据total、ratio、shuffle对数据进行处理
        if self.shuffle:
            random.shuffle(data)
        if self.ratio:
            data = data[:int(len(data) * self.ratio)]
        if self.total:
            data = data[:self.total]

        for item in tqdm.tqdm(data, desc="Processing Flickr data"):
            template = self.get_template()

            img_path = f"{item['image_id']}.jpg"
            # img_path = os.path.json(self.image_dirs[folder_key], img_path)
            for folder_key in self.image_dirs:
                # if folder_key in img_path:
                #     # 应该是'flickr30k'
                #     img_path = os.path.join(self.image_dirs[folder_key], img_path)
                #     break
                # # 这部分有bug，img_path里面不会出现folder_key也就是flickr30k
                # 所以直接：
                img_path = os.path.join(self.image_dirs[folder_key], img_path)
                break
            # flag = 0
            # for folder_key in self.image_dirs:
            #     if folder_key in img_path:
            #         img_path = os.path.join(self.image_dirs[folder_key], img_path)
            #         flag = 1
            #         break

            # if flag == 0:
            #     print(img_path)
            #     exit(0)

            sentence = item['sentence']
            boxes = item['boxes']
            boxes_seq = item['boxes_seq']
            # <ph_st>microphones<ph_ed> .", 
            # "boxes_seq": [[3], [0], [1, 2]]

            answer = process_phrase_placeholder(sentence, boxes, boxes_seq)

            conversation = [
                {
                    "role": "user",
                    "content": template,
                },
                {
                    "role": "assistant",
                    "content": answer,
                }
            ]

            result.append({
                "id": len(result),
                "image": img_path,
                "conversations": conversation,
            })

        if return_dict is not None:
            return_dict[dict_key] = result
        ### <===

        return result


 

class GPT4GenDataset():
    '''
    a: description
    c: description + CoT
    bc: description + CoT + boxes
    '''
    def __init__(self, filename="",
                 template_file="",
                 image_folders={
                    'flickr30k': '',
                 },
                 version='p',
                 total = None,
                 ratio = None,
                 shuffle = False,
                ):

        self.datafile = filename
        self.templates = json.load(open(template_file))
        self.image_dirs = image_folders

        self.version = version
        assert version in ['a', 'c', 'bc']

        self.total = total
        self.ratio = ratio
        self.shuffle = shuffle

    def get_template(self,):
        return nprdm.choice(self.templates, 1)[0]

    def build(self, return_dict=None, dict_key="key"):
        ### ==> TODO: 实现GPT-4生成的数据集
        result = []

        with open(self.datafile, 'r') as f:
            data = [json.loads(line.strip()) for line in f]

        if self.shuffle:
            random.shuffle(data)
        if self.ratio:
            data = data[:int(len(data) * self.ratio)]
        if self.total:
            data = data[:self.total]

        for item in tqdm.tqdm(data, desc="Processing GPT4Gen data"):
            template = self.get_template()
            
            img_path = item['img_path']
            # for folder_key in self.image_dirs:
            #     if folder_key in img_path:
            #         img_path = os.path.join(self.image_dirs[folder_key], img_path)
            #         break

            flag = 0
            for folder_key in self.image_dirs:
                # if folder_key in img_path:
                #     img_path = os.path.join(self.image_dirs[folder_key], img_path)
                #     flag = 1
                #     break
                # # 这部分有bug，img_path里面不会出现folder_key
                # # 所以直接：
                img_path = os.path.join(self.image_dirs[folder_key], img_path)
                flag = 1

            if flag == 0:
                print(img_path)
                exit(0)

            question = item['question']
            question = process_phrase_placeholder(question, item['boxes'], item['question_boxes_seq'])

            if self.version == 'a':
                # 仅回答版本
                answer = item['answer']
            elif self.version == 'c':
                # 带推理链版本
                answer = item['cot_with_ans']
                answer = answer.replace(PHRASE_ST_PLACEHOLDER, '')
                answer = answer.replace(PHRASE_ED_PLACEHOLDER, '')
            elif self.version == 'bc':
                # 带边界框的推理链版本
                cot_text = item['cot_with_ans']
                boxes = item['boxes']
                answer_boxes_seq = item['answer_boxes_seq']

                answer = process_phrase_placeholder(cot_text, boxes, answer_boxes_seq)

            conversation = [
                {
                    "role": "user",
                    "content": template.replace(QUESTION_PLACEHOLDER, question)
                },
                {
                    "role": "assistant",
                    "content": answer
                }
            ]
            
            result.append({
                "id": len(result),
                "image": img_path,
                "conversations": conversation
            })
        
        if return_dict is not None:
            return_dict[dict_key] = result
        ### <===

        return result
    
def process_phrase_placeholder(text, boxes, boxes_seq):
    '''
    boxes  List[List[float]]  图像中所有检测目标的位置信息
    boxes_seq  List[List[int]]  第i个<ph>对应的boxes的index
    '''

    # remove <ph_st>
    text = text.replace(PHRASE_ST_PLACEHOLDER, '')

    # replace <ph_ed>
    parts = re.split(r'(<ph_ed>)', text)
    # re means regular expression 正则表达式
    ph_index = 0
    for i, part in enumerate(parts):
        if part == PHRASE_ED_PLACEHOLDER:
            bbox_strs = []
            for idx in boxes_seq[ph_index]:
                bbox_strs.append(f"[{boxes[idx][0]:.2f}],[{boxes[idx][1]:.2f}],[{boxes[idx][2]:.2f}],[{boxes[idx][3]:.2f}]")
            parts[i] = ''.join(bbox_strs)
            ph_index += 1
    return ''.join(parts)

if __name__ == '__main__':

    datasets = [
        RECDataset(filename="grouding_data/GC_genome196_train.jsonl", 
                  template_file="template/REC.json",
                  version='vg', 
                  ratio=1/20,
                  total=1000,
                  image_folders={'images/VG_100K': '/home/user0/qinyh/codebase/THUNLP_Multimodal_Excercise/data',
                                 'images2/VG_100K_2': '/home/user0/qinyh/codebase/THUNLP_Multimodal_Excercise/data'}),
        RECDataset(filename="grouding_data/REC_ref3_train.jsonl", 
                  template_file="template/REC.json",
                  image_folders={'train2014': '/home/user0/qinyh/codebase/THUNLP_Multimodal_Excercise/data/images/train2014'}, 
                  version='coco',
                  total=1000,
                  ),
        GCDataset(filename="grouding_data/GC_genome196_train.jsonl",
                 template_file="template/GC.json", 
                 ratio=1/20,
                 total=1000,
                 image_folders={'images/VG_100K': '/home/user0/qinyh/codebase/THUNLP_Multimodal_Excercise/data',
                                 'images2/VG_100K_2': '/home/user0/qinyh/codebase/THUNLP_Multimodal_Excercise/data'}),
        REGDataset(filename="grouding_data/REC_ref3_train.jsonl",
                  template_file="template/REG.json",
                  total=1000,
                  image_folders={'train2014': '/home/user0/qinyh/codebase/THUNLP_Multimodal_Excercise/data/images/train2014'}),
        FlickrDataset(filename="grouding_data/CWB_flickr30k_train.jsonl",
                     template_file="template/flickr30k.json",
                     total=1000,
                     image_folders={'flickr30k': '/home/user0/qinyh/codebase/THUNLP_Multimodal_Excercise/data/images/flickr30k-images'}),
        GPT4GenDataset(filename="grouding_data/GPT4GEN_BoxCoT_train.jsonl",
                      version='a', 
                      template_file="template/VQA.json",
                      total=1000,
                      image_folders={'flickr30k': '/home/user0/qinyh/codebase/THUNLP_Multimodal_Excercise/data/images/flickr30k-images'}),
        GPT4GenDataset(filename="grouding_data/GPT4GEN_BoxCoT_train.jsonl",
                      version='c', 
                      template_file="template/VQA_CoT.json",
                      total=1000,
                      image_folders={'flickr30k': '/home/user0/qinyh/codebase/THUNLP_Multimodal_Excercise/data/images/flickr30k-images'}),
        GPT4GenDataset(filename="grouding_data/GPT4GEN_BoxCoT_train.jsonl",
                      version='bc', 
                      template_file="template/VQA_BCoT.json",
                      total=1000,
                      image_folders={'flickr30k': '/home/user0/qinyh/codebase/THUNLP_Multimodal_Excercise/data/images/flickr30k-images'}),
        GPT4GenDataset(filename='grouding_data/GPT4GEN_RD_BoxCoT_train.jsonl',
                      version='bc', 
                      template_file="template/VQA_BCoT.json",
                      total=1000,
                      image_folders={'flickr30k': '/home/user0/qinyh/codebase/THUNLP_Multimodal_Excercise/data/images/flickr30k-images'}),
    ]

    val_datasets = [
        RECDataset(filename='grouding_data/REC_refcoco_unc_val.jsonl',
                   template_file='template/REC.json',
                   version='coco',
                   image_folders={'train2014': '/home/user0/qinyh/codebase/THUNLP_Multimodal_Excercise/data/images/train2014'},
                   ratio=1/20,
                   total=500),
        RECDataset(filename='grouding_data/REC_refcoco+_unc_val.jsonl',
                   template_file='template/REC.json',
                   version='coco',
                   image_folders={'train2014': '/home/user0/qinyh/codebase/THUNLP_Multimodal_Excercise/data/images/train2014'},
                   ratio=1/20,
                   total=500),
        RECDataset(filename='grouding_data/REC_refcocog_umd_val.jsonl',
                   template_file='template/REC.json',
                   version='coco',
                   image_folders={'train2014': '/home/user0/qinyh/codebase/THUNLP_Multimodal_Excercise/data/images/train2014'},
                   ratio=1/20,
                   total=500),
        # REGDataset(filename='grouding_data/REC_refcoco_unc_val.jsonl',
        #            template_file='template/REG.json',
        #            image_folders={'train2014': '/home/user0/qinyh/codebase/THUNLP_Multimodal_Excercise/data/images/train2014'},
        #            ratio=1/20),
        # REGDataset(filename='grouding_data/REC_refcoco+_unc_val.jsonl',
        #            template_file='template/REG.json',
        #            image_folders={'train2014': '/home/user0/qinyh/codebase/THUNLP_Multimodal_Excercise/data/images/train2014'},
        #            ratio=1/20),
        # REGDataset(filename='grouding_data/REC_refcocog_umd_val.jsonl',
        #            template_file='template/REG.json',
        #            image_folders={'train2014': '/home/user0/qinyh/codebase/THUNLP_Multimodal_Excercise/data/images/train2014'},
        #            ratio=1/20),
        # FlickrDataset(filename='grouding_data/CWB_flickr30k_eval.jsonl',
        #               template_file='template/flickr30k.json',
        #               image_folders={'flickr30k': '/home/user0/qinyh/codebase/THUNLP_Multimodal_Excercise/data/images/flickr30k-images'},
        #               ratio=1/20),
        # GPT4GenDataset(filename='grouding_data/GPT4GEN_BoxCoT_test.jsonl',
        #               version='a', 
        #               template_file='template/VQA.json',
        #               image_folders={'flickr30k': '/home/user0/qinyh/codebase/THUNLP_Multimodal_Excercise/data/images/flickr30k-images'},
        #               ratio=1/20),
        # GPT4GenDataset(filename='grouding_data/GPT4GEN_BoxCoT_test.jsonl',
        #               version='c', 
        #               template_file='template/VQA_CoT.json',
        #               image_folders={'flickr30k': '/home/user0/qinyh/codebase/THUNLP_Multimodal_Excercise/data/images/flickr30k-images'},
        #               ratio=1/20),
        # GPT4GenDataset(filename='grouding_data/GPT4GEN_BoxCoT_test.jsonl',
        #               version='bc', 
        #               template_file='template/VQA_BCoT.json',
        #               image_folders={'flickr30k': '/home/user0/qinyh/codebase/THUNLP_Multimodal_Excercise/data/images/flickr30k-images'},
        #               ratio=1/20)
    ]

    ### ==> TODO: 实现用于Visual Grounding的指令微调数据集的构建
    # VG任务的数据集构建
    def build_datasets(datasets):
        tot = 0
        results = []

        for i, dataset in enumerate(datasets):
            print(f"Processing dataset {i+1}/{len(datasets)}: {dataset.__class__.__name__}")
            try:
                data = dataset.build()
                results.extend(data)
                tot += len(data)
                print(f"Added {len(data)} examples from {dataset.__class__.__name__}")
            except Exception as e:
                print(f"Error processing {dataset.__class__.__name__}: {e}")

        random.shuffle(results)

        for i, item in enumerate(results):
            item['id'] = i

            return results, tot

    train_results, train_tot = build_datasets(datasets)
    val_results, val_tot = build_datasets(val_datasets)
    ### <===

    # save
    # Train
    with open(f"train_minicpmv_grounding_{train_tot}.json", 'w') as f:
        json.dump(train_results, f, indent=4)
    print("Total # exmaples: %d" % train_tot)
    # Val
    with open(f"val_minicpmv_grounding_{val_tot}.json", 'w') as f:
        json.dump(val_results, f, indent=4)
    print("Total # examples: %d" % val_tot)