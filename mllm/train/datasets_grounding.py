import os, re
import json
import pickle
import random
import torch
import logging
import pandas as pd
import os.path as op
import transformers
from torch.utils.data import Dataset
import math

from PIL import Image
from typing import Dict
from utils.file_io import read_json, bytes_to_PIL_image
from mllm.train.preprocess import preprocess
from mllm.train.inference_logp import get_dataset_inference_logp
from mllm.train.preprocess import find_best_resize

logger = logging.getLogger(__name__)



class GroundingSupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        raw_data,
        transform,
        tokenizer,
        slice_config,
        patch_size=14,
        query_nums=64,
        batch_vision=False,
        max_length=2048,
    ):
        super(GroundingSupervisedDataset, self).__init__()
        self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.transform = transform
        self.slice_config = slice_config
        self.patch_size = patch_size
        self.query_nums=query_nums
        self.batch_vision = batch_vision
        self.max_length = max_length

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        ### ==> TODO: Visual Grounding数据处理流程
        # ret = dict(
        #     input_ids = ,
        #     position_ids = ,
        #     labels = ,
        #     attention_mask = ,
        #     pixel_values = ,
        #     tgt_sizes = ,
        #     image_bound = ,
        # )

        # print(self.raw_data[i]["image"])

        processed_data = preprocess(
            images_dict={"<image>": Image.open(self.raw_data[i]["image"]).convert("RGB")},
            conversations=self.raw_data[i]["conversations"],
            tokenizer=self.tokenizer,
            transform=self.transform,
            query_nums=self.query_nums,
            slice_config=self.slice_config,
            patch_size=self.patch_size,
            batch_vision=self.batch_vision,
            max_length=self.max_length
        )

        ret = dict(
            input_ids = processed_data["input_ids"],
            position_ids = processed_data["position_ids"],
            labels = processed_data["target"],
            attention_mask = torch.ones_like(processed_data["input_ids"], dtype=torch.bool),
            pixel_values = processed_data["pixel_values"],
            tgt_sizes = processed_data["tgt_sizes"],
            image_bound = processed_data["image_bound"],
        )

        return ret
        ### <===
