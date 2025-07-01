import math
from typing import List, Optional
import json
import torch
import torchvision

from threading import Thread
from copy import deepcopy
from PIL import Image
from transformers import AutoProcessor, TextIteratorStreamer

from .configuration import ModelConfig
from .modeling_navit_siglip import SiglipVisionTransformer
from .resampler import Resampler

from .image_processing import ModelImageProcessor
from .processing import ModelProcessor
from .llm.llm_architecture import LLMPreTrainedModel, LLMForCausalLM


class MLLMPreTrainedModel(LLMPreTrainedModel):
    config_class = ModelConfig


class MLLMModel(MLLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.llm = LLMForCausalLM(config)
        self.vpm = self.init_vision_module()
        self.vision_dim = self.vpm.embed_dim
        self.embed_dim = self.llm.config.hidden_size
        self.resampler = self.init_resampler(self.embed_dim, self.vision_dim)
        self.processor = None

        self.terminators = ['<|im_end|>', '<|endoftext|>']

    def init_vision_module(self):
        # same as HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit add tgt_sizes
        if self.config._attn_implementation == 'flash_attention_2':
            self.config.vision_config._attn_implementation = 'flash_attention_2'
        else:
            # not suport sdpa
            self.config.vision_config._attn_implementation = 'eager'
        model = SiglipVisionTransformer(self.config.vision_config)
        if self.config.drop_vision_last_layer:
            model.encoder.layers = model.encoder.layers[:-1]

        setattr(model, 'embed_dim', model.embeddings.embed_dim)
        setattr(model, 'patch_size', model.embeddings.patch_size)

        return model

    def init_resampler(self, embed_dim, vision_dim):
        return Resampler(
            num_queries=self.config.query_num,
            embed_dim=embed_dim,
            num_heads=embed_dim // 128,
            kv_dim=vision_dim,
            adaptive=True
        )

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.embed_tokens = value

    def get_output_embeddings(self):
        return self.llm.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.llm.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.llm = decoder

    def get_decoder(self):
        return self.llm

    def get_vllm_embedding(self, data):
        vision_hidden_states = self.get_vision_hidden_states(data)

        # for k, v in data.items():
        #     print(k, v, type(v))
        #     print('--------------------------------')

        # input_ids: 语言模型的token序列 (token_id)
        # torch.Tensor  shape=[batch_size, seq_len]  dtype=torch.int32
        # [[151644, 8948, ···, 198]]    128244表示图像占位符的token_id
        
        # image_bound: 表示图像对应在input_ids中被插入的位置index范围
        # List[Tensor[num_img, 2]]，batch中每一个样本一个tensor，其中每一行是[start_idx, end_idx]
        # 这些索引index用于将vision_hidden_states[j]插入vllm_embedding[i][start:end]

        # image_bound [tensor([[ 15,  16],
        # [ 35,  99],
        # [101, 165],
        # [167, 231],
        # [247, 248]], device='cuda:0')] <class 'list'>

        # 正常的image_bound的start_index应该和end_index相差64，这里首位的相差1，可能是人为添加的哨兵（标记）

        # pixel_values: 图像经过预处理（如归一化、resize）后的张量输入，供视觉编码器Visual Encoder使用
        # List[List[Tensor[3, H, W]]]，外层list是batch，内层list是该样本中包含的图像张量（可能有一个或多个图像）
        # Tensor[3, H, W]为图像的RGB通道数据，float类型，像素值可归一化为[-1, 1]

        # tgt_sizes(target_size): 图像在原始输入中的尺寸（用于视觉模型中的位置编码或者反投影（即恢复原先图像尺寸）等用途）
        # List[Tensor[num_img, 2]]，外层list是batch，一个样本一个张量，形状为[num_img, 2]，记录每张图像的(H, W)

        # position_ids: 用于语言模型的位置编码输入，指示每个token的位置信息
        # torch.Tensor[batch_size, seq_len]    dtype=torch.int32或torch.int64（整型）
        # 通常为从0到seq_len-1的整数序列  torch.arrange(seq_len)
        # 用途：例如GPT模型使用绝对位置编码或者rotary position embeddings的时候需要这个

        if hasattr(self.llm.config, 'scale_emb'):
            vllm_embedding = self.llm.model.embed_tokens(data['input_ids']) * self.llm.config.scale_emb
        else:
            vllm_embedding = self.llm.model.embed_tokens(data['input_ids'])

        vision_hidden_states = [i.type(vllm_embedding.dtype) if isinstance(
            i, torch.Tensor) else i for i in vision_hidden_states]

        bs = len(data['input_ids'])
        ### ===> 合并 vision_hidden_states 与 vllm_embedding，
        # # 其中，vision_hidden_states 为视觉编码，当前 vllm_embedding 仅为语言模型编码
        # for i in range(bs):
        #     if 'image_bound' in data and data['image_bound'] is not None:
        #         if i < len(data['image_bound']) and len(data['image_bound'][i]) > 0:
        #             img_bounds = data['image_bound'][i]     # 获取当前批次的图像边界张量（维度应该是[num_images, 2] ??）

        #             # 遍历每个图像的起始和结束位置
        #             for img_idx, bounds in enumerate(img_bounds):
        #                 if img_idx < len(vision_hidden_states[i]) and len(vision_hidden_states[i]) > 0:   # 可能有多余的判断，不过这样更稳健
        #                     # 从张量中获取其实和结束位置
        #                     start_idx = bounds[0].item()
        #                     end_idx = bounds[1].item()

        #                     # 获取当前图像对应的视觉特征
        #                     img_features = vision_hidden_states[i][img_idx]

        #                     print(img_features.shape, end_idx - start_idx)
        #                     if end_idx - start_idx == img_features.shape[0]:
        #                         vllm_embedding[i, start_idx:end_idx] = img_features
        #                     else:
        #                         # 我猜维度不会不匹配
        #                         raise ValueError("维度不匹配了")

        # 注意：要将image_bounds中首位相差不是64的去掉，这些不是图片
        image_bounds = data['image_bound']
        for i in range(len(image_bounds)):
            mask = torch.ones(len(image_bounds[i]), dtype=torch.bool)
            for j in range(len(image_bounds[i])):
                if (image_bounds[i][j][1] - image_bounds[i][j][0]) != 64:
                    mask[j] = False
            image_bounds[i] = image_bounds[i][mask]

        # print(type(vision_hidden_states))
        # # List[torch.tensor] 最外层List表示batch_size

        for i in range(bs):
            cur_vision_hidden_states = vision_hidden_states[i]
            # print(cur_vision_hidden_states.shape, cur_vision_hidden_states)
            # cur_vision_hidden_states.shape = [3, 64, 3584]第一个维度表示这个样本的num_img, 64表示图像映射为64个token，3584表示LLM的标准token的维度（与语言token匹配）
            for j in range(len(cur_vision_hidden_states)):
                slice_start_index = image_bounds[i][j][0]
                slice_end_index = image_bounds[i][j][1]
                vllm_embedding[i][slice_start_index:slice_end_index] = cur_vision_hidden_states[j]
        ### <===

        return vllm_embedding, vision_hidden_states

    def get_vision_hidden_states(self, data):
        # 将输入数据data中的图像信息（pixel values）通过视觉编码器vpm编码成视觉特征（hidden states）
        # 再使用resampler（比如Perceiver Resampler，相当于桥连接层，Adapter）将不定长的视觉token映射为LLM需要的语言token的维度
        if 'vision_hidden_states' not in data:
            dtype = self.llm.model.embed_tokens.weight.dtype
            device = self.llm.model.embed_tokens.weight.device
            tgt_sizes = data['tgt_sizes']
            pixel_values_list = data['pixel_values']
            vision_hidden_states = []
            all_pixel_values = []
            img_cnt = []
            for pixel_values in pixel_values_list:
                img_cnt.append(len(pixel_values))
                all_pixel_values.extend([i.flatten(end_dim=1).permute(1, 0) for i in pixel_values])

            # exist image
            if all_pixel_values:
                tgt_sizes = [tgt_size for tgt_size in tgt_sizes if isinstance(tgt_size, torch.Tensor)]
                tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)

                max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])

                all_pixel_values = torch.nn.utils.rnn.pad_sequence(all_pixel_values, batch_first=True,
                                                                   padding_value=0.0)
                B, L, _ = all_pixel_values.shape
                all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)

                patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool, device=device)
                for i in range(B):
                    patch_attn_mask[i, 0, :tgt_sizes[i][0] * tgt_sizes[i][1]] = True

                vision_batch_size = self.config.vision_batch_size
                all_pixel_values = all_pixel_values.type(dtype)
                if B > vision_batch_size:
                    hs = []
                    for i in range(0, B, vision_batch_size):
                        start_idx = i
                        end_idx = i + vision_batch_size
                        tmp_hs = self.vpm(all_pixel_values[start_idx:end_idx], patch_attention_mask=patch_attn_mask[start_idx:end_idx], tgt_sizes=tgt_sizes[start_idx:end_idx]).last_hidden_state
                        # vpm: Vision Pretrained Module
                        hs.append(tmp_hs)
                    vision_embedding = torch.cat(hs, dim=0)
                else:
                    vision_embedding = self.vpm(all_pixel_values, patch_attention_mask=patch_attn_mask, tgt_sizes=tgt_sizes).last_hidden_state
                vision_embedding = self.resampler(vision_embedding, tgt_sizes)

                start = 0
                for pixel_values in pixel_values_list:
                    img_cnt = len(pixel_values)
                    if img_cnt > 0:
                        vision_hidden_states.append(vision_embedding[start: start + img_cnt])
                        start += img_cnt
                    else:
                        vision_hidden_states.append([])
            else: # no image
                if self.training:
                    dummy_image = torch.zeros(
                        (1, 3, 224, 224),
                        device=device, dtype=dtype
                    )
                    tgt_sizes = torch.Tensor([[(224 // self.config.patch_size), math.ceil(224 / self.config.patch_size)]]).type(torch.int32)
                    dummy_feature = self.resampler(self.vpm(dummy_image).last_hidden_state, tgt_sizes)
                else:
                    dummy_feature = []
                for _ in range(len(pixel_values_list)):
                    vision_hidden_states.append(dummy_feature)

        else:
            vision_hidden_states = data['vision_hidden_states']

        return vision_hidden_states

    def forward(self, data, **kwargs):
        vllm_embedding, vision_hidden_states = self.get_vllm_embedding(data)
        position_ids = data["position_ids"]
        if position_ids.dtype != torch.int64:
            position_ids = position_ids.long()

        return self.llm(
            input_ids=None,
            position_ids=position_ids,
            inputs_embeds=vllm_embedding,
            **kwargs
        )

    def _decode(self, inputs_embeds, tokenizer, attention_mask, decode_text=False, **kwargs):
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        ### ===> TODO: 实现语言模型 generate
        # 使用llm的generate方法生成输出
        generation_kwargs = {   # 仿照下面_decode_stream的写法
            'inputs_embeds': inputs_embeds,
            'attention_mask': attention_mask,
            'pad_token_id': 0,
            'eos_token_id': terminators,
            **kwargs
        }
        output = self.llm.generate(**generation_kwargs)
        ### <===
        if decode_text:
            return self._decode_text(output, tokenizer)
        return output

    def _decode_stream(self, inputs_embeds, tokenizer, **kwargs):
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        streamer = TextIteratorStreamer(tokenizer=tokenizer)
        generation_kwargs = {
            'inputs_embeds': inputs_embeds,
            'pad_token_id': 0,
            'eos_token_id': terminators,
            'streamer': streamer
        }
        generation_kwargs.update(kwargs)

        thread = Thread(target=self.llm.generate, kwargs=generation_kwargs)
        thread.start()

        return streamer

    def _decode_text(self, result_ids, tokenizer):
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        ### TODO: ===> 编写输出解码过程
        # 其中应该去除tokenizer.bos_id（句子起始特殊符号），以及terminators中的符号
        if isinstance(result_ids, torch.Tensor):
            result_ids = result_ids.tolist()

        processed_ids = []
        for i in range(len(result_ids)):
            # 去除bos_token_id（如果它在开头的话）
            if tokenizer.bos_token_id is not None and result_ids[i] and result_ids[i][0] == tokenizer.bos_token_id:
                result_ids[i] = result_ids[i][1:]

            # 找到最早出现的terminators符号
            eos_positions = []
            for j, token in enumerate(result_ids[i]):
                # print(type(result_ids[i]))
                if token in terminators:
                    eos_positions.append(j)

            if len(eos_positions) != 0:
                result_ids[i] = result_ids[i][:min(eos_positions)]

            processed_ids.append(result_ids[i])

        # 解码
        result_text = tokenizer.batch_decode(
            processed_ids, 
            # skip_special_tokens=True,
        )

        # result_text = []
        # for ids in result_ids:
        #     # 找出最早出现的终止符位置
        #     eos_positions = []
        #     for terminator in terminators:
        #         positions = (ids == terminator).nonzero(as_tuple=True)[0]
        #         if len(positions) > 0:
        #             eos_positions.append(positions[0].item())

        #         # 如果找到终止符，截断导第一个终止符
        #         if eos_positions:
        #             eos_pos = min(eos_positions)
        #             ids = ids[:eos_pos]

        #         # 跳过起始特殊标记
        #         if tokenizer.bos_token_id is not None and ids[0] == tokenizer.bos_token_id:
        #             ids = ids[1:]

        #         # 解码为文本
        #         text = tokenizer.decode(ids, skip_special_tokens=True)
        #         result_text.append(text)
        ### <===
        return result_text

    def generate(
        self,
        input_ids=None,
        pixel_values=None,
        tgt_sizes=None,
        image_bound=None,
        attention_mask=None,
        tokenizer=None,
        vision_hidden_states=None,
        return_vision_hidden_states=False,
        stream=False,
        decode_text=False,
        **kwargs
    ):
        assert input_ids is not None
        assert len(input_ids) == len(pixel_values)

        model_inputs = {
            "input_ids": input_ids,
            "image_bound": image_bound,
        }

        if vision_hidden_states is None:
            model_inputs["pixel_values"] = pixel_values
            model_inputs['tgt_sizes'] = tgt_sizes
        else:
            model_inputs["vision_hidden_states"] = vision_hidden_states

        ### ===> TODO: 实现多模态大模型的 generation，注意不要计算模型参数的梯度。
        # 1. 获取模型视觉信号
        # 2. 实现 self._decode()，返回解码后的文本
        
        # 不计算梯度
        # print('-'*50)
        with torch.inference_mode():
            # 1. 获取模型视觉信号
            # if "position_ids" not in model_inputs:
            #     seq_length = input_ids.shape[1]
            #     model_inputs["position_ids"] = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)

            # 多模态嵌入
            # vllm_embedding, vision_hidden_states = self.get_vllm_embedding(model_inputs.to(self.device))
            vllm_embedding, vision_hidden_states = self.get_vllm_embedding(model_inputs)

            #  2. 实现 self._decode()，返回解码后的文本
            if stream:
                result = self._decode_stream(
                    inputs_embeds=vllm_embedding, 
                    tokenizer=tokenizer, 
                    attention_mask=attention_mask.to(self.device),
                    **kwargs)
            else:
                result = self._decode(
                    inputs_embeds=vllm_embedding, 
                    tokenizer=tokenizer,
                    attention_mask=attention_mask.to(self.device),
                    decode_text=decode_text,
                    **kwargs
                )
        ### <===
        if return_vision_hidden_states:
            return result, vision_hidden_states

        return result
