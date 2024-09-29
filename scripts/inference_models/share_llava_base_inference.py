import os
import sys
import torch
from abc import ABC
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

sys.path.append(os.path.join(os.getcwd(), "scripts", "inference_models"))
from share4v.conversation import conv_templates, SeparatorStyle
from share4v.model.language_model.share4v_llama import Share4VLlamaForCausalLM
from share4v.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN
)
from share4v.mm_utils import tokenizer_image_token
from share4v.utils import disable_torch_init

from base_inference_model import BaseInferenceModel


class ShareLLavaInference(BaseInferenceModel, ABC):
    def __init__(self, model_id="Lin-Chen/ShareGPT4V-7B"):
        super().__init__(model_id)
        self.processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path=model_id)
        self.model = Share4VLlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        disable_torch_init()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        self.is_open_source = True

        mm_use_im_start_end = getattr(self.model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(self.model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            self.tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        self.model.resize_token_embeddings(len(self.tokenizer))
        self.vision_tower = self.model.get_vision_tower()

        if not self.vision_tower.is_loaded:
            self.vision_tower.load_model()

        self.model.to("cuda:0")
        self.vision_tower.to(device="cuda:0", dtype=torch.float16)
        self.processor = self.vision_tower.image_processor

    def get_template(self, prompt, image_path):
        if self.model.config.mm_use_im_start_end:
            prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt

        conv = conv_templates["share4v_v1"].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
            0).to("cuda:0")

        image = Image.open(image_path)
        image_tensor = self.processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        return input_ids, image_tensor, conv

    def get_response(self, prompt, image_path):
        input_ids, image_tensor, conv = self.get_template(prompt=prompt, image_path=image_path)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).to(dtype=self.model.dtype).to("cuda:0"),
                do_sample=True,
                max_new_tokens=1600,
                use_cache=True,
                temperature=0.7,
                top_p=None,
                num_beams=1,
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        output = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        output = output.strip()
        if output.endswith(stop_str):
            output = output[:-len(stop_str)]
        output = output.strip()
        return output
