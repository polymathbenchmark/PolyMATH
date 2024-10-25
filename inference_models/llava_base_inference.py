import os
import sys
import torch
from abc import ABC
from PIL import Image
from transformers import AutoTokenizer

sys.path.append(os.path.join(os.getcwd(), "scripts", "inference_models"))

from llava.model import LlavaMistralForCausalLM, LlavaMptForCausalLM, LlavaLlamaForCausalLM
from llava.conversation import conv_templates
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images
from llava.constants import (
    DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
)

from base_inference_model import BaseInferenceModel


class LLavaBaseInference(BaseInferenceModel, ABC):
    def __init__(self, model_id="renjiepi/G-LLaVA-7B"):
        super().__init__(model_id)

        if 'mpt' in model_id.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            self.model = LlavaMptForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True)
        elif 'mistral' in model_id.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = LlavaMistralForCausalLM.from_pretrained(
                model_id,
                low_cpu_mem_usage=True
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
            self.model = LlavaLlamaForCausalLM.from_pretrained(
                model_id,
                low_cpu_mem_usage=True
            )
        disable_torch_init()
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

        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
            0).to("cuda:0")

        image = Image.open(image_path).convert('RGB')
        image_tensor = process_images([image], self.processor, self.model.config)[0]
        return input_ids, image, image_tensor, conv

    def get_response(self, prompt, image_path):
        input_ids, image, image_tensor, conv = self.get_template(prompt=prompt, image_path=image_path)

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

        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return output
