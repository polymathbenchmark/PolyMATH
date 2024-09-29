import os
import sys
import torch
from abc import ABC
from PIL import Image
try:
    from transformers import (
        AutoProcessor, LlavaForConditionalGeneration, 
        LlavaNextForConditionalGeneration, BitsAndBytesConfig
    )
except:
    pass

sys.path.append(os.path.join(os.getcwd(), "scripts", "inference_models"))

from base_inference_model import BaseInferenceModel


class LLavaHfInference(BaseInferenceModel, ABC):
    def __init__(self, model_id="llava-hf/llava-v1.6-34b-hf", is_quantization=False):
        super().__init__(model_id)
        self.processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path=model_id)
        
        if is_quantization:
            print("Running in quantized mode")
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        if "v1.6" in model_id:
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True, 
                quantization_config=quantization_config if is_quantization else None
            )
        else:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                quantization_config=quantization_config if is_quantization else None
            )
        self.model.generation_config.pad_token_id = self.processor.tokenizer.eos_token_id
        if not is_quantization:
            self.model.to("cuda:0")
        self.is_open_source = True

    def get_template(self, prompt, image_path):
        image = Image.open(image_path)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to("cuda:0", torch.float16)
        return inputs

    def get_response(self, prompt, image_path):
        inputs = self.get_template( prompt=prompt, image_path=image_path)
        output = self.model.generate(**inputs, max_new_tokens=600)
        return self.processor.decode(output[0], skip_special_tokens=True)
