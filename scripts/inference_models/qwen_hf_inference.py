import os
import sys
from typing import List
import torch
from abc import ABC
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

torch.manual_seed(1234)

sys.path.append(os.path.join(os.getcwd(), "scripts", "inference_models"))

from base_inference_model import BaseInferenceModel


class QwenHfInference(BaseInferenceModel, ABC):
    def __init__(self, model_id="Qwen/Qwen2-VL-2B-Instruct"):
        super().__init__(model_id)

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype="auto"
        )
        self.model.to("cuda:0")
        self.is_open_source = True

    def get_template(self, prompt: str, image_path: List[str]):
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        prompt = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda:0")
        return inputs

    def get_response(self, prompt: str, image_path: List[str]):
        inputs = self.get_template(prompt=prompt, image_path=image_path)
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output
