import os
import sys
import torch
from abc import ABC
from typing import List
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

torch.manual_seed(1234)

sys.path.append(os.path.join(os.getcwd(), "scripts", "inference_models"))

from base_inference_model import BaseInferenceModel


class AutoModelForVisionLanguageInference(BaseInferenceModel, ABC):
    def __init__(self, model_id="Lin-Chen/ShareGPT4V-7B"):
        super().__init__(model_id)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", trust_remote_code=True).eval()

        self.model.to("cuda:0")
        self.is_open_source = True

    def get_template(self, prompt: str, image_path: List[str]):
        template = self.tokenizer.from_list_format([
            {'image': image_path},
            {'text': prompt},
        ])
        inputs = self.tokenizer(template, return_tensors='pt')
        return inputs

    def get_response(self, prompt: str, image_path: List[str]):
        # Specify hyperparameters for generation (No need to do this if you are using transformers>=4.32.0)
        if transformers.__version__ < 4.32:
            self.model.generation_config = GenerationConfig.from_pretrained(self.model_id, trust_remote_code=True)

        inputs = self.get_template(prompt=prompt, image_path=image_path).to(self.model.device)
        output = self.model.generate(**inputs)
        output = self.tokenizer.decode(output.cpu()[0], skip_special_tokens=False)
        return output
