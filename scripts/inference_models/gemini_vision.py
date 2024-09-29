import os
import logging
from abc import ABC
from PIL import Image
from typing import List
import google.generativeai as genai
from dotenv import load_dotenv

from .base_inference_model import BaseInferenceModel

load_dotenv("../.env")
API_KEY = os.environ.get("GEMINI_API_KEY")
do_gemini = True
genai.configure(api_key=API_KEY)


class GeminiVisionInference(BaseInferenceModel, ABC):
    def __init__(self, sleep_time: int = 20, model_id: str = "gemini-pro-vision", pre_image_prompt: str = None,
                 post_image_prompt: str = None, pre_context: str = None, post_context: str = None, pre_q: str = None,
                 post_q: str = None):
        super().__init__(model_id)
        genai.configure(api_key="AIzaSyBlvY9cc4lxmiJvDegLU-DxlSYvXonnWA4")
        self.pre_image_prompt = pre_image_prompt or """Provide descriptions for the following images."""
        self.post_image_prompt = post_image_prompt or """Output descriptions as a numbered list."""
        self.pre_context = pre_context or """This is the context for a mathematical problem."""
        self.post_context = post_context or """Based on the information above, answer the following question."""
        self.pre_q = pre_q or """Question instruction: """
        self.post_q = post_q or """Output only the correct answer option."""

        self.sleep_time = sleep_time
        self.model = genai.GenerativeModel(model_id)

    @staticmethod
    def print_gemini_models():
        for models_ in genai.list_models():
            print(models_.name, models_.description)

    def get_template(
            self,
            prompt: str,
            img_path: str
            ):
        img_path = Image.open(img_path)
        return [prompt, img_path]

    def get_response(self,
                     model_input: List
                     ):
        response = self.model.generate_content(model_input, stream=True)
        response.resolve()
        try:
            response_text = response.candidates[0].content.parts[0].text
        except Exception as e:
            logging.debug(f"GeminiApiException: {e}")
            response_text = response.prompt_feedback
        return response_text
