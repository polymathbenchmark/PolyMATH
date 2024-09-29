# Reference: https://github.com/abetlen/llama-cpp-python

from llama_cpp import Llama, ChatCompletionMessage, ChatCompletionRequestSystemMessage
from llama_cpp.llama_chat_format import (
    Llava16ChatHandler,
    MoondreamChatHandler,
    NanoLlavaChatHandler,
    Llama3VisionAlphaChatHandler,
    ObsidianChatHandler
)

from .base_inference_model import BaseInferenceModel


class LlamaCppVisionInference(BaseInferenceModel):
    def __init__(self, model_id, context_length):
        super().__init__(model_id)
        if "llava" in model_id:
            chat_handler = Llava16ChatHandler(clip_model_path=model_id)
        elif "moondream" in model_id:
            chat_handler = MoondreamChatHandler(clip_model_path=model_id)
        elif "nanollava" in model_id:
            chat_handler = NanoLlavaChatHandler(clip_model_path=model_id)
        elif "llama-3" in model_id:
            chat_handler = Llama3VisionAlphaChatHandler(clip_model_path=model_id)
        elif "llava" in model_id:
            chat_handler = ObsidianChatHandler(clip_model_path=model_id)

        self.llm = Llama(
            model_path="./path/to/llava/llama-model.gguf",
            chat_handler=chat_handler,
            n_ctx=context_length,  # n_ctx should be increased to accommodate the image embedding
        )

    def get_template(self, image_path):
        """
        This method will transform the sample into the required template that can be consumed by
        the LLM

        :param image_path:
        :param self:
        :return:
        """
        inputs = "DummyMessage"
        return inputs

    def get_response(self, image_path):
        inputs = self.get_template(image_path=image_path)
        output = self.llm.create_chat_completion(
            messages=inputs
        )
        return output
