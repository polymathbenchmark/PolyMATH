from .gemini_vision import GeminiVisionInference
from .gpt4_vision import GPT4VisionInference
from .llama_cpp import LlamaCppVisionInference
from .llava_hf_inference import LLavaHfInference

try:
    from .qwen_hf_inference import QwenHfInference
except:
    pass

try:
    from .llava_base_inference import LLavaBaseInference
except:
    pass

try:
    from .share_llava_base_inference import ShareLLavaInference
except:
    pass

from .base_inference_model import BaseInferenceModel as InferenceModel

from ._factory import inference_model_factory
