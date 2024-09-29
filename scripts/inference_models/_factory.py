from auto_model_inference import AutoModelForVisionLanguageInference

try:
    from llava_hf_inference import LLavaHfInference
except:
    print("Ignoring importing LLavaHfInference due to environment incompatiblity")
    pass

try:
    from qwen_hf_inference import QwenHfInference
except:
    print("Ignoring importing QwenHfInference due to environment incompatiblity")
    pass

# try:
from llava_base_inference import LLavaBaseInference
# except:
#     print("Ignoring importing LLavaBaseInference due to environment incompatiblity")
#     pass

try:
    from share_llava_base_inference import ShareLLavaInference
except:
    print("Ignoring importing ShareLLavaInference due to environment incompatiblity")
    pass
    

def inference_model_factory(model_id, is_quantization=False, is_llama_cpp=False, api_key=None):
    if "llava-hf" in model_id.lower():
        return LLavaHfInference(model_id, is_quantization=is_quantization)
    elif "share" in model_id.lower():
        try:
            return ShareLLavaInference(model_id)
        except:
            return None
    elif "llava" in model_id.lower():
        try:
            return LLavaBaseInference(model_id)
        except:
            return None
    elif "qwen" in model_id.lower():
        try:
            return QwenHfInference(model_id)
        except:
            return None
    else:
        try:
            return AutoModelForVisionLanguageInference(model_id)
        except:
            raise ValueError(f"Unknown model name: {model_id}")
