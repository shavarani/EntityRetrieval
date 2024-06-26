from model.utils import LLMModel
from model.models.hf_llm import HfLLMModel
from model.models.replug.implementation import RePLUG
from model.models.openai_llm import GPTModel
from model.models.hf_llama import HfLLaMAModel

def get_llm(config) -> LLMModel:
    model_name = config['Model']['name']
    if model_name == "HFLLM":
        return HfLLMModel(config)
    elif model_name == "RePLUG":
        return RePLUG(config)
    elif model_name == "OpenAI":
        return GPTModel(config)
    elif model_name == "HFLLAMA":
        return HfLLaMAModel(config)
    else:
        raise ValueError(f"Unknown model: {model_name}")