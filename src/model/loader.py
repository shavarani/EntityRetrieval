from typing import Union
from haystack.nodes.retriever import BaseRetriever

from model.utils import LLMModel
from model.models.hf_llm import HfLLMModel
from model.models.replug.implementation import RePLUG
from model.models.openai_llm import GPTModel
from model.models.hf_llama import HfLLaMAModel

from model.retrievers.prefetched_retrieve import PrefetchedDocumentRetriever
from model.retrievers.fast_prefetched_retrieve import FastPrefetchedDocumentRetriever

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

def get_retriever(config) -> Union[BaseRetriever, None]:
    retriever_type = config["Model.Retriever"]["type"]
    use_retriever = retriever_type.lower() != 'none'
    retriever_top_k = int(config["Model.Retriever"]["retriever_top_k"])
    retriever_load_in_memory = config['Model.Retriever']['load_in_memory'].lower() == 'true'
    if retriever_load_in_memory and use_retriever:
        return FastPrefetchedDocumentRetriever(config, topk=retriever_top_k)
    elif use_retriever:
        return PrefetchedDocumentRetriever(config)
    else:
        return None
