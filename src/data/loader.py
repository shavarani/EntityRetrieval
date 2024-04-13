from typing import Callable, Optional

from data.loaders.utils import QADataset
from data.loaders.sqa import StrategyQA
from data.loaders.fqa import FactoidQA

def get_dataset(config) -> QADataset:
    dataset_name = config['Dataset']['name']
    if dataset_name == "FACTOIDQA":
        return FactoidQA(config)
    elif dataset_name == "STRATEGYQA":
        return StrategyQA(config)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def ralm_qa_prompt(question, context: list=None) -> str:
    if not context or len(context) == 0:
        ex_prompt = f"Answer these questions:\nQ: {question}\nA:"
    elif len(context) == 1:
        ctx = context[0]
        ex_prompt = f"{ctx}\n\nBased on this text, answer these questions:\nQ: {question}\nA:"
    else:
        docs_text = "\n\n".join([f"{ctx}" for ctx in context])
        ex_prompt = f"{docs_text}\n\nBased on these texts, answer these questions:\nQ: {question}\nA:"
    return ex_prompt

def get_prompt_provider(config) -> Callable[[str, Optional[str]], str]:
    """
      Each function receives a question and a context and returns a prompt template filled with th question and optionally context.
    """
    return ralm_qa_prompt
