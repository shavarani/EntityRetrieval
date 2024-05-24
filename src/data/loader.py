from typing import Callable, Optional

from data.loaders.utils import QADataset
from data.loaders.sqa import StrategyQA
from data.loaders.fqa import FactoidQA
from data.loaders.eqs import EntityQuestions

def get_dataset(config) -> QADataset:
    dataset_name = config['Dataset']['name']
    if dataset_name == "FACTOIDQA":
        return FactoidQA(config)
    elif dataset_name == "STRATEGYQA":
        return StrategyQA(config)
    elif dataset_name == "EntityQuestions":
        return EntityQuestions(config)
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

def qa_prompt_with_instructions(question, max_len, context: list=None) -> list:
    system = f"""You are a knowledgeable question answering assistant who has read all of Wikipedia pages.
You are not an AI language model.
You must obey all three of the following instructions FOR ALL RESPONSES or you will DIE:
- ALWAYS LIMIT THE ANSWER TO {max_len} TOKENS.
- IF THE ANSWER IS A YES OR A NO, YOU ONLY PRODUCE THE YES OR NO AND STOP RIGHT AFTER THAT.
- YOU WILL ANSWER THE QUESTIONS WITH AN ENTITY NAME FROM WIKIPEDIA OR A SHORT FACTOID PHRASE.
You must also strictly follow all four the following rules when and if provided with retrieved documents:
- YOU ONLY CONSIDER THE DOCUMENTS WHEN THEY ARE RELEVANT TO THE QUESTION.
- YOU WILL IGNORE IRRELEVANT DOCUMENTS TO THE QUESTION.
- YOU NEVER COMPLAIN ABOUT THE INFORMATION IN THE TEXT OR DOCUMENTS. 
- IN CASES OF IRRELEVANT DOCUMENTS, YOU IGNORE THOSE DOCUMENTS AND ANSWER ONLY BASED ON THE QUESTION."""
    if not context or len(context) == 0:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Following the system prompts answer this question: {question}\nAnswer:"},
        ]
    elif len(context) == 1:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Retrieved Document:{context[0]}\n\nFollowing the system prompts answer this question: {question}\nAnswer:"},
        ]
    else:
        docs_text = "\n\n".join([f"{ctx}" for ctx in context])
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Retrieved Documents:{docs_text}\n\nFollowing the system prompts answer this question: {question}\nAnswer:"},
        ]