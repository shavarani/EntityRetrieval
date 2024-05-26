import os
import openai # pip install openai==0.27.6
import backoff
import logging

from model.utils import LLMModel
from model.retrievers.prefetched_retrieve import PrefetchedDocumentRetriever
from data.loader import qa_prompt_with_instructions

GPT_MODEL_NAME = "gpt-4-0613"
# GPT_MODEL_NAME= "gpt-3.5-turbo-0125"

openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    print(f"WARNING! you need to set environment variable OPENAI_API_KEY to your personal key before running this code,"
          f"\notherwise you may be locked out by openai!")
else:
    print(f"OpenAI API Key retreived from environment variables: {openai.api_key}")

@backoff.on_exception(
    backoff.expo,
    openai.error.RateLimitError,
    max_time=240
)
def chat_completion(prompt, model_name=GPT_MODEL_NAME):
    return openai.ChatCompletion.create(
        model=model_name,
        temperature=0,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        messages=prompt,
        # max_tokens=4097
    )

class GPTModel(LLMModel):
    def __init__(self, config):
        self.config = config
        self.checkpoint_path = self.config["Experiment"]["checkpoint_path"]
        self.use_retriever = config["Model.Retriever"]["use_retriever"].lower() == 'true'
        self._retriever = PrefetchedDocumentRetriever(config) if self.use_retriever else None
        self.top_k = int(config["Model.Retriever"]["retriever_top_k"]) if self.use_retriever else 0
        self.max_tokens_to_generate=int(self.config["Model"]["hf_max_tokens_to_generate"])

    def _get_completion(self, prompt):
        try:
            r = chat_completion(prompt, GPT_MODEL_NAME)
        except openai.error.APIError:
            return ""
        annotated = r.choices[0].message['content'].strip()
        if r.choices[0].finish_reason == "length":
            logging.warning("Potentially longer than maximum token, request returned with finish_reason == length")
        return annotated.split("\n")[0].strip()

    def get_context(self, record):
        if self.use_retriever:
            res = self._retriever.retrieve(record.question, top_k=self.top_k)
            return [(r.content.strip(), r.meta["title"]) for r in res] if res else None
        else:
            return None

    def annotate(self, record, summarize=False, verbose=False):
        if summarize:
            print("warning: summarize is not implemented for HfLLMModel!")
        context_container = self.get_context(record)
        if context_container:
            record.extracted_entity = ";".join([x[1].replace(" ", "_") for x in context_container])
            prompt = qa_prompt_with_instructions(record.question, max_len=self.max_tokens_to_generate, context=[f"{x[1]}\n\n{x[0]}" for x in context_container])
        else:
            extracted_context, record.extracted_entity = None, None
            prompt = qa_prompt_with_instructions(record.question, max_len=self.max_tokens_to_generate, context=None)
        generated_answer = self._get_completion(prompt)
        if verbose:
            print(f"---------------------------\n{prompt}\n---------------------------\nGenerated Answer: {generated_answer}\n---------------------------")
        if generated_answer and generated_answer[0] in ["\"", "\'"] and generated_answer[-1] in ["\"", "\'"]:
            generated_answer = generated_answer[1:-1]
        if verbose:
            print(f"Expected Answer: {record.answer}\n---------------------------")
        record.predicted_answer = generated_answer