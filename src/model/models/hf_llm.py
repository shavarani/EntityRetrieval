import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer

from model.utils import LLMModel
from model.retrievers.prefetched_retrieve import PrefetchedDocumentRetriever
from data.loader import get_prompt_provider

device = "cuda" if torch.cuda.is_available() else "cpu"


class HfLLMModel(LLMModel):
    def __init__(self, config):
        self.config = config
        self.checkpoint_path = self.config["Experiment"]["checkpoint_path"]
        self.cache_dir = f"{self.checkpoint_path}/hf"
        self.hf_model_name = self.config["Model"]["hf_model_name"]

        model_config = AutoConfig.from_pretrained(self.hf_model_name)
        self.model_max_length = model_config.n_positions if hasattr(model_config, "n_positions") \
            else model_config.max_position_embeddings
        self.max_tokens_to_generate=int(self.config["Model"]["hf_max_tokens_to_generate"])
        load_in_8bit = config["Model"]["hf_llm_load_in_8bit"].lower() == 'true'
        self.model = AutoModelForCausalLM.from_pretrained(self.hf_model_name, cache_dir=self.cache_dir, load_in_8bit=load_in_8bit,
                                                          device_map="auto", low_cpu_mem_usage=True).eval()
        if "llama" in self.hf_model_name:
            self.tokenizer = LlamaTokenizer.from_pretrained(self.hf_model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)

        self.prompter = get_prompt_provider(config)
        self.use_retriever = config["Model.Retriever"]["use_retriever"].lower() == 'true'
        self._retriever = PrefetchedDocumentRetriever(config) if self.use_retriever else None
        self.top_k = int(config["Model.Retriever"]["retriever_top_k"]) if self.use_retriever else 0
        print('*********** Loaded Configurations *****************')
        li8b = '(8-bit quantized)' if load_in_8bit else '(non-quantized)'
        print(f'* Loaded model name: {self.hf_model_name} {li8b}')
        print(f'* Max tokens to generate: {self.max_tokens_to_generate}')
        print(f'* Loaded prompt provider: {self.prompter.__name__.upper()}')
        print(f"* {'Open' if self.use_retriever else 'Closed'}-book retrieval mode")
        if self.use_retriever:
            print(f'* Open-book retrieval using top \"{self.top_k}\" DPR cached/fetched documents!')
        print('***************************************************')

    def _get_completion(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        if input_ids.shape[-1] > self.model_max_length - self.max_tokens_to_generate:
            input_ids = input_ids[..., -(self.model_max_length - self.max_tokens_to_generate):]
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids, max_new_tokens=self.max_tokens_to_generate, pad_token_id=self.model.config.eos_token_id)
        generation_str = self.tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)
        generation_str = generation_str[len(prompt):]
        return generation_str.split("\n")[0].strip()

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
            prompt = self.prompter(record.question, context=[f"{x[1]}\n\n{x[0]}" for x in context_container])
        else:
            extracted_context, record.extracted_entity = None, None
            prompt = self.prompter(record.question, context=None)
        generated_answer = self._get_completion(prompt)
        if verbose:
            print(f"---------------------------\n{prompt}\n---------------------------\nGenerated Answer: {generated_answer}\n---------------------------")
        if generated_answer and generated_answer[0] in ["\"", "\'"] and generated_answer[-1] in ["\"", "\'"]:
            generated_answer = generated_answer[1:-1]
        if verbose:
            print(f"Expected Answer: {record.answer}\n---------------------------")
        record.predicted_answer = generated_answer