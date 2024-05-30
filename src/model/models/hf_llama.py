from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch

from model.utils import LLMModel
from model.retrievers.prefetched_retrieve import PrefetchedDocumentRetriever
from data.loader import qa_prompt_with_instructions

device = "cuda" if torch.cuda.is_available() else "cpu"

class HfLLaMAModel(LLMModel):
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
        self.model = AutoModelForCausalLM.from_pretrained(  # torch_dtype=torch.bfloat16,
            self.hf_model_name, cache_dir=self.cache_dir, device_map="auto", load_in_8bit=load_in_8bit,
            low_cpu_mem_usage=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.retriever_type = config["Model.Retriever"]["type"]
        self.use_retriever = self.retriever_type.lower() != 'none'
        self._retriever = PrefetchedDocumentRetriever(config) if self.use_retriever else None
        self.top_k = int(config["Model.Retriever"]["retriever_top_k"]) if self.use_retriever else 0
        print('*********** Loaded Configurations *****************')
        li8b = '(8-bit quantized)' if load_in_8bit else '(non-quantized)'
        print(f'* Loaded model name: {self.hf_model_name} {li8b}')
        print(f'* Max tokens to generate: {self.max_tokens_to_generate}')
        print(f'* Loaded prompt provider: {qa_prompt_with_instructions.__name__.upper()}')
        print(f"* {'Open' if self.use_retriever else 'Closed'}-book retrieval mode")
        if self.use_retriever:
            print(f'* Open-book retrieval using {self.retriever_type} retriever with top \"{self.top_k}\" pre-fetched documents!')
        print('***************************************************')

    def _get_completion(self, prompt):
        input_ids = self.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.max_tokens_to_generate,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.000001,
            top_p=1,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        response = outputs[0][input_ids.shape[-1]:]
        generation_str = self.tokenizer.decode(response, skip_special_tokens=True)
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