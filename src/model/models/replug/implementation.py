"""
Base code taken from https://github.com/IntelLabs/fastRAG/blob/main/examples/replug_parallel_reader.ipynb

You may also look at the original implementation in https://github.com/swj0419/REPLUG
"""
# pip install pydantic==1.10.14
# pip install llama-cpp-python
# pip install 'farm-haystack[all]'
from typing import Dict, Any
import torch

from haystack import Pipeline
from haystack import Document
from haystack.nodes.prompt import PromptNode
from haystack.nodes import PromptModel
from haystack.nodes.prompt.prompt_template import PromptTemplate
from haystack.nodes import AnswerParser
from haystack.nodes.ranker import SentenceTransformersRanker

from model.utils import LLMModel
from model.models.replug.utils import ReplugHFLocalInvocationLayer
from model.loader import get_retriever

def remove_template_params(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return kwargs
PromptTemplate.remove_template_params = remove_template_params

class RePLUG(LLMModel):
    def __init__(self, config, rerank_retrieved=False, reranker_top_k = 10):
        # super().__init__(config)
        self.config = config
        self.checkpoint_path = self.config["Experiment"]["checkpoint_path"]
        self.cache_dir = f"{self.checkpoint_path}/hf"
        self.retriever_top_k = int(config["Model.Retriever"]["retriever_top_k"])
        self.reranker_top_k = reranker_top_k
        load_in_8bit = config["Model"]["hf_llm_load_in_8bit"].lower() == 'true'
        self.retriever = get_retriever(config)
        assert self.retriever is not None, "RePLUG does not run in Closed-Book mode!"
        if rerank_retrieved:
            sbert_path = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            self.reranker = SentenceTransformersRanker(batch_size=32, model_name_or_path=sbert_path, top_k=1,
                                                       use_gpu=False, model_kwargs=dict(cache_dir= self.cache_dir))
        self.llm_model_name = self.config["Model"]["hf_model_name"]
        model_kwargs = dict(device_map = "auto", torch_dtype = torch.bfloat16, cache_dir= self.cache_dir,
                            load_in_8bit=load_in_8bit)
        LFQA = PromptTemplate(prompt=self.zero_shot_prompt_template, output_parser=AnswerParser())
        PrompterModel = PromptModel(
            model_name_or_path= self.llm_model_name, use_gpu= True, invocation_layer_class=ReplugHFLocalInvocationLayer,
            model_kwargs= dict(
                max_new_tokens=int(self.config["Model"]["hf_max_tokens_to_generate"]),
                model_kwargs= model_kwargs,
                generation_kwargs=dict(do_sample=True, max_length=int(self.config["Model"]["hf_max_tokens_to_generate"]))))
        self.prompter = PromptNode(model_name_or_path=PrompterModel, default_prompt_template=LFQA)
        self.llm = self.prompter.prompt_model

        self.pipe = Pipeline()
        self.pipe.add_node(component=self.retriever, name='Retriever', inputs=["Query"])
        if rerank_retrieved:
            self.pipe.add_node(component=self.reranker, name='Reranker', inputs=["Retriever"])
            self.pipe.add_node(component=self.prompter, name='Prompter', inputs=["Reranker"])
            self.params = {"Retriever": {"top_k": self.retriever_top_k}, "Reranker": {"top_k": self.reranker_top_k}}
        else:
            self.pipe.add_node(component=self.prompter, name='Prompter', inputs=["Retriever"])
            self.params = {"Retriever": {"top_k": self.retriever_top_k}}
            self.reranker = None


    def annotate(self, record, summarize=False, verbose=False):
        with torch.no_grad():
            answer_result = self.pipe.run(record.question, params=self.params)
        generated_answer = answer_result["answers"][0].answer
        generated_answer = generated_answer.split("\n")[0].strip()
        if verbose:
            print(f"---------------------------\n{record.question}\n---------------------------\nGenerated Answer: {generated_answer.strip()}\n---------------------------")
        if summarize:
            print("warning: summarize is not implemented for RePLUGModel!")
        if generated_answer and generated_answer[0] == "\"" and generated_answer[-1] == "\"":
            generated_answer = generated_answer[1:-1]
        if verbose:
            print(f"Expected Answer: {record.answer}\n---------------------------")
        record.predicted_answer = generated_answer.strip()

    @property
    def zero_shot_prompt_template(self):
        s = """Answer the Question to the best of your knowledge. 
Attend to the provided context if it contains useful information.
Do not mention the question or the provided context and just provide the answer.
Your answer can only be an entity name or a short phrase.

Context: ###REPLUG-DOC###

Question: {query}
Answer:"""
        return "<s> [INST] " + s + " [/INST]" if "mixtral" in self.llm_model_name else s

    @property
    def few_shot_prompt_template(self):
        s = """Answer the Question to the best of your knowledge. 
Attend to the provided context if it contains useful information.
Do not mention the question or the provided context and just provide the answer.
Your answer can only be an entity name or a short phrase.

Examples:
Context: "The Lamentable and Tragical History of Titus Andronicus," also called "Titus Andronicus\' Complaint," is a ballad from the 17th century about the fictional Roman general, Titus, and his revenge cycle with the Queen of the Goths. Events in the ballad take place near the end of the Roman Empire, and the narrative of the ballad parallels the plot of William Shakespeare\'s play Titus Andronicus.
Question: Shakespeare's "Titus Andronicus" is set during the latter days of which Empire?
Answer: Roman Empire

Context: Popeye the Sailor Man is a fictional cartoon character created by Elzie Crisler Segar. The character first appeared on January 17, 1929, in the daily King Features comic strip Thimble Theatre.
Question: Who created Popeye?
Answer: Elzie Segar

Context: The Rose Period (Spanish: Período rosa) comprises the works produced by Spanish painter Pablo Picasso between 1904 and 1906. It began when Picasso settled in Montmartre at the Bateau-Lavoir among bohemian poets and writers.
Question: What period preceded Picasso's "Rose Period"?
Answer: Blue Period

Context: ###REPLUG-DOC###

Question: {query}
Answer:"""
        return "<s> [INST] " + s + " [/INST]" if "mixtral" in self.llm_model_name else s