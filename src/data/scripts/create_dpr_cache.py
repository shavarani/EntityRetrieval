import os
import re
import string
import argparse
import configparser
import pathlib
import jsonlines
from transformers import DPRQuestionEncoderTokenizer, DPRQuestionEncoder
from datasets import load_dataset
from tqdm import tqdm

from data.loader import get_dataset


path_ =  pathlib.Path(os.path.abspath(__file__)).parent.parent.parent / '..' / '.checkpoints' / 'hf'

def parse_args():
    parser = argparse.ArgumentParser(description="Script to create DPR context")
    parser.add_argument("--dataset", type=str, help="Name of the dataset", default='NQ')
    parser.add_argument("--split", type=str, help="Split of the dataset", default="train")
    parser.add_argument("--index_name", type=str, help="Index name", default="compressed", choices=["compressed", "exact"])
    parser.add_argument("--question_model_type", type=str, help="Question model type", default="single-nq", choices=["single-nq", "multiset"])
    parser.add_argument("--retriever_k", type=int, help="Number of documents retrived for each question", default=100)
    parser.add_argument("--output_file", type=str, help="Output file name", default="output.jsonl")
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read_dict({
        'Dataset': {'name' : args.dataset, 'split': args.split},
        'Experiment': {'checkpoint_path': path_,
                       'question_model_type': f'facebook/dpr-question_encoder-{args.question_model_type}-base',
                       'index_name': args.index_name, 'retriever_k': args.retriever_k, 'output_file': args.output_file}
    })
    return config

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def text_has_answer(answers, text) -> bool:
    """
    Taken from https://github.com/AI21Labs/in-context-ralm/blob/main/eval_qa.py
    """
    if isinstance(answers, str):
        answers = [answers]
    text = normalize_answer(text)
    for single_answer in answers:
        single_answer = normalize_answer(single_answer)
        if single_answer in text:
            return True
    return False

class DPR:
    def __init__(self, config):
        self.cache_dir = config["Experiment"]["checkpoint_path"]
        self.index_name = config["Experiment"]["index_name"]
        self.question_model_type = config["Experiment"]["question_model_type"]
        self.retriever_k = int(config["Experiment"]["retriever_k"])
        self.wiki = load_dataset("wiki_dpr", with_embeddings=True, with_index=True, index_name=self.index_name,
                                 cache_dir=self.cache_dir, trust_remote_code=True)['train']
        self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(self.question_model_type)
        self.question_encoder = DPRQuestionEncoder.from_pretrained(self.question_model_type)

    def get_context(self, question, answer_aliases=None):
        if answer_aliases is None:
            answer_aliases = []
        question_emb = self.question_encoder(**self.question_tokenizer(question, return_tensors="pt"))[0].detach().numpy()
        passages_scores, passages = self.wiki.get_nearest_examples("embeddings", question_emb, k=self.retriever_k)
        context = []
        for p_id, (passage_score, passage_id, passage_title, passage) in enumerate(
                zip(passages_scores, passages['id'], passages['title'], passages['text'])):
            context.append({'id': passage_id,'rank': p_id, 'title': passage_title, 'text': passage,
                            'score': str(passage_score), 'has_answer': text_has_answer(answer_aliases, passage)})
        return context


if __name__ == "__main__":
    cfg = parse_args()
    dataset = get_dataset(cfg)
    dpr_model = DPR(cfg)
    with jsonlines.open(cfg['Experiment']['output_file'], mode='w') as writer:
        for e in tqdm(dataset):
            writer.write({"question": e.question, "context": dpr_model.get_context(e.question, e.answer_aliases)})
