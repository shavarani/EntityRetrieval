"""
This is a standalone script that uses pre-trained entity linking models to find the entities in the questions, and
 then uses a pre-built entity identifier to article to fetch the first k words in the fetched articles.

"""
import os
import re
import string
import argparse
import configparser
import pathlib
import jsonlines
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.abspath('../../model/entity_linking/'))

from data.loader import get_dataset
from model.entity_linking.spel_annotator import SpELAnnotate
from model.entity_linking.spel_vocab_to_wikipedia import SpELVocab2Wikipedia

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

def parse_args():
    _path = pathlib.Path(os.path.abspath(__file__)).parent.parent.parent / '..' / '.checkpoints'
    parser = argparse.ArgumentParser(description="Script to prefetch retrieval documents")
    parser.add_argument("--dataset",     type=str, help="Name of the dataset", default='FACTOIDQA')
    parser.add_argument("--split",       type=str, help="Split of the dataset", default="train")
    parser.add_argument("--type",        type=str, help="Type of the retriever", default="spel")
    parser.add_argument("--max_w",       type=int, help="Number of first words of the retrieved article for each question", default=100)
    parser.add_argument("--output_file", type=str, help="Output file name", default="el_output.jsonl")
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read_dict({
        'Dataset': {'name' : args.dataset, 'split': args.split},
        'Model.Retriever': {'type': args.type, 'max_w': args.max_w},
        'Experiment': {'checkpoint_path': _path, 'output_file': args.output_file}
    })
    return config

def instantiate_entity_linker(retriever_type):
    if retriever_type == "spel":
        linker = SpELAnnotate()
        lookup_index = SpELVocab2Wikipedia()
    else:
        raise ValueError(f"Invalid entity linking retriever type: {retriever_type}")
    return linker, lookup_index

class FetchEntityRetrievalDocuments:
    def __init__(self, config):
        super().__init__()
        self.retriever_type = config["Model.Retriever"]["type"]
        self.retriever_max_w = int(config["Model.Retriever"]["max_w"])
        self.checkpoint_path = config["Experiment"]["checkpoint_path"]
        self.linker, self.lookup_index = instantiate_entity_linker(self.retriever_type)

    def fetch_documents(self, question, answer_aliases):
        results = []
        considered_entities = set()
        for line_no, x in enumerate(self.linker.annotate(question)):
            if x['annotation'] in considered_entities:
                continue
            entity = x['annotation']
            considered_entities.add(entity)
            wikipedia_txt = self.lookup_index.get_wikipedia_article(x['annotation'])
            title = entity.replace("_", " ")
            passage = title + "\n" + " ".join(wikipedia_txt.split()[:self.retriever_max_w])
            results.append({'id': line_no,'rank': line_no + 1, 'title': title, 'text': passage, 'score': str(1.0), 'has_answer': text_has_answer(answer_aliases, passage)})
        return results

if __name__ == '__main__':
    cfg = parse_args()
    dataset = get_dataset(cfg)
    retriever = FetchEntityRetrievalDocuments(cfg)
    with jsonlines.open(cfg['Experiment']['output_file'], mode='w') as writer:
        for e in tqdm(dataset):
            writer.write({"question": e.question, "context": retriever.fetch_documents(e.question, e.answer_aliases)})