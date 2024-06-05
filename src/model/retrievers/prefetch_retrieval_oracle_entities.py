"""
This is a standalone script that uses the entity annotations of the questions that come with the dataset and the created
  wikipedia identifier to content using model.retrievers.wikipedia.get_content.py script to create oracle retrieval
   documents.
"""
import os
import re
import string
import argparse
import configparser
import pathlib
import jsonlines
import json
from tqdm import tqdm


from data.loader import get_dataset

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
    parser.add_argument("--dataset",            type=str, help="Name of the dataset", default='FACTOIDQA')
    parser.add_argument("--split",              type=str, help="Split of the dataset", default="train")
    parser.add_argument("--wikipedia_articles", type=str, help="Path to the created output of model.retrievers.wikipedia.get_content.py")
    parser.add_argument("--max_w",              type=int, help="Number of first words of the retrieved article for each question", default=100)
    parser.add_argument("--output_file",        type=str, help="Output file name", default="oracle_output.jsonl")
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read_dict({
        'Dataset': {'name' : args.dataset, 'split': args.split},
        'Model.Retriever': {'wikipedia_articles_path': args.wikipedia_articles, 'max_w': args.max_w},
        'Experiment': {'checkpoint_path': _path, 'output_file': args.output_file}
    })
    return config


class FetchOracleEntityRetrievalDocuments:
    def __init__(self, config):
        super().__init__()
        self.wikipedia_articles_path = config["Model.Retriever"]["wikipedia_articles_path"]
        self.wikiepdia_article_lines = self.scan_jsonl_file(self.wikipedia_articles_path)
        self.retriever_max_w = int(config["Model.Retriever"]["max_w"])
        self.checkpoint_path = config["Experiment"]["checkpoint_path"]
        self.missing_entities = set()

    @staticmethod
    def scan_jsonl_file(file_name):
        newline_positions_in_file = []
        tmp_dict = dict()
        wikipedia_lines = dict()
        starting_byte = 0
        with open(file_name, "rb") as fh:
            for line_id, line in tqdm(enumerate(fh)):
                obj = json.loads(line)
                k, _ = next(iter(obj.items()))
                tmp_dict[line_id] = k.replace(' ', '_')
                newline_positions_in_file.append(starting_byte)
                starting_byte = fh.tell()

        for obj_ind, obj in enumerate(newline_positions_in_file):
            wikipedia_lines[tmp_dict[obj_ind]] = obj
        return wikipedia_lines

    def get_wikipedia_article(self, entity):
        if entity not in self.wikiepdia_article_lines:
            return ''
        with open(self.wikipedia_articles_path, 'r') as bigFile:
            bigFile.seek(self.wikiepdia_article_lines[entity])
            json_line = bigFile.readline().strip()
            _, wikipedia_txt = next(iter(json.loads(json_line).items()))
        return " ".join(wikipedia_txt.split())

    def fetch_documents(self, record):
        results = []
        considered_entities = set()
        entity_annotations = record.entity_annotations
        answer_aliases = record.answer_aliases
        for line_no, entity in enumerate(entity_annotations):
            if entity in considered_entities:
                continue
            considered_entities.add(entity)
            if entity not in self.wikiepdia_article_lines and entity not in self.missing_entities:
                print(f'{entity} not in the fetched articles, You may need to refine your wikipedia_articles file to contain this entity.')
                self.missing_entities.add(entity)
            wikipedia_txt = self.get_wikipedia_article(entity)
            title = entity.replace("_", " ")
            passage = title + "\n" + " ".join(wikipedia_txt.split()[:self.retriever_max_w])
            results.append({'id': line_no,'rank': line_no + 1, 'title': title, 'text': passage, 'score': str(1.0), 'has_answer': text_has_answer(answer_aliases, passage)})
        return results

if __name__ == '__main__':
    cfg = parse_args()
    dataset = get_dataset(cfg)
    retriever = FetchOracleEntityRetrievalDocuments(cfg)
    with jsonlines.open(cfg['Experiment']['output_file'], mode='w') as writer:
        for e in tqdm(dataset):
            writer.write({"question": e.question, "context": retriever.fetch_documents(e)})