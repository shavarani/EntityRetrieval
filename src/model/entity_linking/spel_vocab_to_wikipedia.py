import os
import json
import torch
import pathlib
from tqdm import tqdm

from spel.src.spel.configuration import get_checkpoints_dir

class SpELVocab2Wikipedia:
    """
    How to use:
    w = SpELVocab2Wikipedia()
    wiki_text = w.get_wikipedia_article(wikipedia_entity_title_without_spaces)
    """
    def __init__(self):
        self.spel_vocab2wikipedia_lines, self.spel_vocab2wikipedia_address_to_load = self.load_articles()

    @staticmethod
    def load_articles():
        """
        Makes sure spel-vocab-to-wikipedia-articles.jsonl is locally downloaded and
        loads its keys into a dictionary along with their line number in the downloaded file.
        """
        print('Loading SpELVocab2Wikipedia jsonl file ...')
        file_name = 'spel-vocab-to-wikipedia-articles.jsonl'
        if not (get_checkpoints_dir() / file_name).exists():
            print(f'downloading {file_name} ...')
            torch.hub.download_url_to_file('https://vault.sfu.ca/index.php/s/xVteLdW57jsAHZx/download',
                                           str(get_checkpoints_dir() / file_name))
        newline_positions_in_file = []
        tmp_dict = dict()
        spel_vocab2wikipedia_lines = dict()
        starting_byte = 0
        with open(get_checkpoints_dir() / file_name, "rb") as fh:
            for line_id, line in tqdm(enumerate(fh)):
                obj = json.loads(line)
                k, _ = next(iter(obj.items()))
                tmp_dict[line_id] = k
                newline_positions_in_file.append(starting_byte)
                starting_byte = fh.tell()

        for obj_ind, obj in enumerate(newline_positions_in_file):
            spel_vocab2wikipedia_lines[tmp_dict[obj_ind]] = obj
        return spel_vocab2wikipedia_lines, get_checkpoints_dir() / file_name

    def get_wikipedia_article(self, entity):
        if entity not in self.spel_vocab2wikipedia_lines:
            return ''
        with open(self.spel_vocab2wikipedia_address_to_load, 'r') as bigFile:
            bigFile.seek(self.spel_vocab2wikipedia_lines[entity])
            json_line = bigFile.readline().strip()
            _, wikipedia_txt = next(iter(json.loads(json_line).items()))
        return " ".join(wikipedia_txt.split())