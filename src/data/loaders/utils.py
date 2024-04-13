import os
import json
import requests
from typing import Optional, Iterable
from dataclasses import dataclass
from enum import Enum
from tqdm import tqdm

def download_public_file(file_url, dest_path):
    """Downloads a publicly accessible file"""
    if os.path.exists(dest_path):
        return
    resp = requests.get(file_url, stream=True)

    total_length = int(resp.headers.get("Content-Length", 0))
    dl_progress = tqdm(total=total_length, unit='B', unit_scale=True, desc=os.path.basename(dest_path))

    with open(dest_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            dl_progress.update(len(chunk))

    dl_progress.close()

class QADataset(Iterable):
    """Used to unify the different dataset loader implementations."""
    @staticmethod
    def normalize_question(question):
        if not question.endswith("?"):
            question = question + "?"
        return question[0].lower() + question[1:]

@dataclass
class QARecord:
    question: str
    answer: str
    dataset: str
    split: str
    answer_aliases: Optional[list] = None
    entity_annotations: Optional[list] = None
    answer_entity_name: Optional[str] = None
    predicted_answer: Optional[str] = None
    extracted_entity: Optional[str] = None

    def to_json(self):
        return {
            "question": self.question,
            "answer": self.answer,
            "dataset": self.dataset,
            "split": self.split,
            "answer_aliases": self.answer_aliases,
            "entity_annotations": self.entity_annotations,
            "answer_entity_name": self.answer_entity_name,
            "predicted_answer": self.predicted_answer,
            "extracted_entity": self.extracted_entity
        }

    def __str__(self) -> str:
        return json.dumps(self.to_json())

class DatasetSplit(Enum):
    TRAIN = 'train'
    DEV = 'dev'
    TEST = 'test'

    @staticmethod
    def from_str(s: str):
        if s.lower() == 'train':
            return DatasetSplit.TRAIN
        elif s.lower() in ['dev', 'val', 'validation']:
            return DatasetSplit.DEV
        elif s.lower() == 'test':
            return DatasetSplit.TEST
        else:
            raise ValueError(f"Invalid split: {str}")