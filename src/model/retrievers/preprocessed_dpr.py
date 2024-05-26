"""
The main objective of implementation of this class is to cope with the hardware complexities of loading a DPR retriever
 trained over the entire Wikipedia while experimenting on Question Answering experiments.
For this we have abstracted the retrieval process by caching the top-k highly ranked documents in huggingface's
 wiki_dpr data (https://huggingface.co/datasets/wiki_dpr) for each question in all of our supported QA datasets.
wiki_dpr contains 21_015_300 wikipedia articles from the dump of Dec. 20, 2018, chunked and split into non-overlapping
 text blocks of size 100 words. The pre-created dataset can come with two different FAISS indexes one 'compressed' (less
  accurate but lighter) and one 'exact' (more accurate but more memory demanding). We have run our dumping script
   in src/data/scripts/create_dpr_context_all.sh using both index types and have stored the created dump URLS in
    PREPROCESSED_URLS. You can create the dumps for newly implemented datasets using the same script.
"""
from typing import List, Optional, Union, Dict
import json
from zipfile import ZipFile
from tqdm import tqdm
from haystack import Document
from haystack.nodes.retriever import BaseRetriever
from haystack.document_stores import BaseDocumentStore
from haystack.schema import FilterType
from data.loaders.utils import download_public_file, DatasetSplit

PREPROCESSED_URLS = {
    "FACTOIDQA_exact_single-nq_100.zip":       "ETyE2KbMjsZJgEUDN-fFagQBraZoazON2tumbrBkYAfoKg?e=fd0Q1i",
    "FACTOIDQA_oracle_wiki_first_100_words.zip":"EaFkI0ecTM1As54ocMo5BFEBLBRsa7k1iWdQc3bpcrgEKA?e=afVrfM",
    "FACTOIDQA_spel_wiki_first_100_words.zip": "EaMFthYtZYtDn4GovmztYRcBDzBS3gac5ZqC4GlfTyPlpw?e=exwjKi",
    "STRATEGYQA_exact_single-nq_100.zip":      "Ecp8nr50lUVBrGCX9xR550oBY8T9VdJWo19CoZgwSMDasA?e=gnw8Tg",
    "STRATEGYQA_spel_wiki_first_100_words.zip":"EdiDKQSjTY5EkPD3LqzJD8UBnLF8Fn4_AGl3Xru9gZWIeA?e=zXVnrX",
}

class DPRContext:
    def __init__(self):
        self.id = None
        self.rank = None
        self.title = None
        self.text = None
        self.score = None
        self.has_answer = None

    @staticmethod
    def convert(record):
        self = DPRContext()
        self.id = record['id']
        self.rank = record['rank']
        self.title = record['title']
        self.text = record['text']
        self.score = float(record['score'])
        self.has_answer = record['has_answer']
        return self

    def __str__(self):
        return json.dumps({'id': self.id, 'rank': self.rank, 'title': self.title, 'text': self.text,
                           'score': str(self.score), 'has_answer': self.has_answer})

    @property
    def retriever_document(self):
        return Document(id=self.id, content=self.text, meta={'rank': self.rank, 'title': self.title,
                                                             'score': str(self.score), 'has_answer': self.has_answer})

class PreprocessedDPRRetriever(BaseRetriever):
    """
    How to test this retriever:
        import configparser
        config = configparser.ConfigParser()
        config.read_dict({
            'Dataset': {'name' : 'STRATEGYQA', 'split': 'dev'},
            'Experiment': {'checkpoint_path': 'src/entity_linking/.checkpoints/'},
            'Model.Retriever': {'dpr_index_type': 'exact', 'dpr_question_model': 'single-nq', 'prefetched_k_size': '100'}
        })
        ctxt = PreprocessedDPRRetriever(config).fetch_dpr_context(question)
    """
    def __init__(self, config):
        super().__init__()
        dataset_name = config['Dataset']['name']
        if dataset_name in ["NQ", "NQSimplified", "NaturalQuestions"]:
            dataset_name = "NQ"
        if dataset_name in ["FSQ", "FrequentSimpleQuestions"]:
            dataset_name = "FSQ"
        index_type = config["Model.Retriever"]["dpr_index_type"] # compressed or exact
        question_model_name = config["Model.Retriever"]["dpr_question_model"] # single-nq or multiset or in_context_ralm
        k_size = int(config["Model.Retriever"]["prefetched_k_size"])
        self.preprocessed_file_name = f"{dataset_name}_{index_type}_{question_model_name}_{k_size}.zip"
        if question_model_name == "in_context_ralm" and dataset_name in ["TRIVIAQA", "NQ"] and config["Dataset"]["split"] == 'dev':
            self.preprocessed_file_name = f"{dataset_name}_dev_only_in_context_ralm_20.zip"
        elif question_model_name == "in_context_ralm":
            raise ValueError("InContext RALM preprocessed data unavailable for DPR configuration setting ("
                             f"{index_type}/{question_model_name}/{k_size}/{dataset_name})")
        elif question_model_name == "spel_wiki_first_100_words":
            self.preprocessed_file_name = f"{dataset_name}_spel_wiki_first_100_words.zip"
        elif question_model_name == "oracle_wiki_first_100_words":
            self.preprocessed_file_name = f"{dataset_name}_oracle_wiki_first_100_words.zip"
        if self.preprocessed_file_name in PREPROCESSED_URLS:
            _url = PREPROCESSED_URLS[self.preprocessed_file_name]
        else:
            raise ValueError(f"Invalid preprocessed DPR configuration setting ("
                             f"{index_type}/{question_model_name}/{k_size}) for {dataset_name}")
        self.dataset_url = f"https://1sfu-my.sharepoint.com/:u:/g/personal/sshavara_sfu_ca/{_url}&download=1"
        self.checkpoint_path = config["Experiment"]["checkpoint_path"]
        download_public_file(self.dataset_url, f"{self.checkpoint_path}/{self.preprocessed_file_name}")
        self.split = DatasetSplit.from_str(config["Dataset"]["split"])
        self.dataset_name = dataset_name
        self.lookup_lines = self.scan_data_file()

    def scan_data_file(self):
        """
        loads the questions along with their line number in the downloaded file into a dictionary.
        """
        print(f'Scanning/Indexing the cached DPR retrieved documents in {self.preprocessed_file_name} jsonl file [{self.split.value} split] ...')
        newline_positions_in_file = []
        tmp_dict = dict()
        _lines = dict()
        starting_byte = 0
        with ZipFile(f"{self.checkpoint_path}/{self.preprocessed_file_name}", 'r') as zip_ref:
            with zip_ref.open(self.current_file) as fh:
                for line_id, line in tqdm(enumerate(fh)):
                    obj = json.loads(line)
                    k = obj['question']
                    tmp_dict[line_id] = k
                    newline_positions_in_file.append(starting_byte)
                    starting_byte = fh.tell()

        for obj_ind, obj in enumerate(newline_positions_in_file):
            _lines[tmp_dict[obj_ind]] = obj
        return _lines

    def fetch_dpr_context(self, question: str):
        if question not in self.lookup_lines:
            return []
        with ZipFile(f"{self.checkpoint_path}/{self.preprocessed_file_name}", 'r') as zip_ref:
            with zip_ref.open(self.current_file) as bigFile:
                bigFile.seek(self.lookup_lines[question])
                json_line = bigFile.readline().strip()
                return [DPRContext.convert(x).retriever_document for x in json.loads(json_line)['context']]

    @property
    def current_file(self):
        if self.dataset_name == "FACTOIDQA":
            return "data.jsonl"
        if self.split == DatasetSplit.TRAIN:
            return "train.jsonl"
        elif self.split == DatasetSplit.DEV:
            return "dev.jsonl"
        else:
            raise ValueError(f"Invalid split {self.split}")

    def retrieve(self,
                 query: str,
                 filters: Optional[FilterType] = None,
                 top_k: Optional[int] = None,
                 index: Optional[str] = None,
                 headers: Optional[Dict[str, str]] = None,
                 scale_score: Optional[bool] = None,
                 document_store: Optional[BaseDocumentStore] = None) -> List[Document]:
        return self.fetch_dpr_context(query)[:top_k]

    def retrieve_batch(self,
                       queries: List[str],
                       filters: Optional[Union[FilterType, List[Optional[FilterType]]]] = None,
                       top_k: Optional[int] = None,
                       index: Optional[str] = None,
                       headers: Optional[Dict[str, str]] = None,
                       batch_size: Optional[int] = None,
                       scale_score: Optional[bool] = None,
                       document_store: Optional[BaseDocumentStore] = None) -> List[List[Document]]:
        return [self.fetch_dpr_context(query)[:top_k] for query in queries]
