"""
The main objective of implementation of this class is to cope with the hardware complexities of loading a retriever
 trained over the entire Wikipedia while experimenting on Question Answering experiments.
For this, we abstract the retrieval process by caching the top-k highly ranked documents for each question in each of
our supported QA datasets. We use the pre-built indexes in pyserini which are built using the 21_015_325 wikipedia
 passages (non-overlapping text blocks of size 100 words) in https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz .
We have run our dumping script in src/prefetch_retrieval_documents.sh and have stored the created dump URLS in PREPROCESSED_URLS.
You can create the dumps for newly implemented datasets using the same script.
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

# in the following oracle and spel retriever types refer to documents that are collected as the first 100 words of the
#  wikipedia articles of the salient entities in questions which have either been gold annotated (oracle) or identified
#   using spel entity linking method (spel).
PREPROCESSED_URLS = {
    "FACTOIDQA_oracle_varying.zip":"EaFkI0ecTM1As54ocMo5BFEBLBRsa7k1iWdQc3bpcrgEKA?e=afVrfM",

    "FACTOIDQA_spel_varying.zip":  "EaMFthYtZYtDn4GovmztYRcBDzBS3gac5ZqC4GlfTyPlpw?e=exwjKi",
    "STRATEGYQA_spel_varying.zip": "EdiDKQSjTY5EkPD3LqzJD8UBnLF8Fn4_AGl3Xru9gZWIeA?e=zXVnrX",

    "EntityQuestions_bm25_100.zip":"EdpGpxLIy-NLkfvQ0eurplwBroLsQFsfQs_hjrN2FrJp9g?e=OocWX3",
    "STRATEGYQA_bm25_100.zip":     "EVMmR908BepNssLfP8sFXacBbDFQHVpBUrr62MOtJYGVIQ?e=78Ahsv",
    "FACTOIDQA_bm25_100.zip":      "EROATmeR6QNEvCARa8VlG2cBo7COMpnvTsOm6yHRRnkLFQ?e=5QnQTX",
}

class RetrievedContext:
    def __init__(self):
        self.id = None
        self.rank = None
        self.title = None
        self.text = None
        self.score = None
        self.has_answer = None

    @staticmethod
    def convert(record):
        self = RetrievedContext()
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

class PrefetchedDocumentRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__()
        dataset_name = config['Dataset']['name']
        retriever_type = config["Model.Retriever"]["type"]
        k_size = int(config["Model.Retriever"]["prefetched_k_size"])
        if retriever_type in ['spel', 'oracle']:
            k_size = 'varying' # the maximum number of documents depends on the number of salient entities in question.
        self.preprocessed_file_name = f"{dataset_name}_{retriever_type}_{k_size}.zip"
        if self.preprocessed_file_name in PREPROCESSED_URLS:
            _url = PREPROCESSED_URLS[self.preprocessed_file_name]
        else:
            raise ValueError(
                f"Invalid prefetched retriever configuration setting ({retriever_type}/{k_size}) for {dataset_name}!")
        self.dataset_url = f"https://1sfu-my.sharepoint.com/:u:/g/personal/sshavara_sfu_ca/{_url}&download=1"
        self.checkpoint_path = config["Experiment"]["checkpoint_path"]
        download_public_file(self.dataset_url, f"{self.checkpoint_path}/{self.preprocessed_file_name}")
        self.split = DatasetSplit.from_str(config["Dataset"]["split"])
        self.dataset_name = dataset_name
        self.lookup_lines = self.scan_data_file()

    @property
    def current_file(self):
        if self.dataset_name == "FACTOIDQA":
            return "data.jsonl"
        if self.split == DatasetSplit.TRAIN:
            return "train.jsonl"
        elif self.split == DatasetSplit.DEV:
            return "dev.jsonl"
        elif self.split == DatasetSplit.TEST:
            return "test.jsonl"
        else:
            raise ValueError(f"Invalid split {self.split}")

    def scan_data_file(self):
        """
        loads the questions along with their line number in the downloaded file into a dictionary.
        """
        print(f'Scanning/Indexing prefetched retrieval documents in {self.preprocessed_file_name} jsonl file '
              f'[{self.split.value} split] ...')
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

    def fetch_documents(self, question: str):
        if question not in self.lookup_lines:
            return []
        with ZipFile(f"{self.checkpoint_path}/{self.preprocessed_file_name}", 'r') as zip_ref:
            with zip_ref.open(self.current_file) as bigFile:
                bigFile.seek(self.lookup_lines[question])
                json_line = bigFile.readline().strip()
                return [RetrievedContext.convert(x).retriever_document for x in json.loads(json_line)['context']]

    def retrieve(self,
                 query: str,
                 filters: Optional[FilterType] = None,
                 top_k: Optional[int] = None,
                 index: Optional[str] = None,
                 headers: Optional[Dict[str, str]] = None,
                 scale_score: Optional[bool] = None,
                 document_store: Optional[BaseDocumentStore] = None) -> List[Document]:
        return self.fetch_documents(query)[:top_k]

    def retrieve_batch(self,
                       queries: List[str],
                       filters: Optional[Union[FilterType, List[Optional[FilterType]]]] = None,
                       top_k: Optional[int] = None,
                       index: Optional[str] = None,
                       headers: Optional[Dict[str, str]] = None,
                       batch_size: Optional[int] = None,
                       scale_score: Optional[bool] = None,
                       document_store: Optional[BaseDocumentStore] = None) -> List[List[Document]]:
        return [self.fetch_documents(query)[:top_k] for query in queries]