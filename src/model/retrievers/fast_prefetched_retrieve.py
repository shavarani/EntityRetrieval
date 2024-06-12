"""
A faster version of PrefetchedDocumentRetriever which caches everything initially and keeps all the retrieved documents
 in memory.
"""
import os
import pickle
from typing import List, Optional, Union, Dict
from tqdm import tqdm

from haystack import Document
from haystack.nodes.retriever import BaseRetriever
from haystack.document_stores import BaseDocumentStore
from haystack.schema import FilterType

from data.loader import get_dataset
from model.retrievers.prefetched_retrieve import PrefetchedDocumentRetriever



class FastPrefetchedDocumentRetriever(BaseRetriever):
    def __init__(self, config, topk):
        super().__init__()
        dataset_name = config['Dataset']['name']
        split = config["Dataset"]["split"]
        retriever_type = config["Model.Retriever"]["type"].lower()
        k_size = int(config["Model.Retriever"]["prefetched_k_size"])
        self.topk = topk
        self.config = config
        cache_directory = config["Experiment"]["checkpoint_path"] + "/cache"
        if not os.path.exists(cache_directory):
            os.makedirs(cache_directory)
        self.preprocessed_file_name = f"{cache_directory}/{dataset_name}_{split}_{retriever_type}_{k_size}_{topk}.cache.pkl"
        if not os.path.exists(self.preprocessed_file_name):
            print('Creating FastPrefetchedDocumentRetriever cache ...')
            self.prefetched_documents = {}
            dataset = get_dataset(self.config)
            retriever = PrefetchedDocumentRetriever(self.config)
            for record in tqdm(dataset):
                self.prefetched_documents[record.question] = retriever.retrieve(record.question, top_k=self.topk)
            with open(self.preprocessed_file_name, 'wb') as file:
                pickle.dump(self.prefetched_documents, file)
        else:
            with open(self.preprocessed_file_name, 'rb') as file:
                self.prefetched_documents = pickle.load(file)

    def fetch_documents(self, query):
        if query in self.prefetched_documents:
            return self.prefetched_documents[query]
        else:
            raise ValueError(f"Query \"{query}\" not found in pre-fetched documents!")
        
    def retrieve(self,
                 query: str,
                 filters: Optional[FilterType] = None,
                 top_k: Optional[int] = None,
                 index: Optional[str] = None,
                 headers: Optional[Dict[str, str]] = None,
                 scale_score: Optional[bool] = None,
                 document_store: Optional[BaseDocumentStore] = None) -> List[Document]:
        assert top_k is not None and top_k <= self.topk, f"Top-k should be less than or equal to {self.topk}"
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
        assert top_k is not None and top_k <= self.topk, f"Top-k should be less than or equal to {self.topk}"
        return [self.fetch_documents(query)[:top_k] for query in queries]