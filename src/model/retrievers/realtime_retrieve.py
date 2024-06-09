"""
This class is implemented to provide realtime retrieval functionality for retrieval-augmented question answering.
"""
from typing import List, Optional, Union, Dict
from haystack import Document
from haystack.nodes.retriever import BaseRetriever
from haystack.document_stores import BaseDocumentStore
from haystack.schema import FilterType

from model.retrievers.prefetched_retrieve import RetrievedContext
from model.retrievers.prefetch_retrieval_pyserini import PrefetchRetrievalDocuments
try:
    from model.retrievers.prefetch_retrieval_entity_linking import FetchEntityRetrievalDocuments
except Exception:
    FetchEntityRetrievalDocuments = None


class RealtimeDocumentRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__()
        self.checkpoint_path = config["Experiment"]["checkpoint_path"]
        self.retriever_type = config["Model.Retriever"]["type"].lower()
        if self.retriever_type in ["bm25", "dpr", "ance", "dkrr"]:
            self.backend_retriever = PrefetchRetrievalDocuments(config)
        elif self.retriever_type == "spel":
            self.backend_retriever = FetchEntityRetrievalDocuments(config)
        else:
            raise ValueError(f"Undefined retriever type: {self.retriever_type}!")

    def fetch_documents(self, question: str):
        return [RetrievedContext.convert(x).retriever_document for x in
                self.backend_retriever.fetch_documents(question, [])]

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