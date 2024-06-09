from typing import Union
from haystack.nodes.retriever import BaseRetriever

from model.retrievers.prefetched_retrieve import PrefetchedDocumentRetriever
from model.retrievers.fast_prefetched_retrieve import FastPrefetchedDocumentRetriever
from model.retrievers.realtime_retrieve import RealtimeDocumentRetriever


def get_retriever(config) -> Union[BaseRetriever, None]:
    retriever_type = config["Model.Retriever"]["type"]
    use_retriever = retriever_type.lower() != 'none'
    retriever_top_k = int(config["Model.Retriever"]["retriever_top_k"])
    retriever_load_in_memory = config['Model.Retriever']['load_in_memory'].lower() == 'true'
    retriever_realtime_retrieve = config['Model.Retriever']['realtime_retrieve'].lower() == 'true'
    if retriever_realtime_retrieve and use_retriever:
        return RealtimeDocumentRetriever(config)
    elif retriever_load_in_memory and use_retriever:
        return FastPrefetchedDocumentRetriever(config, topk=retriever_top_k)
    elif use_retriever:
        return PrefetchedDocumentRetriever(config)
    else:
        return None
