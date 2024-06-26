"""
This is a standalone script that uses pre-built pyserini (https://github.com/castorini/pyserini) indexes to fetch and
 store top-k retrieval documents for each question in a defined dataset.

Extra documentation can be looked-up from pyserini/docs/prebuilt-indexes.md
    - Standard Lucene Indexes (for sparse retrieval)
        wikipedia-dpr-100w
    - Faiss Indexes
        wikipedia-dpr-100w.dpr-multi
        wikipedia-dpr-100w.dpr-single-nq
        wikipedia-dpr-100w.bpr-single-nq
        wikipedia-dpr-100w.ance-multi
        wikipedia-dpr-100w.dkrr-nq
        wikipedia-dpr-100w.dkrr-tqa
"""
import os
import re
import string
import json
import argparse
import configparser
import pathlib
import jsonlines
from tqdm import tqdm

from data.loader import get_dataset

try:
    from pyserini.search.lucene import LuceneSearcher
    from pyserini.search.faiss import FaissSearcher, DprQueryEncoder, DkrrDprQueryEncoder
except Exception:
    LuceneSearcher, FaissSearcher, DprQueryEncoder, DkrrDprQueryEncoder = None, None, None, None

DEVICE = 'cuda'

def instantiate_retriever(retriever_type, device):
    if retriever_type == "bm25":
        encoder = None
        searcher = LuceneSearcher.from_prebuilt_index('wikipedia-dpr-100w')
    elif retriever_type == "dpr":
        encoder = DprQueryEncoder('facebook/dpr-question_encoder-multiset-base', device=device)
        searcher = FaissSearcher.from_prebuilt_index('wikipedia-dpr-100w.dpr-multi', encoder)
    elif retriever_type == "ance":
        encoder = DprQueryEncoder('castorini/ance-dpr-question-multi', device=device)
        searcher = FaissSearcher.from_prebuilt_index('wikipedia-dpr-100w.ance-multi', encoder)
    elif retriever_type == "dkrr":
        encoder = DkrrDprQueryEncoder('castorini/dkrr-dpr-nq-retriever', device=device)
        searcher = FaissSearcher.from_prebuilt_index('wikipedia-dpr-100w.dkrr-nq', encoder)
    else:
        raise ValueError(f'Retriever: {retriever_type} not available!')
    return encoder, searcher

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
    parser.add_argument("--type",        type=str, help="Type of the retriever", default="bm25")
    parser.add_argument("--retriever_k", type=int, help="Number of documents retrieved for each question", default=100)
    parser.add_argument("--output_file", type=str, help="Output file name", default="output.jsonl")
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read_dict({
        'Dataset': {'name' : args.dataset, 'split': args.split},
        'Model.Retriever': {'type': args.type, 'retriever_top_k': args.retriever_k},
        'Experiment': {'checkpoint_path': _path, 'output_file': args.output_file}
    })
    return config

class PrefetchRetrievalDocuments:
    def __init__(self, config):
        super().__init__()
        self.retriever_type = config["Model.Retriever"]["type"].lower()
        self.checkpoint_path = config["Experiment"]["checkpoint_path"]
        self.k = int(config['Model.Retriever']['retriever_top_k'])
        os.environ["PYSERINI_CACHE"] = self.checkpoint_path
        self.encoder, self.searcher = instantiate_retriever(self.retriever_type, device=DEVICE)

    def _get_raw(self, element):
        if self.retriever_type == "bm25":
            return element.raw
        elif self.retriever_type in ["dpr", "ance", "dkrr"]:
            return self.searcher.doc(element.docid).raw()
        else:
            raise ValueError(f'Retriever: {self.retriever_type} not available!')

    def fetch_documents(self, question, answer_aliases):
        hits = self.searcher.search(question, k=self.k)
        results = []
        for i in range(0, self.k):
            element = hits[i]
            passage = json.loads(self._get_raw(element))['contents']
            title = passage.split("\n")[0]
            results.append({'id': element.docid,'rank': i + 1, 'title': title, 'text': passage,
             'score': str(element.score), 'has_answer': text_has_answer(answer_aliases, passage)})
        return results

    def fetch_batch_documents(self, batch_question_answers, threads):
        assert self.encoder is not None
        batch_questions = [x[0] for x in batch_question_answers]
        batch_answers = [x[1] for x in batch_question_answers]

        input_ids = self.encoder.tokenizer(batch_questions, return_tensors='pt', padding=True, truncation=True)
        input_ids.to(DEVICE)
        encoded_queries = self.encoder.model(input_ids["input_ids"]).pooler_output.detach().cpu().numpy()
        all_hits = self.searcher.batch_search(encoded_queries, q_ids=[str(i) for i, _ in enumerate(batch_questions)], k=self.k, threads=threads)
        all_results = []
        for hits, answer_aliases in zip(all_hits.values(), batch_answers):
            _results_ = []
            for i in range(0, self.k):
                element = hits[i]
                passage = json.loads(self._get_raw(element))['contents']
                title = passage.split("\n")[0]
                _results_.append({'id': element.docid,'rank': i + 1, 'title': title, 'text': passage,
                                'score': str(element.score), 'has_answer': text_has_answer(answer_aliases, passage)})
            all_results.append(_results_)
        return all_results, batch_questions

if __name__ == '__main__':
    # TODO get these in configs
    batch_size = 512
    threads_count = 32
    cfg = parse_args()
    dataset = get_dataset(cfg)
    retriever = PrefetchRetrievalDocuments(cfg)
    with jsonlines.open(cfg['Experiment']['output_file'], mode='w') as writer:
        if cfg["Model.Retriever"]["type"] == "bm25":
            for e in tqdm(dataset):
                writer.write({"question": e.question, "context": retriever.fetch_documents(e.question, e.answer_aliases)})
        else:
            batch = []
            for item in tqdm(dataset):
                if len(batch) < batch_size:
                    batch.append((item.question, item.answer_aliases))
                else:
                    all_res, bq = retriever.fetch_batch_documents(batch, threads=threads_count)
                    for r, q in zip(all_res, bq):
                        writer.write({"question": q, "context": r})
                    del batch[:]
            if batch:
                all_res, bq = retriever.fetch_batch_documents(batch, threads=threads_count)
                for r, q in zip(all_res, bq):
                    writer.write({"question": q, "context": r})