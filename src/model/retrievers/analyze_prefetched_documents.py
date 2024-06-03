"""
This script performs retrieval coverage analysis on the prefetched retrieval document indexes and creates visualization
 plots for the supported datasets in the project.
"""
import os
import pathlib
import json
import argparse
import configparser
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

from data.loader import get_dataset
from model.retrievers.prefetched_retrieve import PrefetchedDocumentRetriever

class NDCG:
    @staticmethod
    def dcg_at_k(r, k):
        r = np.asfarray(r)[:k]
        if r.size:
            return np.sum((2**r - 1) / np.log2(np.arange(2, r.size + 2)))
        return 0.

    @staticmethod
    def ndcg_at_k(r, k):
        k = min(len(r), k)
        dcg_max = NDCG.dcg_at_k(sorted(r, reverse=True), k)
        if not dcg_max:
            return 0.
        return NDCG.dcg_at_k(r, k) / dcg_max

    @staticmethod
    def score(relevance_scores_list, num_docs=100):
        ndcg_scores = []
        for relevance_scores in relevance_scores_list:
            ndcg_score = NDCG.ndcg_at_k(relevance_scores, num_docs)
            ndcg_scores.append(ndcg_score)
        return np.mean(ndcg_scores)

# The following function is not currently use in this script
def parse_args():
    _path = pathlib.Path(os.path.abspath(__file__)).parent.parent.parent / '..' / '.checkpoints'
    parser = argparse.ArgumentParser(description="Script to prefetch retrieval documents")
    parser.add_argument("--dataset",     type=str, help="Name of the dataset", default='FACTOIDQA')
    parser.add_argument("--split",       type=str, help="Split of the dataset", default="train")
    parser.add_argument("--type",        type=str, help="Type of the retriever", default="bm25")
    #parser.add_argument("--retriever_top_k", type=int, help="Number of documents returned for each question", default=4)
    parser.add_argument("--prefetched_k_size", type=int, help="Number of documents retrieved for each question", default=100)
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read_dict({
        'Dataset': {'name' : args.dataset, 'split': args.split},
        'Model.Retriever': {'type': args.type, 'prefetched_k_size': args.prefetched_k_size}, # 'retriever_top_k': args.retriever_top_k
        'Experiment': {'checkpoint_path': _path}
    })
    return config

def first_non_zero_index(arr):
    for index, value in enumerate(arr):
        if value != 0:
            return index
    return -1

def calculate_mrr(reciprocal_ranks):
    total_queries = len(reciprocal_ranks)
    total_rr = 0
    for rr in reciprocal_ranks:
        if rr == 0:
            total_rr += 0
        else:
            total_rr += 1 / rr
    mrr = total_rr / total_queries
    return mrr

def create_plot(dataset_name, split, retriever_type, retriever_prefetched_k_size, dataset_length):
    cfg = configparser.ConfigParser()
    cfg.read_dict({
        'Dataset': {'name' : dataset_name, 'split': split},
        'Model.Retriever': {'type': retriever_type, 'prefetched_k_size': retriever_prefetched_k_size},
        'Experiment': {'checkpoint_path': pathlib.Path(os.path.abspath(__file__)).parent.parent.parent / '..' / '.checkpoints'}
    })
    dataset = get_dataset(cfg)
    retriever = PrefetchedDocumentRetriever(cfg)
    max_k = int(cfg["Model.Retriever"]["prefetched_k_size"])
    # retrieval accuracy = percentage of retrieved passages that contain the answer.
    retrieval_counter = Counter()
    print('Collecting retrieval coverage data ...')
    relevance_scores_array = np.zeros((dataset_length, max_k), dtype=int)
    reciprocal_ranks = []
    for item_idx, item in enumerate(tqdm(dataset)):
        r = retriever.retrieve(query=item.question, top_k=max_k)
        if r:
            relevance_scores = [int(x.meta['has_answer']) for x in r]
            relevance_scores += [0] * (max_k - len(relevance_scores))
            relevance_scores_array[item_idx] = relevance_scores
            coverage = first_non_zero_index(relevance_scores) + 1
        else:
            coverage = 0
        reciprocal_ranks.append(coverage)
        if 0 < coverage < max_k + 1:
            retrieval_counter[coverage] += 1
    relevance_scores_list = relevance_scores_array.tolist()
    print('='*120)
    print(f'Calculating scores for {dataset_name}/{split}/{retriever_type}:')
    print('='*120)
    # Top-k Retrieval Accuracy as reported in https://aclanthology.org/2021.emnlp-main.496.pdf
    #  Top-K Retrieval Accuracy = \frac{\text{Number of queries with at least one relevant document in the top } k \text{ results}}{\text{Total number of queries}}
    # nDCG@k: nDCG@k (normalized Discounted Cumulative Gain at rank k) evaluates the quality of a ranking system by
    #  considering both the relevance and the position of documents in the top k results.
    #  It calculates the Discounted Cumulative Gain (DCG@k) by summing the relevance scores of documents, weighted by
    #  their position in the ranking. This score is then normalized by dividing it by the Ideal Discounted Cumulative
    #  Gain (IDCG@k), which represents the best possible ranking of documents. The resulting nDCG@k value ranges from 0
    #  to 1, where 1 indicates a perfect ranking with the most relevant documents appearing at the top.
    for k in [1, 2, 3, 4, 5, 20, max_k]:
        ndcg_score = NDCG.score(relevance_scores_list, k)
        print(f"NDCG@{k}: {ndcg_score:.4f}")
    for k in [1, 2, 3, 4, 5, 20, max_k]:
        raatk = sum([retrieval_counter[x] for x in range(1, k+1)]) * 100. / dataset_length
        print(f"RAcc@{k}: {raatk:.2f}")
    # Mean Reciprocal Rank: the average of the reciprocal ranks of the first relevant document for each query
    print(f"MRR: {calculate_mrr(reciprocal_ranks):.4f}")
    data = sorted(retrieval_counter.items(), key=lambda x:x[0])
    x  = [i[0] for i in data]
    y = [i[1] for i in data]
    cumulative_y = [sum(y[:i+1]) * 100. / dataset_length for i in range(len(y))]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
    ax1.bar(x, y, label='Frequency')
    ax1.set_yscale('log')
    ax1.set_ylabel('Count of Questions (log scale)')
    ax1.set_title(f'{dataset_name}-{split}-{retriever_type}')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.plot(x, cumulative_y, marker='.', color='black', label='Cumulative')
    ax2.set_xlabel('Index of the retrieved document where first hit identified')
    ax2.set_ylabel('Cumulative Coverage Percentage')
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))
    plt.tight_layout()
    plt.savefig(f'retrieval_coverage_{dataset_name}_{split}_{retriever_type}.png')
    with open(f'retrieval_coverage_data_{dataset_name}_{split}_{retriever_type}.json', 'w') as f:
        json.dump(dict(retrieval_counter), f)

if __name__ == '__main__':
    dataset_configurations = [('FACTOIDQA', 'train', 2203), ('STRATEGYQA', 'train', 2290), ('STRATEGYQA', 'dev', 2821),
                              ('EntityQuestions', 'dev', 4710), ('EntityQuestions', 'test', 4741)]
    retriever_configurations = [('bm25', 100), ('dpr', 100), ('ance', 100), ('dkrr', 100), ('spel', 50), ('spel', 100), ('spel', 300)]
    for _dataset_, _split_, _dataset_length_ in dataset_configurations:
        for _retriever_type_, _retriever_prefetched_k_size_ in retriever_configurations:
            create_plot(_dataset_, _split_, _retriever_type_, _retriever_prefetched_k_size_, _dataset_length_)