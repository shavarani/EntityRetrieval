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

from data.loader import get_dataset
from model.retrievers.prefetched_retrieve import PrefetchedDocumentRetriever

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
    for item in tqdm(dataset):
        r = retriever.retrieve(query=item.question, top_k=max_k)
        for coverage in range(1, max_k+2):
            if any([x.meta['has_answer'] for x in r[:coverage]]):
                break
        if coverage < max_k + 1:
            retrieval_counter[coverage] += 1

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
                              ('EntityQuestions', 'dev', 22068), ('EntityQuestions', 'test', 22075)]
    retriever_configurations = [('bm25', 100)]
    for _dataset_, _split_, _dataset_length_ in dataset_configurations:
        for _retriever_type_, _retriever_prefetched_k_size_ in retriever_configurations:
            create_plot(_dataset_, _split_, _retriever_type_, _retriever_prefetched_k_size_, _dataset_length_)