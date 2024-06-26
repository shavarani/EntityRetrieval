# Entity Retrieval: Entity Linking for Answering Entity-Centric Questions

This repository contains the implementation and the experiments from our paper 
"Entity Retrieval: Entity Linking for Answering Entity-Centric Questions", currently under review for EMNLP 2024.

<p align="center" width="100%"><img src="DPRvsER.png" width="92%" alt="DPR vs. Entity Retrieval"></p>

## Quick Start

The implemented source code is stored under `src` directory. 
### How to add New Datastes
You can start by looking at the `data` package which
contains data readers for the datasets on which we have conducted our experiments.
To add a new dataset you can easily implement a new class which implements `data.loaders.utils.QADataset`. The most 
important functions to be implemented are `__iter__` and `__next__`. The `__next__` method must return instances of 
`data.loaders.utils.QARecord`. Once implemented add the new dataset to `data.loader.get_dataset` and you should be 
able to run all the experiments with the new dataset flawlessly.

#### Creation of the EntityQuestions dataset
We have used the following script to create the selected EntityQuestions dataset which we have used in the paper:

```python
import json
from zipfile import ZipFile
from model.entity_linking.spel_vocab_to_wikipedia import SpELVocab2Wikipedia
lookup_index = SpELVocab2Wikipedia()
zip_ref = ZipFile("entity_questions.zip", 'r')
for c_file in ["train.jsonl", "dev.jsonl", "test.jsonl"]:
    data = zip_ref.open(c_file)
    with open(c_file, 'w') as f:
        selected_size = 0
        all_size = 0
        for line in data:
            record = json.loads(line)
            entity = record['entity'].replace(' ', '_')
            all_size += 1
            if entity in lookup_index.spel_vocab2wikipedia_lines:
                f.write(f"{json.dumps(record)}\n")
                selected_size += 1
        print(f"Selected {selected_size} out of {all_size} from {c_file}")    
zip_ref.close()
```

### How to create pre-fetched retrieval repositories
As mentioned in the paper, we treat document retrieval as a pre-processing step, caching the most relevant passages 
for each question - considering different retrieval techniques - before conducting the question answering experiments.
[`prefetch_pyserini_retrieval_documents.sh`](src/prefetch_pyserini_retrieval_documents.sh) located directly under `src` directory, creates all such pre-fetched passages and 
stores them for re-use in experiments. You may use the same script with slight modifications for creating pre-processed
passage repositories for your newly added datasets. Please note that the underlying implementation for these processes
uses [PySerini](https://github.com/castorini/pyserini) and loads pre-built FAISS indexes for dense retrieval techniques. Such processes can require up to 65GBs
of main memory and about the same amount of available disk storage to download and store the index.

For entity retrieval prefetched documents, you may follow the same procedure using [`prefetch_entity_retrieval_documents.sh`](src/prefetch_entity_retrieval_documents.sh). 
Please remember to instantiate the entity linker directory using `git submodule update --init --recursive` before running this script.

### Replicating the Results Tables and Figures
You can run [`analyze_prefetched_documents.py`](src/model/retrievers/analyze_prefetched_documents.py) to reproduce retrieval 
coverage analysis results using which you can recreate Figures 3 and 4 and Table 1.

As well, the results of Tables 2 and 3 can be replicated using [`run_llama_raqa_experiments.sh`](src/run_llama_raqa_experiments.sh), 
and the realtime efficiency experiments of Table 4 can be replicated using [`run_realtime_raqa_experiments.sh`](src/run_realtime_raqa_experiments.sh).
Once your script finishes, you may use [`organize_and_pack_up_experimental_results.sh`](src/organize_and_pack_up_experimental_results.sh) 
to organize the created result files into proper folders.
