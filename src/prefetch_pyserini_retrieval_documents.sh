datasets=('FACTOIDQA' 'STRATEGYQA' 'EntityQuestions')
splits=('train' 'dev' 'test')
types=('bm25' 'dpr')
k=100

for dataset in "${datasets[@]}"; do
    for split in "${splits[@]}"; do
        for type in "${types[@]}"; do
          if [[ "$dataset" == "STRATEGYQA" && "$split" == "test" ]]; then
            continue
          fi     
          if [[ "$dataset" == "FACTOIDQA" && ("$split" == "test" || "$split" == "dev") ]]; then
            continue
          fi
          if [[ "$dataset" == "EntityQuestions" && "$split" == "train" ]]; then
            continue
          fi
          output_file="prefetched_retrieval_${dataset}_${split}_${type}_${k}.jsonl"
          python model/retrievers/prefetch_retrieval_pyserini.py --dataset "$dataset" --split "$split" --type "$type" --retriever_k "$k" --output_file "$output_file"
        done
    done
done