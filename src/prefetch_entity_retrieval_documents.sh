datasets=('FACTOIDQA' 'STRATEGYQA' 'EntityQuestions')
splits=('train' 'dev' 'test')
types=('spel')
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
          output_file="prefetched_entity_retrieval_${dataset}_${split}_${type}_${k}.jsonl"
          python model/retrievers/prefetch_retrieval_entity_linking.py --dataset "$dataset" --split "$split" --type "$type" --max_w "$k" --output_file "$output_file"
        done
    done
done