experiments=('Closed-Book' 'Open-Book' 'RePLUG')
retrievers=('None' 'DKRR' 'ANCE' 'DPR' 'BM25' 'Oracle' 'SpEL')
datasets=('FACTOIDQA' 'EntityQuestions')
splits=('dev')
retriever_top_k=4
max_tokens_to_generate=10
for experiment in "${experiments[@]}"; do
  for retriever_type in "${retrievers[@]}"; do
    for dataset in "${datasets[@]}"; do
      for split in "${splits[@]}"; do
          if [[ $experiment == "Closed-Book" && $retriever_type != "None" ]]; then
            continue
          fi
          if [[ $experiment != "Closed-Book" && $retriever_type == "None" ]]; then
            continue
          fi

          python3.10 main.py dataset-name="${dataset}" dataset-split="${split}" model-type=OpenAI retriever-type="${retriever_type}" hf-max-tokens-to-generate="${max_tokens_to_generate}" retriever-top-k="${retriever_top_k}" experiment-name="${experiment}-${retriever_type}" prefetched-k-size=100 verbose-logging=False perform-annotation=True perform-evaluation=False retriever-load-in-memory=True

      done
    done
  done
done
