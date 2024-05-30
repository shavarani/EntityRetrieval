experiments=('Closed-Book' 'Open-Book' 'RePLUG')
retrievers=('None' 'DKRR' 'ANCE' 'DPR' 'BM25' 'Oracle' 'SpEL')
datasets=('FACTOIDQA' 'STRATEGYQA' 'EntityQuestions')
splits=('dev')
hf_model_names=('meta-llama/Llama-2-7b-hf' 'meta-llama/Llama-2-13b-hf' 'meta-llama/Llama-2-70b-hf')
retriever_top_k=4
for experiment in "${experiments[@]}"; do
  for retriever_type in "${retrievers[@]}"; do
    for dataset in "${datasets[@]}"; do
      for split in "${splits[@]}"; do
        for hf_model_name in "${hf_model_names[@]}"; do
          if [[ $experiment == "Closed-Book" && $retriever_type != "None" ]]; then
            continue
          fi
          if [[ $experiment != "Closed-Book" && $retriever_type == "None" ]]; then
            continue
          fi
          if [[ $experiment == "RePLUG" ]]; then
            mtype="RePLUG"
          else
            mtype="HFLLM"
          fi
          if [[ $hf_model_name == "meta-llama/Llama-2-70b-hf" ]]; then
            eight_bit=True
          else
            eight_bit=False
          fi
          if [[ $dataset == "STRATEGYQA" && $retriever_type == "Oracle" ]]; then
            continue
          fi
          if [[ $dataset == "STRATEGYQA" ]]; then
            max_tokens_to_generate=1
          else
            max_tokens_to_generate=10
          fi
          python3.10 main.py dataset-name="${dataset}" dataset-split="${split}" model-type="${mtype}" hf-model-name="${hf_model_name}" hf-llm-load-in-8bit="${eight_bit}" retriever-type="${retriever_type}" hf-max-tokens-to-generate="${max_tokens_to_generate}" retriever-top-k="${retriever_top_k}" experiment-name="${experiment}-${retriever_type}" prefetched-k-size=100 verbose-logging=False perform-annotation=True perform-evaluation=False

        done
      done
    done
  done
done
