experiments=('Closed-Book' 'Open-Book') # 'RePLUG'
retrievers=('None' 'BM25' 'DPR' 'ANCE' 'Oracle' 'SpEL')
datasets=('FACTOIDQA' 'EntityQuestions' 'STRATEGYQA')
splits=('train' 'dev' 'test')
# hf_model_names=('meta-llama/Llama-2-7b-hf' 'meta-llama/Llama-2-13b-hf' 'meta-llama/Llama-2-70b-hf')
hf_model_names=('meta-llama/Meta-Llama-3-8B' 'meta-llama/Meta-Llama-3-70B')
retriever_top_k=4
pre_fetched_k=(50 100 300 1000)
for experiment in "${experiments[@]}"; do
  for retriever_type in "${retrievers[@]}"; do
    for dataset in "${datasets[@]}"; do
      for split in "${splits[@]}"; do
        for pf_k in "${pre_fetched_k[@]}"; do
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
            if [[ $hf_model_name == "meta-llama/Llama-2-70b-hf" || $hf_model_name == "meta-llama/Meta-Llama-3-70B" ]]; then
              eight_bit=True
            else
              eight_bit=False
            fi
            if [[ $retriever_type != "Oracle" && $retriever_type != "SpEL" && $pf_k != 100 ]]; then
              continue
            fi
            if [[ $dataset != "EntityQuestions" && $split == "test" ]]; then
              continue
            fi
            if [[ $dataset != "STRATEGYQA" && $split == "train" ]]; then
              continue
            fi
            if [[ $dataset == "STRATEGYQA" && $retriever_type == "Oracle" ]]; then
              continue
            fi
            if [[ $dataset == "STRATEGYQA" ]]; then
              max_tokens_to_generate=1
            else
              max_tokens_to_generate=10
            fi
            python3.10 main.py dataset-name="${dataset}" dataset-split="${split}" model-type="${mtype}" hf-model-name="${hf_model_name}" hf-llm-load-in-8bit="${eight_bit}" retriever-type="${retriever_type}" hf-max-tokens-to-generate="${max_tokens_to_generate}" retriever-top-k="${retriever_top_k}" experiment-name="${experiment}-${retriever_type}" prefetched-k-size="${pf_k}" verbose-logging=False perform-annotation=True perform-evaluation=False retriever-load-in-memory=True
          done
        done
      done
    done
  done
done
