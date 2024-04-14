experiments=('Closed-Book' 'DPR' 'RePLUG' 'EntityRetrieval-Oracle' 'EntityRetrieval-SpEL')
datasets=('FACTOIDQA' 'STRATEGYQA')
splits=('dev')
hf_model_names=('meta-llama/Llama-2-7b-hf' 'meta-llama/Llama-2-13b-hf' 'meta-llama/Llama-2-70b-hf')
for experiment in "${experiments[@]}"; do
  for dataset in "${datasets[@]}"; do
    for split in "${splits[@]}"; do
      for hf_model_name in "${hf_model_names[@]}"; do
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
        if [[ $experiment == "Closed-Book" ]]; then
          use_retriever=False
        else
          use_retriever=True
        fi
        if [[ $experiment == "EntityRetrieval-SpEL" ]]; then
          dpr_question_model="spel_wiki_first_100_words"
        elif [[ $experiment == "EntityRetrieval-Oracle" ]]; then
          dpr_question_model="oracle_wiki_first_100_words"
        else
          dpr_question_model="single-nq"
        fi     
        if [[ $dataset == "STRATEGYQA" && $experiment == "EntityRetrieval-Oracle" ]]; then
          continue
        fi
        
        python3.10 main.py dataset-name="${dataset}" dataset-split="${split}" model-type="${mtype}" hf-model-name="${hf_model_name}" hf-llm-load-in-8bit="${eight_bit}" use-retriever="${use_retriever}" hf-max-tokens-to-generate=10 retriever-top-k=4 dpr-index-type=exact dpr-question-model="${dpr_question_model}" experiment-name="${experiment}" dpr-k-size=100 verbose-logging=False perform-annotation=True perform-evaluation=False
      
      done
    done
  done
done
