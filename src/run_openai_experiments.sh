experiments=('Closed-Book' 'DPR' 'EntityRetrieval-Oracle' 'EntityRetrieval-SpEL')
datasets=('FACTOIDQA')
splits=('dev')
for experiment in "${experiments[@]}"; do
  for dataset in "${datasets[@]}"; do
    for split in "${splits[@]}"; do
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

        python3.10 main.py dataset-name="${dataset}" dataset-split="${split}" model-type=OpenAI use-retriever="${use_retriever}" hf-max-tokens-to-generate=10 retriever-top-k=4 dpr-index-type=exact dpr-question-model="${dpr_question_model}" experiment-name="${experiment}" dpr-k-size=100 verbose-logging=False perform-annotation=True perform-evaluation=False

    done
  done
done
