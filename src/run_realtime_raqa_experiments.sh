retriever_type="BM25"
python3.10 main.py dataset-name=FACTOIDQA dataset-split=dev model-type=HFLLM hf-model-name=meta-llama/Meta-Llama-3-8B hf-llm-load-in-8bit=False retriever-type="${retriever_type}" hf-max-tokens-to-generate=10 retriever-top-k=4 experiment-name="Open-Book-${retriever_type}" prefetched-k-size=4 verbose-logging=False perform-annotation=True perform-evaluation=False retriever-load-in-memory=True retriever-realtime-retrieve=True

retriever_type="ANCE"
python3.10 main.py dataset-name=FACTOIDQA dataset-split=dev model-type=HFLLM hf-model-name=meta-llama/Meta-Llama-3-8B hf-llm-load-in-8bit=False retriever-type="${retriever_type}" hf-max-tokens-to-generate=10 retriever-top-k=4 experiment-name="Open-Book-${retriever_type}" prefetched-k-size=4 verbose-logging=False perform-annotation=True perform-evaluation=False retriever-load-in-memory=True retriever-realtime-retrieve=True

retriever_type="SpEL"
echo 'roberta-large' > base_model.cfg
python3.10 main.py dataset-name=FACTOIDQA dataset-split=dev model-type=HFLLM hf-model-name=meta-llama/Meta-Llama-3-8B hf-llm-load-in-8bit=False retriever-type="${retriever_type}" hf-max-tokens-to-generate=10 retriever-top-k=4 experiment-name="Open-Book-${retriever_type}" prefetched-k-size=4 verbose-logging=False perform-annotation=True perform-evaluation=False retriever-load-in-memory=True retriever-realtime-retrieve=True max-w=100