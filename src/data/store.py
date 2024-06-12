
class StoreResult:
    """
    Stores the annotated datarecords as a jsonl file
    """
    def __init__(self, config) -> None:
        dataset_name = config['Dataset']['name'].lower()
        model_type = config['Model']['name'].lower()
        dataset_split = config['Dataset']['split'].lower()
        if model_type == 'hfllm' or model_type == 'hfllama':
            hf_model_name = config["Model"]["hf_model_name"].split("/")[-1]
            retriever_type = config["Model.Retriever"]["type"]
            quantized = config["Model"]["hf_llm_load_in_8bit"].lower() == 'true'
            top_k = config["Model.Retriever"]["retriever_top_k"]
            max_gen = config["Model"]["hf_max_tokens_to_generate"]
            k_size = int(config["Model.Retriever"]["prefetched_k_size"])
            experiment_desc = f'open_book_{retriever_type}_{k_size}_topk_{top_k}' if retriever_type.lower() != 'none' else 'closed_book'
            qant = '_8bitq' if quantized else ''
            experiment_desc =experiment_desc + f'_{hf_model_name}{qant}_max_gen_{max_gen}'
        elif model_type == 'replug':
            hf_model_name = config["Model"]["hf_model_name"].split("/")[-1]
            retriever_type = config["Model.Retriever"]["type"]
            top_k = config["Model.Retriever"]["retriever_top_k"]
            quantized = config["Model"]["hf_llm_load_in_8bit"].lower() == 'true'
            max_gen = config["Model"]["hf_max_tokens_to_generate"]
            k_size = int(config["Model.Retriever"]["prefetched_k_size"])
            qant = '_8bitq' if quantized else ''
            experiment_desc = f'open_book_{retriever_type}_{k_size}_topk_{top_k}'
            experiment_desc =experiment_desc + f'_{hf_model_name}{qant}_max_gen_{max_gen}'
        else:
            experiment_desc = (config['Experiment']['name']).replace(' ', '_').lower()
        self.out = open(f"{dataset_name}_{dataset_split}_{model_type}_{experiment_desc}.jsonl", 'w', encoding='utf-8')

    def store(self, record):
        # TODO log the question, answer and the ground truth in here!
        self.out.write(f"{str(record)}\n")

    def __del__(self):
        self.out.close()