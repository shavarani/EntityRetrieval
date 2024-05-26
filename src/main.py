"""
To run the code, use can both use a customized configuration file or pass the arguments directly to the script.
"""
import sys
import os
import pathlib
import configparser
from tqdm import tqdm
import numpy as np
from data.loader import get_dataset
from data.store import StoreResult
from model.loader import get_llm
from eval import QAEvaluate, EvaluationMetrics

sys.path.append("src")

def read_configs_from_args(args):
    config = {}
    default_config = {
        'Dataset': {'name': 'FACTOIDQA', 'split': 'dev'}, 
        'Model': {'name': 'HFLLM', 'hf_model_name': 'TinyLlama/TinyLlama-1.1B-step-50K-105b', 'hf_max_tokens_to_generate': '10', 'hf_llm_load_in_8bit': 'False'}, 
        'Model.Retriever': {'type': 'dpr', 'use_retriever': 'False', 'retriever_top_k': '4', 'dpr_index_type': 'exact', 'dpr_question_model': 'single-nq', 'prefetched_k_size': '100'},
        'Experiment': {'name': 'experiment description', 'summarize_results': 'False', 'verbose_logging': 'False', 'perform_annotation': 'False'}, 
        'Evaluate': {'experimental_results_path': '../results/Table1/', 'evaluate_rouge': 'False', 'evaluate_bem': 'False', 'perform_evaluation': 'True'}
        }
    config.update(default_config)
    for arg in args:
        if '=' in arg:
            key, value = arg.split('=')
            if key == 'dataset-name':
                config['Dataset']['name'] = value
            elif key == 'dataset-split':
                config['Dataset']['split'] = value
            elif key == 'model-type':
                config['Model']['name'] = value
            elif key == 'hf-model-name':
                config['Model']['hf_model_name'] = value
            elif key == 'hf-max-tokens-to-generate':
                config['Model']['hf_max_tokens_to_generate'] = value
            elif key == 'hf-llm-load-in-8bit':
                config['Model']['hf_llm_load_in_8bit'] = value
            elif key == 'retriever-type':
                config['Model.Retriever']['type'] = value
            elif key == 'use-retriever':
                config['Model.Retriever']['use_retriever'] = value
            elif key == 'retriever-top-k':
                config['Model.Retriever']['retriever_top_k'] = value
            elif key == 'dpr-index-type':
                config['Model.Retriever']['dpr_index_type'] = value
            elif key == 'dpr-question-model':
                config['Model.Retriever']['dpr_question_model'] = value
            elif key == 'dpr-k-size':
                config['Model.Retriever']['prefetched_k_size'] = value
            elif key == 'experiment-name':
                config['Experiment']['name'] = value
            elif key == 'summarize-results':
                config['Experiment']['summarize_results'] = value
            elif key == 'verbose-logging':
                config['Experiment']['verbose_logging'] = value
            elif key == 'perform-annotation':
                config['Experiment']['perform_annotation'] = value
            elif key == 'experimental-results-path':
                config['Evaluate']['experimental_results_path'] = value
            elif key == 'evaluate-rouge':
                config['Evaluate']['evaluate_rouge'] = value
            elif key == 'evaluate-bem':
                config['Evaluate']['evaluate_bem'] = value
            elif key == 'perform-evaluation':
                config['Evaluate']['perform_evaluation'] = value
    return config

def main():
    path_ =  pathlib.Path(os.path.abspath(__file__)).parent / '..' / '.checkpoints'
    if not os.path.exists(path_):
        os.mkdir(path_)
    config = configparser.ConfigParser()
    if len(sys.argv) == 2 and sys.argv[1].endswith('.ini'):
        config.read(sys.argv[1])
    else:
        config.read_dict(read_configs_from_args(sys.argv[1:]))
    config['Experiment']['checkpoint_path'] = str(path_)
    if config['Experiment']['perform_annotation'].lower() == 'true':
        dataset = get_dataset(config)
        output = StoreResult(config)
        llm = get_llm(config)
        metrics = EvaluationMetrics(config, config['Dataset']['name'], config['Dataset']['split'])
        itr = tqdm(dataset)
        for record in itr:
            llm.annotate(record, summarize=config['Experiment']['summarize_results'].lower() == 'true', 
                         verbose=config['Experiment']['verbose_logging'].lower() == 'true')
            metrics.add_open_domain_prediction(
                record.question, record.answer_aliases, record.predicted_answer)
            exact_match = np.mean([x['em'] for x in metrics.open_domain_predictions]) * 100
            f1 = np.mean([x['f1'] for x in metrics.open_domain_predictions]) * 100
            itr.set_description(f"EM: {exact_match:.1f} F1: {f1:.1f}")
            output.store(record)
    if config['Evaluate']['perform_evaluation'].lower() == 'true':
        evaluator = QAEvaluate(config)
        evaluator.evaluate()
    
if __name__ == "__main__":
    main()
    