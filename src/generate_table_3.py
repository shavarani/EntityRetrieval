import numpy as np
import configparser
import json
from sklearn.metrics import accuracy_score

from eval import EvaluationMetrics
dataset = "StrategyQA"

def extract_strategy_qa_answer(model_answer):
    model_answer = model_answer.strip().lower()
    if model_answer[:3] == 'yes':
        model_answer = 'Yes'
    elif model_answer[:2] == 'no':
        model_answer = 'No'
    else:
        model_answer = 'Wrong!'
    return model_answer

def extract_factoid_qa_answer(model_answer):
    ma = model_answer.strip().lower()
    if ma[:3] == 'yes':
        model_answer = 'Yes'
    elif ma[:2] == 'no':
        model_answer = 'No'
    elif not model_answer:
        model_answer = 'N/A'
    return model_answer

def evaluate(filepath, split):
    cfg = configparser.ConfigParser()
    cfg.read_dict({'Evaluate': {'evaluate_rouge': False, 'evaluate_bem': False}})
    metrics = EvaluationMetrics(cfg, dataset, split)
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            annotated_record = json.loads(line)
            expected_answer = annotated_record['answer']
            if dataset == 'StrategyQA':
                model_answer = extract_strategy_qa_answer(annotated_record['predicted_answer'])
                metrics.add_predictions(expected_answer, model_answer)
            elif dataset == 'FactoidQA':
                if 'answer_aliases' not in annotated_record or not annotated_record['answer_aliases'] or \
                        annotated_record['answer_aliases'] == 'None':
                    continue
                model_answer = extract_factoid_qa_answer(annotated_record['predicted_answer'])
                metrics.add_open_domain_prediction(
                    annotated_record['question'], annotated_record['answer_aliases'], model_answer)
            else:
                raise ValueError(f"Answer extractor undefined for: {filepath}")
        if dataset == 'StrategyQA':
            accuracy = accuracy_score(metrics.expected_answers, metrics.model_answers)
            invalid_count = metrics.model_answers.count('Wrong!')
            return {'accuracy': accuracy, 'invalid_count': invalid_count, 'exact_match': 'N/A', 'f1': 'N/A'}
        elif dataset == 'FactoidQA':
            exact_match = np.mean([x['em'] for x in metrics.open_domain_predictions])
            f1 = np.mean([x['f1'] for x in metrics.open_domain_predictions])
            return {'accuracy': 'N/A', 'invalid_count': 'N/A', 'exact_match': exact_match, 'f1': f1}
beginnig = """\\begin{table}
\t\\centering
\t\\begin{tabular}{l|ll|ll}
\t\\toprule
\t\\multicolumn{1}{c|}{\\multirow{3}{*}{\\begin{tabular}[c]{@{}c@{}}\\textbf{LLaMA3} \\\\ \\textbf{(8B)}\\end{tabular}}} & \\multicolumn{2}{c|}{\\textbf{train}}                            & \\multicolumn{2}{c}{\\textbf{train\\_filtered}}                  \\\\ \\cmidrule{2-5} 
\t\\multicolumn{1}{c|}{} & \\multicolumn{1}{c}{\\textbf{Acc.}} & \\multicolumn{1}{c|}{\\textbf{Inv \\#}} & \\multicolumn{1}{c}{\\textbf{Acc.}} & \\multicolumn{1}{c}{\\textbf{Inv \\#}} \\\\ \\midrule
"""

endig= """
\t\\bottomrule
\t\\end{tabular}
\t\\caption{Comparison of \\textit{Entity Retrieval} using \\textsc{SpEL} identified entities to the best-performing dense and sparse retrieval methods of Table \\ref{tab:llama_3_8b_raqa_results} on the StrategyQA dataset. Given the expected boolean results for StrategyQA questions, we restricted LLaMA 3 to generate only one token. \\textit{Acc.} indicates the fraction of answers that correctly match the expected Yes or No responses in the dataset, while \\textit{Inv \\#} represents the count of labels that are neither Yes nor No, but another invalid answer.}
\t\\label{tab:llama_3_8b_strategyqa_results_new}
\\end{table}
"""
model_convertions = {
    'Closed-Book':'Closed-book',
    'DPR': 'DPR',
    'BM25': 'BM25',
    'ANCE': 'ANCE',
    'SpEL50': 'ERSp50w',
    'SpEL100': 'ERSp100w',
    'SpEL300': 'ERSp300w',
    'SpEL1000': 'ERSp1000w',
    'Oracle50': 'ER50w',
    'Oracle100': 'ER100w',
    'Oracle300': 'ER300w',
    'Oracle1000': 'ER1000w',
}
print(beginnig)
for setting_name in ['BM25', 'ANCE', 'SpEL50', 'SpEL100', 'SpEL300', 'SpEL1000']:
    printing_line  = "\t\\multicolumn{1}{l|}{" + model_convertions[setting_name] + "} "
    for split in ['train', 'train_filtered']:
        for model_name in ['Meta-Llama-3-8B']:
            results = []
            for experiment_id in range(1, 4):
                address = f"results/{dataset}/{experiment_id}/{setting_name}/{split}_{model_name}.jsonl"
                results.append(evaluate(address, split))
            if results[0]['accuracy'] != 'N/A':
                accuracy_values = [result['accuracy'] *100 for result in results]
                average_accuracy = np.mean(accuracy_values)
                std_dev_accuracy = np.std(accuracy_values)
                margin_of_error_accuracy = 2.576 * (std_dev_accuracy / np.sqrt(len(accuracy_values)))
                invalid_count_values = [result['invalid_count'] for result in results]
                average_invalid_count = np.mean(invalid_count_values)
                std_dev_invalid_count = np.std(invalid_count_values)
                margin_of_error_invalid_count = 2.576 * (std_dev_invalid_count / np.sqrt(len(invalid_count_values)))
                if margin_of_error_accuracy > 0.0 or margin_of_error_invalid_count > 0.0:
                    printing_line += f" & {average_accuracy:.1f}$\\pm${margin_of_error_accuracy:.1f} & {int(average_invalid_count)}$\\pm${int(margin_of_error_invalid_count)} "
                else:
                    printing_line += f" & {average_accuracy:.1f} & {int(average_invalid_count)}"
    printing_line += " \\\\"
    if setting_name == "ANCE":
        printing_line += "\\midrule"
    print(printing_line)
print(endig)