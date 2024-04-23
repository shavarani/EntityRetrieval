from tqdm import tqdm
import numpy as np
import configparser
import json
from sklearn.metrics import accuracy_score

from eval import EvaluationMetrics
setting_names = ['Closed-Book','DPR','RePLUG','EntityRetrieval-Oracle','EntityRetrieval-SpEL']
model_names = ['Llama-2-7b-hf','Llama-2-13b-hf','Llama-2-70b-hf_8bQ']
dataset = "FactoidQA"

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
beginnig = """\\begin{table*}
\t\centering
\t\\begin{tabular}{l|cc|cc|cc|cc}
\t\\multicolumn{1}{c|}{~}                                                                   & \\multicolumn{6}{c|}{LLaMA 2\\textsuperscript{$\\star$}} & \\multicolumn{2}{c}{\\multirow{2}{*}{GPT 4\\textsuperscript{$	\\ddagger$}}}\\\\\\cmidrule{2-7}
\t\\multicolumn{1}{c|}{Setting}                                                             & \\multicolumn{2}{c|}{7B} & \\multicolumn{2}{c|}{13B} & \\multicolumn{2}{c|}{70B-8bQ} & \\\\\\cmidrule{2-9}
\t\\multicolumn{1}{c|}{~}                                                                   &  EM  &  F1  &  EM  &  F1  &  EM  &  F1  &  EM  &  F1   \\\\\\midrule
"""

endig= """                                                   \\bottomrule
\t\\end{tabular}
\t\\caption{FactoidQA evaluation results. EM refers to the exact match between predicted and expected answers, disregarding punctuation and articles (\\texttt{a}, \\texttt{an}, \\texttt{the}). \\textsuperscript{$\\dagger$}\\textit{Entity Retrieval} with oracle results are not directly comparable to other approaches, as they leverage gold annotated entity links from the dataset. \\textsuperscript{$\\ddagger$}GPT experiments cost \\$207.4 USD. \\textsuperscript{$\\star$} Results represent the average of three runs, accompanied by a margin of error based on a 95\\% confidence interval.}
\t\\label{tab:factoidqa_evaluation_results}
\t\\vspace{-0.3cm}
\\end{table*}
"""
model_convertions = {
    'Closed-Book':'Closed-book',
    'DPR': 'DPR',
    'RePLUG': '\\textsc{RePlug}',
    'EntityRetrieval-Oracle': '\\textit{Entity Retrieval}  w/ \\textsc{SpEL}',
    'EntityRetrieval-SpEL': '\\textit{Entity Retrieval} w/ oracle entities\\textsuperscript{$\\dagger$}'
}
print(beginnig)
for split in ['dev']:
    for setting_name in setting_names:
        printing_line  = "                                                    " + model_convertions[setting_name]
        for model_name in model_names:
            if setting_name == 'RePLUG':
                    model_name += '_zero_shot'
            results = []
            for experiment_id in range(1, 4):
                address = f"results/{dataset}/{experiment_id}/{setting_name}/{model_name}.jsonl"
                results.append(evaluate(address, split))
            #print(f"Setting: {setting_name}, Model: {model_name}, Dataset: {dataset}, Split: {split}, Results:")
            if results[0]['f1'] != 'N/A':
                f1_values = [result['f1'] * 100 for result in results]
                average_f1 = np.mean(f1_values)
                std_dev_f1 = np.std(f1_values)
                margin_of_error_f1 = 2.576 * (std_dev_f1 / np.sqrt(len(f1_values)))
                exact_match_values = [result['exact_match'] * 100 for result in results]
                average_exact_match = np.mean(exact_match_values)
                std_dev_exact_match = np.std(exact_match_values)
                margin_of_error_exact_match = 2.576 * (std_dev_exact_match / np.sqrt(len(exact_match_values)))
                printing_line += f" & {average_exact_match:.1f}$\\pm${margin_of_error_exact_match:.1f} & {average_f1:.1f}$\\pm${margin_of_error_f1:.1f} " 
            if results[0]['accuracy'] != 'N/A':
                accuracy_values = [result['accuracy'] for result in results]
                average_accuracy = np.mean(accuracy_values)
                std_dev_accuracy = np.std(accuracy_values)
                margin_of_error_accuracy = 2.576 * (std_dev_accuracy / np.sqrt(len(accuracy_values)))
                print(f"Accuracy: {average_accuracy:.2f}±{margin_of_error_accuracy:.2f}")
            if results[0]['invalid_count'] != 'N/A':
                invalid_count_values = [result['invalid_count'] for result in results]
                average_invalid_count = np.mean(invalid_count_values)
                std_dev_invalid_count = np.std(invalid_count_values)
                margin_of_error_invalid_count = 2.576 * (std_dev_invalid_count / np.sqrt(len(invalid_count_values)))
                print(f"Invalid Count: {average_invalid_count:.2f}±{margin_of_error_invalid_count:.2f}")
        printing_line += " & - & - \\\\"
        print(printing_line)
print(endig)