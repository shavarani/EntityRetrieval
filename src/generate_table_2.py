from tqdm import tqdm
import numpy as np
import configparser
import json
from sklearn.metrics import accuracy_score

from eval import EvaluationMetrics
#setting_names = ['Closed-Book','DPR','RePLUG','EntityRetrieval-Oracle','EntityRetrieval-SpEL']
#model_names = ['Llama-2-7b-hf','Llama-2-13b-hf','Llama-2-70b-hf_8bQ']
#dataset = "FactoidQA"

def extract_entity_questions_answer(model_answer):
    model_answer = model_answer.strip().lower()
    return model_answer

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

def evaluate(filepath, dataset, split):
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
            elif dataset == 'EntityQuestions':
                if 'answer_aliases' not in annotated_record or not annotated_record['answer_aliases'] or \
                        annotated_record['answer_aliases'] == 'None':
                    continue
                model_answer = extract_entity_questions_answer(annotated_record['predicted_answer'])
                metrics.add_open_domain_prediction(
                    annotated_record['question'], annotated_record['answer_aliases'], model_answer)
            else:
                raise ValueError(f"Answer extractor undefined for: {filepath}")
        if dataset == 'StrategyQA':
            accuracy = accuracy_score(metrics.expected_answers, metrics.model_answers)
            invalid_count = metrics.model_answers.count('Wrong!')
            return {'accuracy': accuracy, 'invalid_count': invalid_count, 'exact_match': 'N/A', 'f1': 'N/A'}
        elif dataset in ['FactoidQA', 'EntityQuestions']:
            exact_match = np.mean([x['em'] for x in metrics.open_domain_predictions])
            f1 = np.mean([x['f1'] for x in metrics.open_domain_predictions])
            return {'accuracy': 'N/A', 'invalid_count': 'N/A', 'exact_match': exact_match, 'f1': f1}
beginnig = """\\begin{table}
\t\centering
\t\setlength{\\tabcolsep}{2.5pt}
\t\\begin{tabular}{l|cc|cc|cc}
\t\\toprule
\t\\multicolumn{1}{c|}{\\multirow{3}{*}{\\begin{tabular}[c]{@{}c@{}}\\textbf{LLaMA3} \\\\ \\textbf{(8B)}\\end{tabular}}} & \\multicolumn{2}{c|}{\\multirow{2}{*}{\\textbf{FactoidQA}}} & \\multicolumn{4}{c}{\\textbf{EntityQuestions}}                    \\\\ \\cmidrule{4-7} 
\t\\multicolumn{1}{c|}{}                             & \\multicolumn{2}{c|}{}                           & \\multicolumn{2}{c|}{\\textbf{dev}}     & \\multicolumn{2}{c}{\\textbf{test}} \\\\ \\cmidrule{2-7} 
\t\\multicolumn{1}{c|}{}                             & \\textbf{EM}           & \\multicolumn{1}{c|}{\\textbf{F1}}          & \\textbf{EM} & \\multicolumn{1}{c|}{\\textbf{F1}} & \\textbf{EM}          & \\textbf{F1}         \\\\ \\midrule
"""

endig= """\t\\bottomrule
\t\\end{tabular}
\t\\caption{Question answering efficacy comparison between Closed-book and Retrieval-augmentation using BM25, DPR, ANCE, and \\textit{Entity Retrieval}. EM refers to the exact match between predicted and expected answers, disregarding punctuation and articles (\\texttt{a}, \\texttt{an}, \\texttt{the}).}
\t\\label{tab:llama_3_8b_raqa_results_new}
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
for setting_name in ['Closed-Book', 'BM25', 'DPR', 'ANCE', 'Oracle50', 'Oracle100', 'Oracle300', 'Oracle1000', 'SpEL50', 'SpEL100', 'SpEL300', 'SpEL1000']:
    printing_line  = "\t\\multicolumn{1}{l|}{" + model_convertions[setting_name] + "} "
    for dataset, split in [('FactoidQA', 'train'), ('EntityQuestions', 'dev'), ('EntityQuestions', 'test')]:
        for model_name in ['Meta-Llama-3-8B']:
            results = []
            for experiment_id in range(1, 4):
                if dataset == "FactoidQA":
                    address = f"results/{dataset}/{experiment_id}/{setting_name}/{model_name}.jsonl"
                elif dataset == "EntityQuestions":
                    address = f"results/{dataset}/{experiment_id}/{setting_name}/{split}_{model_name}.jsonl"
                else:
                    raise ValueError(f"dataset {dataset} not used in this Table!")
                results.append(evaluate(address, dataset, split))
            if results[0]['f1'] != 'N/A':
                f1_values = [result['f1'] * 100 for result in results]
                average_f1 = np.mean(f1_values)
                std_dev_f1 = np.std(f1_values)
                margin_of_error_f1 = 2.576 * (std_dev_f1 / np.sqrt(len(f1_values)))
                exact_match_values = [result['exact_match'] * 100 for result in results]
                average_exact_match = np.mean(exact_match_values)
                std_dev_exact_match = np.std(exact_match_values)
                margin_of_error_exact_match = 2.576 * (std_dev_exact_match / np.sqrt(len(exact_match_values)))
                if margin_of_error_exact_match > 0.0 or margin_of_error_f1 > 0.0:
                    printing_line += f" & {average_exact_match:.1f}$\\pm${margin_of_error_exact_match:.1f} & {average_f1:.1f}$\\pm${margin_of_error_f1:.1f} "
                else:
                    printing_line += f" & {average_exact_match:.1f} & {average_f1:.1f} "
    printing_line += "\\\\"
    if setting_name == 'Closed-Book':
        printing_line += "\\midrule\n\t\\multicolumn{7}{c}{Retrieval-Augmented QA} \\\\ \\midrule"
    if setting_name == 'ANCE':
        printing_line += "\\midrule\n\t\\multicolumn{7}{c}{\\textit{Entity Retrieval} w/ Question Entity Annotations} \\\\ \\midrule"
    if setting_name == 'Oracle1000':
        printing_line += "\\midrule\n\t\\multicolumn{7}{c}{\\textit{Entity Retrieval} w/ \\textsc{SpEL} Entity Annotations} \\\\ \\midrule"
    print(printing_line)
print(endig)