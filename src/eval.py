from typing import List
import os
import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, f1_score
from oqaeval.eval import em_eval, f1_eval # pip install git+https://github.com/ehsk/OpenQA-eval.git
from oqaeval.data_utils import Question
from oqaeval.squad_evaluate import metric_max_over_ground_truths, normalize_answer

class EvaluationMetrics:
    def __init__(self, config, dataset, split):
        self.evaluate_rouge = config['Evaluate']['evaluate_rouge'].lower() == 'true'
        self.evaluate_bem = config['Evaluate']['evaluate_bem'].lower() == 'true'
        self.split = split
        self.dataset = dataset
        if dataset == 'strategy_qa' and split == 'dev':
            self.split = 'train_filtered (easy)'
        self.expected_answers = []
        self.model_answers = []
        self.open_domain_predictions = []

    def add_predictions(self, expected_answer, model_answer):
        self.expected_answers.append(expected_answer)
        self.model_answers.append(model_answer)

    def add_open_domain_prediction(self, question: str, gold_answers: List, candidate_answer: str):
        q = Question(question, gold_answers)
        self.open_domain_predictions.append({
            "em": em_eval(q, candidate_answer),
            "f1": f1_eval(q, candidate_answer),})

    def print_scores(self):
        accuracy = accuracy_score(self.expected_answers, self.model_answers)
        recall = recall_score(self.expected_answers, self.model_answers, average='macro', zero_division=0)
        f1 = f1_score(self.expected_answers, self.model_answers, average='macro', zero_division=0)
        print('\t'+'='*50)
        print(f'\tDataset: {self.dataset}, Split: {self.split}')
        print('\t'+'='*50)
        print('\t==== Accuracy: {:.2f}'.format(accuracy * 100))
        print('\t==== Recall: {:.2f}'.format(recall * 100))
        print('\t==== Macro F1: {:.2f}'.format(f1 * 100))
        count_na = self.model_answers.count('Wrong!')
        print(f'\t==== Incorrect answers count: {count_na} out of {len(self.model_answers)}')
        print('\t'+'='*50)

    def print_open_domain_eval_results(self):
        exact_match = np.mean([x['em'] for x in self.open_domain_predictions])
        f1 = np.mean([x['f1'] for x in self.open_domain_predictions])
        print('\t'+'='*50)
        print(f'\tDataset: {self.dataset}, Split: {self.split}')
        print('\t'+'='*50)
        print('\t==== Exact Match: {:.2f}'.format(exact_match * 100))
        print('\t==== F1 Score: {:.2f}'.format(f1 * 100))
        print('\t'+'='*50)

class QAEvaluate:
    def __init__(self, config):
        self.results_path = config['Evaluate']['experimental_results_path']
        self.config = config
    @staticmethod
    def extract_strategy_qa_answer(model_answer):
        model_answer = model_answer.strip().lower()
        if model_answer[:3] == 'yes':
            model_answer = 'Yes'
        elif model_answer[:2] == 'no':
            model_answer = 'No'
        else:
            model_answer = 'Wrong!'
        return model_answer

    @staticmethod
    def extract_factoid_qa_answer(model_answer):
        ma = model_answer.strip().lower()
        if ma[:3] == 'yes':
            model_answer = 'Yes'
        elif ma[:2] == 'no':
            model_answer = 'No'
        elif not model_answer:
            model_answer = 'N/A'
        return model_answer


    def analysis(self):
        correct_answers = {}
        incorrect_answers = {}
        total_systems = 0
        for filename in os.listdir(self.results_path):
            if filename.endswith('.jsonl'):
                filepath = os.path.join(self.results_path, filename)
                print(f'Processing {filepath} ...')
                total_systems += 1
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        annotated_record = json.loads(line)
                        dataset = annotated_record['dataset']
                        question = annotated_record['question']
                        expected_answer = annotated_record['answer']
                        answer_aliases = annotated_record.get('answer_aliases', [])
                        assert expected_answer or answer_aliases, f"Expected answer not found for: {filename}"
                        answer_aliases = [x.lower() for x in answer_aliases]
                        if dataset == 'strategy_qa':
                            model_answer = self.extract_strategy_qa_answer(annotated_record['predicted_answer'])
                        elif dataset == 'factoid_qa':
                            model_answer = self.extract_factoid_qa_answer(annotated_record['predicted_answer'])
                        else:
                            raise ValueError(f"Answer extractor undefined for: {filename}")
                        correct_answers.setdefault(question, 0)
                        incorrect_answers.setdefault(question, [])
                        assert correct_answers[question] + len(incorrect_answers[question]) == total_systems - 1, f"Question: {question}"
                        if model_answer.lower() in answer_aliases:
                            correct_answers[question] = correct_answers.get(question, 0) + 1
                        else:
                            incorrect_answers[question].append((filepath.split("/")[-1], model_answer, expected_answer.lower() if expected_answer else None))
                        assert correct_answers[question] + len(incorrect_answers[question]) == total_systems, f"Question: {question}"

        total_all_correct = len([question for question, count in correct_answers.items() if count == total_systems])
        print(f"{total_all_correct} questions received correct answers in all experimental settings.")
        total_none_correct = len([question for question, count in correct_answers.items() if count == 0])
        print(f"{total_none_correct} questions were incorrectly answered regardless of the experimental setting.")

        system_mistakes = {}

        for question, count in correct_answers.items():
            if 0 < count < total_systems:
                system_answers = incorrect_answers[question]
                for system, answer,expected_answer in system_answers:
                    system_mistakes.setdefault(system, [])
                    system_mistakes[system].append((question, answer, expected_answer))

        with open('system_mistakes.txt', 'w', encoding='utf-8') as f:
            f.write("-" * 80)
            for system, mistakes in system_mistakes.items():
                print(f"System: {system}")
                print(f"\tNumber of incorrectly answered questions: {len(mistakes)}")
                f.write(f"\nSystem: {system}\n")
                f.write("-" * 80)
                f.write(f"\nNumber of incorrectly answered questions: {len(mistakes)}\n")
                for question, answer, expected_answer in mistakes:
                    f.write(f"\nQuestion: {question}\n")
                    f.write(f"Answer: {answer}\n")
                    if expected_answer:
                        f.write(f"Expected answer: {expected_answer}\n")
                f.write("\n")
                f.write("-" * 80)

    def evaluate(self):
        print('Evaluating the results...')
        for filename in os.listdir(self.results_path):
            results = {}
            if filename.endswith('.jsonl'):
                filepath = os.path.join(self.results_path, filename)
                print(f'Processing {filepath} ...')
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in tqdm(f):
                        annotated_record = json.loads(line)
                        dataset = annotated_record['dataset']
                        split = annotated_record['split']
                        results.setdefault(dataset, {})
                        results[dataset].setdefault(split, EvaluationMetrics(self.config, dataset, split))
                        expected_answer = annotated_record['answer']
                        if dataset == 'strategy_qa':
                            model_answer = self.extract_strategy_qa_answer(annotated_record['predicted_answer'])
                            results[dataset][split].add_predictions(expected_answer, model_answer)
                        elif dataset == 'factoid_qa':
                            if 'answer_aliases' not in annotated_record or not annotated_record['answer_aliases'] or \
                                    annotated_record['answer_aliases'] == 'None':
                                continue
                            model_answer = self.extract_factoid_qa_answer(annotated_record['predicted_answer'])
                            results[dataset][split].add_open_domain_prediction(
                                annotated_record['question'], annotated_record['answer_aliases'], model_answer)
                        else:
                            raise ValueError(f"Answer extractor undefined for: {filename}")
            for dataset in results:
                for split in results[dataset]:
                    if dataset == 'strategy_qa':
                        results[dataset][split].print_scores()
                    elif dataset == 'factoid_qa':
                        results[dataset][split].print_open_domain_eval_results()
                    else:
                        raise ValueError(f"dataset: {dataset} undefined!")
